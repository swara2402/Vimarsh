import { FormEvent, useState } from "react";

type ReviewCase = {
  review_id: string;
  case_id: string;
  status: string;
  priority: number;
  prediction: string;
  rationale: string[];
  reviewer_id?: string | null;
  reviewer_note?: string | null;
  final_prediction?: string | null;
  ground_truth?: string | null;
};

type AuditEvent = { event_id: string; event_type: string; actor: string; created_at: string; payload: Record<string, unknown>; event_hash: string };
type EvidenceItem = { evidence_id: string; source_name: string; source_reference: string; evidence_text: string; content_sha256: string; integrity_valid: boolean; created_at: string };

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export default function ReviewQueue() {
  const [token, setToken] = useState("");
  const [evidenceToken, setEvidenceToken] = useState("");
  const [evidenceSource, setEvidenceSource] = useState("");
  const [doi, setDoi] = useState("");
  const [evidenceReference, setEvidenceReference] = useState("");
  const [evidenceText, setEvidenceText] = useState("");
  const [evidenceCaseId, setEvidenceCaseId] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [caseId, setCaseId] = useState("");
  const [prediction, setPrediction] = useState("");
  const [rationale, setRationale] = useState("");
  const [priority, setPriority] = useState("2");
  const [note, setNote] = useState("");
  const [finalPrediction, setFinalPrediction] = useState("");
  const [groundTruth, setGroundTruth] = useState("");
  const [cases, setCases] = useState<ReviewCase[]>([]);
  const [queueStatus, setQueueStatus] = useState("pending");
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [evidenceItems, setEvidenceItems] = useState<EvidenceItem[]>([]);
  const [selected, setSelected] = useState<ReviewCase | null>(null);
  const [status, setStatus] = useState("Enter the reviewer token to connect.");
  const [busy, setBusy] = useState(false);

  async function request(path: string, method = "GET", body?: unknown) {
    const response = await fetch(`${API_BASE}${path}`, {
      method,
      headers: {
        "X-Vimarsh-Reviewer-Token": token,
        ...(body ? { "Content-Type": "application/json" } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    return data;
  }

  async function refresh(statusOverride = queueStatus) {
    setBusy(true);
    try {
      const data = await request(`/v1/reviews?status=${statusOverride}&limit=100`);
      setCases(data.cases);
      setStatus(`Connected · ${data.cases.length} ${statusOverride} case(s)`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to load review queue.");
    } finally {
      setBusy(false);
    }
  }

  async function cancel(item: ReviewCase) {
    const reason = window.prompt("Why is this review being cancelled?");
    if (!reason?.trim()) return;
    setBusy(true);
    try {
      await request(`/v1/reviews/${item.review_id}/cancel`, "POST", {
        actor: reviewer || "operator",
        reason: reason.trim(),
      });
      setStatus(`Cancelled ${item.case_id}`);
      await refresh();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to cancel review.");
    } finally {
      setBusy(false);
    }
  }

  async function inspectHistory(item: ReviewCase) {
    setBusy(true);
    try {
      const data = await request(`/v1/reviews/${item.review_id}/events`);
      setAuditEvents(data.events);
      setStatus(`Audit chain verified · ${data.events.length} event(s) for ${item.case_id}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to load audit history.");
    } finally {
      setBusy(false);
    }
  }

  async function inspectEvidence(item: ReviewCase) {
    if (!evidenceToken) {
      setStatus("Enter the evidence intake token before viewing evidence.");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/v1/evidence/${encodeURIComponent(item.case_id)}`, {
        headers: { "X-Vimarsh-Evidence-Token": evidenceToken },
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      setEvidenceItems(data.evidence);
      setStatus(`${data.evidence.length} evidence item(s) for ${item.case_id}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to load evidence.");
    } finally {
      setBusy(false);
    }
  }

  async function submitCase(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    try {
      await request("/v1/reviews", "POST", {
        case_id: caseId,
        prediction,
        rationale: rationale.split(String.fromCharCode(10)).map((item) => item.trim()).filter(Boolean),
        context: {},
        priority: Number(priority),
      });
      setCaseId("");
      setPrediction("");
      setRationale("");
      await refresh();
      setStatus("Case added to the protected review queue.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to enqueue case.");
    } finally {
      setBusy(false);
    }
  }

  async function claim(item: ReviewCase) {
    setBusy(true);
    try {
      const data = await request(`/v1/reviews/${item.review_id}/claim`, "POST", { reviewer_id: reviewer });
      setSelected(data);
      setFinalPrediction(data.prediction);
      setStatus("Case claimed. Record the adjudication below.");
      await refresh();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to claim case.");
    } finally {
      setBusy(false);
    }
  }

  async function resolveDoi() {
    if (!evidenceToken || !evidenceCaseId || !doi.trim()) {
      setStatus("Enter the evidence token, case ID, and DOI first.");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/v1/evidence/resolve-doi`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Vimarsh-Evidence-Token": evidenceToken,
        },
        body: JSON.stringify({ case_id: evidenceCaseId, doi: doi.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      setStatus(`Crossref metadata recorded · ${data.content_sha256.slice(0, 16)}…`);
      setDoi("");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to resolve DOI metadata.");
    } finally {
      setBusy(false);
    }
  }

  async function submitEvidence(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/v1/evidence`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Vimarsh-Evidence-Token": evidenceToken,
        },
        body: JSON.stringify({
          case_id: evidenceCaseId,
          source_name: evidenceSource,
          source_reference: evidenceReference,
          evidence_text: evidenceText,
          metadata: {},
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      setEvidenceText("");
      setStatus(`Evidence recorded · SHA-256 ${data.content_sha256.slice(0, 16)}…`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to record evidence.");
    } finally {
      setBusy(false);
    }
  }

  async function complete(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selected) return;
    setBusy(true);
    try {
      await request(`/v1/reviews/${selected.review_id}/complete`, "POST", {
        reviewer_id: reviewer,
        reviewer_note: note,
        final_prediction: finalPrediction,
        ground_truth: groundTruth.trim() || null,
      });
      setSelected(null);
      setNote("");
      setGroundTruth("");
      await refresh();
      setStatus("Review recorded. The queue entry is now completed.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to complete review.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="review-workspace">
      <div className="review-heading">
        <div><p className="eyebrow">HUMAN-IN-THE-LOOP WORKFLOW</p><h2>Operator review queue</h2></div>
        <span className="review-status">{status}</span>
      </div>
      <div className="review-security-note">
        Protected local queue. The token stays in this page's memory only and is not saved in browser storage.
        Reviewer IDs are labels, not authenticated individual accounts. Do not expose this demo queue publicly.
      </div>
      <div className="review-connect">
        <label><span>Reviewer API token</span><input type="password" value={token} onChange={(e) => setToken(e.target.value)} autoComplete="off" placeholder="Enter server-side reviewer token" /></label>
        <label><span>Reviewer ID</span><input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="e.g. reviewer-01" /></label>
        <button type="button" disabled={!token || busy} onClick={refresh}>Connect / refresh queue</button>
      </div>
      <div className="review-columns">
        <form className="review-card" onSubmit={submitCase}>
          <h3>Send a case for review</h3>
          <label><span>Case ID</span><input required maxLength={128} value={caseId} onChange={(e) => setCaseId(e.target.value)} /></label>
          <label><span>Current prediction</span><input required maxLength={256} value={prediction} onChange={(e) => setPrediction(e.target.value)} /></label>
          <label><span>Rationale, one item per line</span><textarea rows={3} value={rationale} onChange={(e) => setRationale(e.target.value)} /></label>
          <label><span>Priority (0 to 5)</span><input type="number" min="0" max="5" value={priority} onChange={(e) => setPriority(e.target.value)} /></label>
          <button type="submit" disabled={!token || busy}>Queue for human review</button>
        </form>
        <div className="review-card">
          <div className="review-card-title"><h3>Review cases</h3><button type="button" disabled={!token || busy} onClick={refresh}>Refresh</button></div>
          <label><span>Queue status</span><select value={queueStatus} onChange={(e) => { setQueueStatus(e.target.value); void refresh(e.target.value); }}><option value="pending">Pending</option><option value="claimed">Claimed</option><option value="completed">Completed</option><option value="cancelled">Cancelled</option></select></label>
          {cases.length === 0 ? <p className="review-empty">No {queueStatus} cases loaded.</p> : cases.map((item) => (
            <div className="review-case" key={item.review_id}>
              <div><strong>{item.case_id}</strong><span>Priority {item.priority} · {item.prediction}</span></div>
              <div className="review-case-actions">{item.status === "pending" && <button type="button" disabled={!reviewer || busy} onClick={() => claim(item)}>Claim</button>}{["pending", "claimed"].includes(item.status) && <button type="button" disabled={busy} onClick={() => cancel(item)}>Cancel</button>}<button type="button" disabled={busy} onClick={() => inspectHistory(item)}>History</button><button type="button" disabled={busy} onClick={() => inspectEvidence(item)}>Evidence</button></div>
            </div>
          ))}
        </div>
      </div>
      <form className="review-card review-evidence" onSubmit={submitEvidence}>
        <h3>Attach operator-submitted evidence</h3>
        <p className="review-empty">Manual entries are operator-submitted. DOI lookup retrieves Crossref bibliographic metadata only, not full text or scientific-claim verification.</p>
        <div className="review-connect">
          <label><span>Evidence intake token</span><input type="password" autoComplete="off" value={evidenceToken} onChange={(e) => setEvidenceToken(e.target.value)} /></label>
          <label><span>Case ID</span><input required value={evidenceCaseId} onChange={(e) => setEvidenceCaseId(e.target.value)} /></label>
        </div>
        <div className="review-connect">
          <label><span>Source name</span><input required value={evidenceSource} onChange={(e) => setEvidenceSource(e.target.value)} placeholder="Source / provider" /></label>
          <label><span>Source reference</span><input required value={evidenceReference} onChange={(e) => setEvidenceReference(e.target.value)} placeholder="URL or reference identifier" /></label>
        </div>
        <label><span>Evidence content</span><textarea required rows={4} maxLength={20000} value={evidenceText} onChange={(e) => setEvidenceText(e.target.value)} /></label>
        <div className="review-connect">
          <label><span>Optional DOI lookup (Crossref metadata)</span><input value={doi} onChange={(e) => setDoi(e.target.value)} placeholder="10.xxxx/example" /></label>
          <button type="button" disabled={!evidenceToken || busy} onClick={resolveDoi}>Retrieve DOI metadata</button>
        </div>
        <button type="submit" disabled={!evidenceToken || busy}>Record evidence</button>
      </form>

      {auditEvents.length > 0 && (
        <div className="review-card">
          <h3>Verified review audit history</h3>
          {auditEvents.map((event) => <div className="review-case" key={event.event_id}><div><strong>{event.event_type}</strong><span>{event.actor} · {event.created_at}</span></div><code>{event.event_hash.slice(0, 18)}…</code></div>)}
        </div>
      )}
      {evidenceItems.length > 0 && (
        <div className="review-card">
          <h3>Evidence records</h3>
          {evidenceItems.map((item) => <div className="review-case" key={item.evidence_id}><div><strong>{item.source_name}</strong><span>{item.source_reference} · {item.integrity_valid ? "Digest verified" : "Digest mismatch"}</span><p>{item.evidence_text}</p></div><code>{item.content_sha256.slice(0, 18)}…</code></div>)}
        </div>
      )}
      {selected && (
        <form className="review-card review-completion" onSubmit={complete}>
          <h3>Complete review · {selected.case_id}</h3>
          <label><span>Final prediction</span><input required value={finalPrediction} onChange={(e) => setFinalPrediction(e.target.value)} /></label>
          <label><span>Reviewer note</span><textarea required rows={3} value={note} onChange={(e) => setNote(e.target.value)} /></label>
          <label><span>Ground truth (optional, retrospective only)</span><input value={groundTruth} onChange={(e) => setGroundTruth(e.target.value)} /></label>
          <button type="submit" disabled={busy}>Record completed review</button>
        </form>
      )}
    </section>
  );
}
