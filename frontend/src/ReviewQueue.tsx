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

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export default function ReviewQueue() {
  const [token, setToken] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [caseId, setCaseId] = useState("");
  const [prediction, setPrediction] = useState("");
  const [rationale, setRationale] = useState("");
  const [priority, setPriority] = useState("2");
  const [note, setNote] = useState("");
  const [finalPrediction, setFinalPrediction] = useState("");
  const [groundTruth, setGroundTruth] = useState("");
  const [cases, setCases] = useState<ReviewCase[]>([]);
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

  async function refresh() {
    setBusy(true);
    try {
      const data = await request("/v1/reviews?status=pending&limit=100");
      setCases(data.cases);
      setStatus(`Connected · ${data.cases.length} pending case(s)`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to load review queue.");
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
        rationale: rationale.split("\n").map((item) => item.trim()).filter(Boolean),
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
          <div className="review-card-title"><h3>Pending cases</h3><button type="button" disabled={!token || busy} onClick={refresh}>Refresh</button></div>
          {cases.length === 0 ? <p className="review-empty">No pending cases loaded.</p> : cases.map((item) => (
            <div className="review-case" key={item.review_id}>
              <div><strong>{item.case_id}</strong><span>Priority {item.priority} · {item.prediction}</span></div>
              <button type="button" disabled={!reviewer || busy} onClick={() => claim(item)}>Claim</button>
            </div>
          ))}
        </div>
      </div>
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
