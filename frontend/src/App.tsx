import { FormEvent, useEffect, useMemo, useState } from "react";

type Schema = {
  feature_names: string[];
  primary_model_index: number;
  base_model_count: number;
  artifact_version: number;
};

type AlternateRun = {
  alternate_models: Array<{
    model_index: number;
    model_type: string;
    prediction: string;
    confidence: number;
    latency_ms: number;
  }>;
  total_latency_ms: number;
  execution_status: string;
};

type Decision = {
  prediction: string;
  confidence: number;
  estimated_error_probability: number;
  baseline_risk: number;
  action: string;
  recommendation_only: boolean;
  execution_status: string;
  rationale: string[];
  eligible_actions: string[];
  base_model_count: number;
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

const WINE_SAMPLE: Record<string, string> = {
  "fixed acidity": "7.4",
  "volatile acidity": "0.70",
  "citric acid": "0.00",
  "residual sugar": "1.9",
  chlorides: "0.076",
  "free sulfur dioxide": "11",
  "total sulfur dioxide": "34",
  density: "0.9978",
  pH: "3.51",
  sulphates: "0.56",
  alcohol: "9.4",
  wine_type: "0",
};

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function App() {
  const [schema, setSchema] = useState<Schema | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [harm, setHarm] = useState("2");
  const [budget, setBudget] = useState("2");
  const [alternate, setAlternate] = useState(true);
  const [alternateCost, setAlternateCost] = useState("1");
  const [decision, setDecision] = useState<Decision | null>(null);
  const [lastSample, setLastSample] = useState<Record<string, number> | null>(null);
  const [alternateRun, setAlternateRun] = useState<AlternateRun | null>(null);
  const [alternateLoading, setAlternateLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Connecting to the decision API…");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/v1/reliability/schema`)
      .then(async (response) => {
        if (!response.ok) throw new Error(response.status === 503
          ? "Reliability bundle not configured. Set VIMARSH_RELIABILITY_BUNDLE on the API."
          : `API returned HTTP ${response.status}`);
        return response.json() as Promise<Schema>;
      })
      .then((data) => {
        if (!active) return;
        setSchema(data);
        setValues(Object.fromEntries(data.feature_names.map((name) => [name, ""])));
        setStatus("Reliability bundle connected");
      })
      .catch((reason: Error) => {
        if (!active) return;
        setError(reason.message);
        setStatus("Waiting for a configured model bundle");
      });
    return () => { active = false; };
  }, []);

  const filledCount = useMemo(
    () => Object.values(values).filter((value) => value.trim() !== "").length,
    [values],
  );

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!schema) return;
    setLoading(true);
    setError("");
    setDecision(null);
    try {
      const sample = Object.fromEntries(
        schema.feature_names.map((name) => [name, Number(values[name])]),
      );
      if (Object.values(sample).some((value) => !Number.isFinite(value))) {
        throw new Error("Enter a valid number for every model feature.");
      }
      setLastSample(sample);
      setAlternateRun(null);
      const response = await fetch(`${API_BASE}/v1/reliability/decisions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample,
          harm: Number(harm),
          available_actions: alternate ? ["alternate_model"] : [],
          estimated_action_costs: alternate ? { alternate_model: Number(alternateCost) } : {},
          remaining_budget: Number(budget),
        }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `API returned HTTP ${response.status}`);
      setDecision(body as Decision);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to obtain a recommendation.");
    } finally {
      setLoading(false);
    }
  }


  async function runAlternateModels() {
    if (!lastSample) return;
    setAlternateLoading(true);
    setError("");
    setAlternateRun(null);
    try {
      const response = await fetch(`${API_BASE}/v1/actions/alternate-model/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample: lastSample }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `API returned HTTP ${response.status}`);
      setAlternateRun(body as AlternateRun);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to run alternate models.");
    } finally {
      setAlternateLoading(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="VIMARSH home">
          <span className="brand-mark">V</span>
          <span>VIMARSH</span>
        </a>
        <div className="top-meta">
          <span className="status-dot" />
          <span>{status}</span>
          <span className="research-tag">RESEARCH PREVIEW</span>
        </div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">RELIABILITY-DRIVEN DECISION INTELLIGENCE</p>
          <h1>Know when a prediction<br /><em>needs a second look.</em></h1>
          <p className="hero-text">
            VIMARSH combines model uncertainty, disagreement and input-quality signals
            to estimate prediction reliability, then applies transparent decision constraints.
          </p>
        </div>
        <div className="hero-note">
          <span className="note-kicker">THE PRINCIPLE</span>
          <p>Confidence is a signal.<br />Reliability is the question.</p>
          <span className="note-line" />
        </div>
      </section>

      <section className="workspace">
        <div className="section-heading">
          <div>
            <p className="eyebrow">LIVE EVALUATION</p>
            <h2>Decision workspace</h2>
          </div>
          <span className={schema ? "connection connected" : "connection"}>{schema ? "MODEL CONNECTED" : "MODEL OFFLINE"}</span>
        </div>

        {!schema && (
          <div className="notice">
            <strong>Model bundle unavailable</strong>
            <p>{error || "Start the API and configure a trusted VIMARSH reliability bundle to load its feature schema."}</p>
            <code>VIMARSH_RELIABILITY_BUNDLE=research/models/reliability/wine_quality_reliability_bundle.joblib</code>
          </div>
        )}

        <div className="workspace-grid">
          <form className="input-panel" onSubmit={submit}>
            <div className="panel-title">
              <div><span className="step">01</span><h3>Input features</h3></div>
              <span className="muted">{schema ? `${filledCount}/${schema.feature_names.length} entered` : "Awaiting schema"}</span>
            </div>
            {schema && (
              <>
                <div className="sample-row">
                  <span>Use a known benchmark example to populate the feature fields.</span>
                  <button
                    className="sample-button"
                    type="button"
                    onClick={() => setValues((current) => ({
                      ...current,
                      ...Object.fromEntries(
                        schema.feature_names
                          .filter((name) => WINE_SAMPLE[name] !== undefined)
                          .map((name) => [name, WINE_SAMPLE[name]]),
                      ),
                    }))}
                  >
                    Load example wine sample
                  </button>
                </div>
                <div className="feature-grid">
                  {schema.feature_names.map((name) => (
                    <label className="field" key={name}>
                      <span>{name.replace(/_/g, " ")}</span>
                      <input
                        type="number"
                        step="any"
                        required
                        value={values[name] ?? ""}
                        onChange={(event) => setValues((current) => ({ ...current, [name]: event.target.value }))}
                        placeholder="Enter value"
                      />
                    </label>
                  ))}
                </div>
                <div className="panel-divider" />
                <div className="panel-title compact">
                  <div><span className="step">02</span><h3>Decision context</h3></div>
                </div>
                <div className="context-grid">
                  <label className="field">
                    <span>Impact / harm score <small>0 to 10</small></span>
                    <input type="number" min="0" max="10" step="any" required value={harm} onChange={(event) => setHarm(event.target.value)} />
                  </label>
                  <label className="field">
                    <span>Remaining action budget</span>
                    <input type="number" min="0" step="any" required value={budget} onChange={(event) => setBudget(event.target.value)} />
                  </label>
                </div>
                <div className="action-option">
                  <label className="toggle-label">
                    <input type="checkbox" checked={alternate} onChange={(event) => setAlternate(event.target.checked)} />
                    <span className="toggle-ui" />
                    <span><strong>Alternate model available</strong><small>Recommendation only. No model action is executed.</small></span>
                  </label>
                  {alternate && (
                    <label className="cost-field">
                      <span>Estimated cost</span>
                      <input type="number" min="0" step="any" required value={alternateCost} onChange={(event) => setAlternateCost(event.target.value)} />
                    </label>
                  )}
                </div>
                <button className="submit-button" type="submit" disabled={loading}>
                  {loading ? "Evaluating…" : "Evaluate reliability"} <span>↗</span>
                </button>
              </>
            )}
          </form>

          <aside className="result-panel">
            <div className="panel-title">
              <div><span className="step">03</span><h3>Decision output</h3></div>
              <span className="muted">INFERENCE RESULT</span>
            </div>
            {!decision ? (
              <div className="empty-result">
                <div className="orbit"><span /><span /><span /></div>
                <h3>Waiting for an evaluation</h3>
                <p>Complete the feature inputs to see the model prediction, estimated error probability and constrained recommendation.</p>
              </div>
            ) : (
              <div className="result-content">
                <div className="prediction-block">
                  <span className="result-label">PRIMARY MODEL PREDICTION</span>
                  <strong>{decision.prediction}</strong>
                  <span className="confidence">{percent(decision.confidence)} confidence</span>
                </div>
                <div className="metric-row">
                  <div><span>Estimated error</span><strong>{percent(decision.estimated_error_probability)}</strong></div>
                  <div><span>Baseline risk</span><strong>{decision.baseline_risk.toFixed(3)}</strong></div>
                </div>
                <div className="recommendation">
                  <span className="result-label">RECOMMENDED ACTION</span>
                  <strong>{decision.action.replace(/_/g, " ")}</strong>
                  <p>Recommendation only · not executed</p>
                </div>
                <div className="rationale">
                  <span className="result-label">WHY THIS DECISION</span>
                  <ul>{decision.rationale.map((item, index) => <li key={index}>{item}</li>)}</ul>
                </div>
                <div className="result-foot">Scored using {decision.base_model_count} base models and the fitted reliability estimator.</div>
                <div className="alternate-action">
                  <button type="button" onClick={runAlternateModels} disabled={alternateLoading || !lastSample}>
                    {alternateLoading ? "Running alternate models…" : "Run alternate models"}
                  </button>
                  <span>Explicit local inference · measured latency · no automatic model switch</span>
                </div>
                {alternateRun && (
                  <div className="alternate-results">
                    <span className="result-label">ALTERNATE MODEL OUTPUTS</span>
                    {alternateRun.alternate_models.map((item) => (
                      <div className="alternate-model" key={item.model_index}>
                        <div><strong>{item.model_type}</strong><span>Model {item.model_index}</span></div>
                        <strong>{item.prediction}</strong>
                        <span>{percent(item.confidence)} confidence · {item.latency_ms.toFixed(2)} ms</span>
                      </div>
                    ))}
                    <p>Total alternate inference: {alternateRun.total_latency_ms.toFixed(2)} ms. Outputs are evidence for review, not an automatic replacement of the primary prediction.</p>
                  </div>
                )}
              </div>
            )}
            {error && schema && <p className="error-message">{error}</p>}
          </aside>
        </div>
      </section>

      <footer>
        <span>VIMARSH · RESEARCH BUILD 0.1</span>
        <span>Experimental decision support. Not validated for safety-critical use.</span>
      </footer>
    </main>
  );
}

export default App;
