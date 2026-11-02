import { useMemo, useState } from "react";

type Cell = string;
type Row = Record<string, Cell>;
type NumericSummary = { name: string; count: number; missing: number; mean: number; median: number; min: number; max: number; std: number };

function parseCSV(source: string): { headers: string[]; rows: Row[] } {
  const records: string[][] = [];
  let row: string[] = [], cell = "", quoted = false;
  for (let i = 0; i < source.length; i += 1) {
    const ch = source[i];
    if (ch === '"') {
      if (quoted && source[i + 1] === '"') { cell += '"'; i += 1; }
      else quoted = !quoted;
    } else if (ch === "," && !quoted) { row.push(cell.trim()); cell = ""; }
    else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && source[i + 1] === "\n") i += 1;
      row.push(cell.trim()); cell = "";
      if (row.some((v) => v !== "")) records.push(row);
      row = [];
    } else cell += ch;
  }
  row.push(cell.trim());
  if (row.some((v) => v !== "")) records.push(row);
  if (records.length < 2) throw new Error("The CSV needs a header row and at least one data row.");
  const headers = records[0].map((h, i) => h || `column_${i + 1}`);
  if (new Set(headers).size !== headers.length) throw new Error("Column names must be unique.");
  const rows = records.slice(1).filter((r) => r.some((v) => v !== "")).map((r) =>
    Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])) as Row
  );
  return { headers, rows };
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
function correlation(a: number[], b: number[]) {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 0;
  const ma = a.slice(0, n).reduce((s, v) => s + v, 0) / n;
  const mb = b.slice(0, n).reduce((s, v) => s + v, 0) / n;
  const numerator = a.slice(0, n).reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0);
  const da = Math.sqrt(a.slice(0, n).reduce((s, v) => s + (v - ma) ** 2, 0));
  const db = Math.sqrt(b.slice(0, n).reduce((s, v) => s + (v - mb) ** 2, 0));
  return da && db ? numerator / (da * db) : 0;
}
function fmt(value: number) { return Number.isFinite(value) ? value.toFixed(2) : "N/A"; }

export default function DataAnalysis() {
  const [dataset, setDataset] = useState<{ name: string; headers: string[]; rows: Row[] } | null>(null);
  const [error, setError] = useState("");
  const [target, setTarget] = useState("");
  const [focus, setFocus] = useState("");
  const [busy, setBusy] = useState(false);

  async function loadFile(file?: File) {
    if (!file) return;
    setBusy(true); setError("");
    try {
      if (!file.name.toLowerCase().endsWith(".csv")) throw new Error("Please upload a CSV file.");
      if (file.size > 15 * 1024 * 1024) throw new Error("CSV must be 15 MB or smaller.");
      const parsed = parseCSV(await file.text());
      setDataset({ name: file.name, ...parsed });
      setTarget(parsed.headers[parsed.headers.length - 1]);
      setFocus("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to read this CSV.");
    } finally { setBusy(false); }
  }

  const analysis = useMemo(() => {
    if (!dataset) return null;
    const { headers, rows } = dataset;
    const numeric = headers.filter((h) => {
      const values = rows.map((r) => r[h]).filter((v) => v !== "");
      return values.length > 0 && values.filter((v) => Number.isFinite(Number(v))).length / values.length >= 0.85;
    });
    const summaries: NumericSummary[] = numeric.map((name) => {
      const values = rows.map((r) => r[name]).filter((v) => v !== "" && Number.isFinite(Number(v))).map(Number);
      const mean = values.reduce((s, v) => s + v, 0) / Math.max(values.length, 1);
      return { name, count: values.length, missing: rows.length - values.length, mean, median: median(values), min: Math.min(...values), max: Math.max(...values), std: Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(values.length, 1)) };
    });
    const missing = headers.map((name) => ({ name, count: rows.filter((r) => r[name] === "").length })).sort((a, b) => b.count - a.count);
    const selected = numeric.find((n) => n === focus) || numeric[0] || "";
    const values = selected ? rows.map((r) => Number(r[selected])).filter(Number.isFinite) : [];
    const min = values.reduce((current, value) => Math.min(current, value), Number.POSITIVE_INFINITY);
    const max = values.reduce((current, value) => Math.max(current, value), Number.NEGATIVE_INFINITY);
    const bins = Array.from({ length: 10 }, (_, i) => ({ label: i + 1, count: 0 }));
    values.forEach((v) => { const index = max === min ? 0 : Math.min(9, Math.floor(((v - min) / (max - min)) * 10)); bins[index].count += 1; });
    const categories = target ? rows.reduce<Record<string, number>>((acc, r) => { const value = r[target] || "(missing)"; acc[value] = (acc[value] || 0) + 1; return acc; }, {}) : {};
    const categoryRows = Object.entries(categories).sort((a, b) => b[1] - a[1]).slice(0, 12);
    const corrNames = numeric.slice(0, 6);
    const corr = corrNames.map((a) => corrNames.map((b) => {
      const pairs = rows.filter((r) => r[a] !== "" && r[b] !== "" && Number.isFinite(Number(r[a])) && Number.isFinite(Number(r[b])));
      return correlation(pairs.map((r) => Number(r[a])), pairs.map((r) => Number(r[b])));
    }));
    return { numeric, summaries, missing, selected, bins, categoryRows, corrNames, corr };
  }, [dataset, target, focus]);

  return <section className="dav-section" id="data-analysis">
    <div className="dav-header">
      <div><p className="eyebrow">MILESTONE 74 · DATA ANALYSIS & VISUALISATION</p><h2>Explore the evidence behind the model.</h2><p className="dav-subtitle">Upload a benchmark CSV to inspect data quality, distributions, class balance and feature relationships. Analysis runs locally in this browser.</p></div>
      <label className="dav-upload"><input type="file" accept=".csv,text/csv" onChange={(e) => void loadFile(e.target.files?.[0])} />{busy ? "Reading CSV…" : "＋ Upload CSV"}</label>
    </div>
    {error && <p className="dav-error" role="alert">{error}</p>}
    {!dataset ? <div className="dav-empty"><strong>Start with a dataset</strong><span>CSV format · maximum 15 MB · header row required</span><p>Try a benchmark export such as Wine Quality, MAGIC Gamma Telescope or HTRU2. No sample metrics are fabricated or prefilled.</p></div> : analysis && <>
      <div className="dav-fileline"><span><strong>{dataset.name}</strong> · {dataset.rows.length.toLocaleString()} records · {dataset.headers.length} columns</span><button type="button" onClick={() => { setDataset(null); setTarget(""); setFocus(""); }}>Clear dataset</button></div>
      <div className="dav-kpis">
        <article><span>ROWS</span><strong>{dataset.rows.length.toLocaleString()}</strong><small>observations loaded</small></article>
        <article><span>COLUMNS</span><strong>{dataset.headers.length}</strong><small>{analysis.numeric.length} numeric features detected</small></article>
        <article><span>MISSING CELLS</span><strong>{analysis.missing.reduce((s, v) => s + v.count, 0).toLocaleString()}</strong><small>across all columns</small></article>
        <article><span>COMPLETE ROWS</span><strong>{dataset.rows.filter((r) => dataset.headers.every((h) => r[h] !== "")).length.toLocaleString()}</strong><small>no blank cells</small></article>
      </div>
      <div className="dav-grid">
        <article className="dav-card dav-wide"><div className="dav-card-head"><div><span className="eyebrow">01 · DISTRIBUTION</span><h3>Feature histogram</h3></div><select aria-label="Feature for histogram" value={analysis.selected} onChange={(e) => setFocus(e.target.value)}>{analysis.numeric.map((n) => <option key={n}>{n}</option>)}</select></div>
          {analysis.numeric.length ? <div className="dav-bars">{analysis.bins.map((b) => <div className="dav-bin" key={b.label}><span style={{ height: `${Math.max(3, b.count / Math.max(...analysis.bins.map((x) => x.count), 1) * 150)}px` }} /><small>{b.label}</small><em>{b.count}</em></div>)}</div> : <p className="dav-muted">No numeric columns detected.</p>}
          <p className="dav-caption">Ten equal-width bins for {analysis.selected || "the selected feature"}. Counts are computed from non-missing numeric values.</p>
        </article>
        <article className="dav-card"><div className="dav-card-head"><div><span className="eyebrow">02 · CLASS BALANCE</span><h3>Category distribution</h3></div><select aria-label="Target category column" value={target} onChange={(e) => setTarget(e.target.value)}>{dataset.headers.map((h) => <option key={h} value={h}>{h}</option>)}</select></div>
          <div className="dav-category-list">{analysis.categoryRows.map(([label, count]) => <div key={label}><span title={label}>{label}</span><div><i style={{ width: `${count / dataset.rows.length * 100}%` }} /></div><strong>{count}</strong></div>)}</div>
          <p className="dav-caption">Top 12 values in the selected column. Useful for spotting class imbalance.</p>
        </article>
        <article className="dav-card"><div className="dav-card-head"><div><span className="eyebrow">03 · DATA QUALITY</span><h3>Missing values</h3></div></div>
          <div className="dav-missing-list">{analysis.missing.slice(0, 9).map((item) => <div key={item.name}><span>{item.name}</span><strong>{item.count}</strong><small>{(item.count / dataset.rows.length * 100).toFixed(1)}%</small></div>)}</div>
          <p className="dav-caption">Blank-cell count by column, sorted highest first.</p>
        </article>
        <article className="dav-card dav-wide"><div className="dav-card-head"><div><span className="eyebrow">04 · FEATURE RELATIONSHIPS</span><h3>Correlation matrix</h3></div></div>
          {analysis.corrNames.length ? <div className="dav-correlation"><div className="dav-corr-row dav-corr-label" style={{ gridTemplateColumns: `95px repeat(${analysis.corrNames.length}, minmax(48px, 1fr))` }}><span />{analysis.corrNames.map((n) => <small key={n} title={n}>{n.slice(0, 9)}</small>)}</div>{analysis.corr.map((line, i) => <div className="dav-corr-row" key={analysis.corrNames[i]} style={{ gridTemplateColumns: `95px repeat(${analysis.corrNames.length}, minmax(48px, 1fr))` }}><small title={analysis.corrNames[i]}>{analysis.corrNames[i].slice(0, 12)}</small>{line.map((v, j) => <span key={j} title={`${analysis.corrNames[i]} × ${analysis.corrNames[j]}: ${v.toFixed(3)}`} className="dav-corr-cell" style={{ opacity: 0.18 + Math.abs(v) * 0.82, background: v >= 0 ? "#7769b4" : "#bd8c72" }}>{v.toFixed(2)}</span>)}</div>)}</div> : <p className="dav-muted">Correlation requires at least one numeric column.</p>}
          <p className="dav-caption">Pearson correlation on pairwise complete rows. Purple indicates positive, clay indicates negative correlation. Showing up to six numeric columns.</p>
        </article>
        <article className="dav-card dav-wide"><div className="dav-card-head"><div><span className="eyebrow">05 · DESCRIPTIVE STATISTICS</span><h3>Numerical summary</h3></div></div>
          <div className="dav-table-wrap"><table className="dav-table"><thead><tr><th>Feature</th><th>Valid</th><th>Missing</th><th>Mean</th><th>Median</th><th>Std. dev.</th><th>Min</th><th>Max</th></tr></thead><tbody>{analysis.summaries.map((s) => <tr key={s.name}><td>{s.name}</td><td>{s.count}</td><td>{s.missing}</td><td>{fmt(s.mean)}</td><td>{fmt(s.median)}</td><td>{fmt(s.std)}</td><td>{fmt(s.min)}</td><td>{fmt(s.max)}</td></tr>)}</tbody></table></div>
        </article>
      </div>
      <p className="dav-disclaimer">Exploratory analysis only. This view does not retrain models, infer causality, or replace the benchmark evaluation pipeline. Exported data remains in the browser session and is not uploaded to the VIMARSH API.</p>
    </>}
  </section>;
}
