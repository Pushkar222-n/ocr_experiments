import { useEffect, useMemo, useState } from "react";
import { MODELS, MODEL_BY_ID, CATEGORIES } from "./models";
import PdfViewer from "./PdfViewer";
import Markdown from "./Markdown";

/* ---------- data hooks ---------- */

function usePersistedState(key, initial) {
  const [v, setV] = useState(() => {
    try {
      const s = localStorage.getItem(key);
      return s ? JSON.parse(s) : initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(v));
  }, [key, v]);
  return [v, setV];
}

function useComparison() {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    fetch("/outputs/comparison.json")
      .then((r) => r.json())
      .then(setRows)
      .catch(() => setRows([]));
  }, []);
  return rows;
}

const summaryCache = {};
function useSummaries(dirs) {
  const key = dirs.join(",");
  const [data, setData] = useState({});
  useEffect(() => {
    let dead = false;
    Promise.all(
      dirs.map(async (d) => {
        if (!summaryCache[d]) {
          summaryCache[d] = await fetch(`/outputs/${d}/summary.json`)
            .then((r) => (r.ok ? r.json() : []))
            .catch(() => []);
        }
        return [d, summaryCache[d]];
      })
    ).then((entries) => !dead && setData(Object.fromEntries(entries)));
    return () => {
      dead = true;
    };
  }, [key]);
  return data;
}

function useMarkdown(url) {
  const [state, setState] = useState({ loading: true, text: "", error: null });
  useEffect(() => {
    let cancelled = false;
    setState({ loading: true, text: "", error: null });
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.text();
      })
      .then((text) => !cancelled && setState({ loading: false, text, error: null }))
      .catch((e) => !cancelled && setState({ loading: false, text: "", error: e.message }));
    return () => {
      cancelled = true;
    };
  }, [url]);
  return state;
}

// same markup-stripping rule as scripts/compare.py — a markup-inflation detector,
// not a scoring function (doesn't strip LaTeX or grounding tags)
const visibleChars = (raw) =>
  raw ? raw.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim().length : 0;

const benchmarkVariant = (m) => m.variants.find((v) => v.benchmark) ?? m.variants[0];

/* ---------- compare view ---------- */

function StatBadges({ row, visible }) {
  if (!row) return null;
  const mem = row.max_gpu_mem_mb ?? row.gpu_mem_mb;
  const memTitle = row.max_gpu_mem_mb
    ? "peak VRAM (max over per-page samples)"
    : "VRAM from a single post-run sample — may understate the peak";
  return (
    <div className="badges">
      <span title="wall time per page">{row.seconds_per_page}s/pg</span>
      {visible != null && (
        <span title="text after stripping markup tags — same rule as compare.py">
          {visible.toLocaleString()} visible
        </span>
      )}
      {mem != null && <span title={memTitle}>{(mem / 1024).toFixed(1)}GB</span>}
      {row.cost_usd != null && (
        <span className="cost-badge" title="estimated cost for this document — build-time rate, verify">
          ${row.cost_usd}
        </span>
      )}
      {row.credits != null && row.credits > 0 && (
        <span title="credits the API reported for this document">{row.credits} cr</span>
      )}
      {row.mean_mean_confidence != null && (
        <span title="surya's self-reported confidence — page-level, scores only what it chose to read; NOT a coverage metric">
          conf {row.mean_mean_confidence}
        </span>
      )}
    </div>
  );
}

function ModelPane({ model, category, onClose }) {
  const [variantId, setVariantId] = useState(benchmarkVariant(model).id);
  const variant = model.variants.find((v) => v.id === variantId) ?? model.variants[0];
  const url = `/outputs/${variant.dir}/${category}.md`;
  const { loading, text: fetched, error } = useMarkdown(url);
  const [raw, setRaw] = useState(false);
  const [showNote, setShowNote] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [saveErr, setSaveErr] = useState(null);
  const [override, setOverride] = useState(null); // saved edits, shown until next fetch
  useEffect(() => {
    setOverride(null);
    setEditing(false);
    setSaveErr(null);
  }, [url]);
  const text = override ?? fetched;
  const visible = useMemo(() => (text ? visibleChars(text) : null), [text]);
  const summaries = useSummaries([variant.dir]);
  const statRow = (summaries[variant.dir] ?? []).find((r) => r.pdf === `${category}.pdf`);

  const startEdit = () => {
    setDraft(text);
    setSaveErr(null);
    setEditing(true);
  };
  const save = async () => {
    try {
      const r = await fetch(url, { method: "PUT", body: draft });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      setOverride(draft);
      setEditing(false);
      setSaveErr(null);
    } catch (e) {
      setSaveErr(e.message);
    }
  };
  const onEditKey = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      save();
    }
    if (e.key === "Escape") setEditing(false);
    e.stopPropagation();
  };

  return (
    <div className="pane">
      <div className="pane-header">
        <span className="dot" style={{ background: model.color }} />
        <strong>{model.label}</strong>
        {model.variants.length > 1 && (
          <div className="variant-switch">
            {model.variants.map((v) => (
              <button
                key={v.id}
                className={v.id === variantId ? "active" : ""}
                onClick={() => setVariantId(v.id)}
                title={v.benchmark ? "the benchmark run" : "kept engine artifact — not the benchmark row"}
              >
                {v.label}
              </button>
            ))}
          </div>
        )}
        <StatBadges row={statRow} visible={visible} />
        <div className="pane-actions">
          <button
            className={"mini" + (showNote ? " on" : "")}
            onClick={() => setShowNote((v) => !v)}
            title="field notes for this model"
          >
            ⓘ
          </button>
          {editing ? (
            <>
              <button className="mini save" onClick={save} title="save to disk (Ctrl+S) — first save keeps a .md.orig backup">
                save
              </button>
              <button className="mini" onClick={() => setEditing(false)} title="discard changes (Esc)">
                cancel
              </button>
            </>
          ) : (
            <>
              <button className="mini" onClick={startEdit} disabled={loading || !!error} title="edit the .md on disk">
                edit
              </button>
              <button className="mini" onClick={() => setRaw((v) => !v)} title="toggle raw markdown source">
                {raw ? "rendered" : "raw"}
              </button>
            </>
          )}
          <button className="mini" onClick={onClose} title="remove pane">
            ×
          </button>
        </div>
      </div>
      {showNote && <div className="note-strip">{model.note}</div>}
      {saveErr && <div className="note-strip error-strip">save failed: {saveErr}</div>}
      <div className="pane-body">
        {loading && <p className="muted">loading…</p>}
        {error && (
          <p className="error">
            no output at {url} ({error})
          </p>
        )}
        {!loading && !error && editing && (
          <textarea
            className="md-editor"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onEditKey}
            spellCheck={false}
          />
        )}
        {!loading && !error && !editing && raw && <pre className="raw">{text}</pre>}
        {!loading && !error && !editing && !raw && <Markdown text={text} />}
      </div>
    </div>
  );
}

function CompareView({ category, active, toggle }) {
  return (
    <div className="workspace">
      <div className="pane pdf-pane">
        <div className="pane-header">
          <strong>{category.replaceAll("_", " ")}.pdf</strong>
          <span className="muted-inline">source document</span>
        </div>
        <PdfViewer stem={category} />
      </div>

      {active.length === 0 && (
        <div className="empty-hint">
          <p>Pick one or more models above to compare against the PDF.</p>
        </div>
      )}

      {active.map((id) => (
        <ModelPane key={id} model={MODEL_BY_ID[id]} category={category} onClose={() => toggle(id)} />
      ))}
    </div>
  );
}

/* ---------- metrics view ---------- */

const ALL = "__all__";

function aggregate(rows) {
  const by = {};
  for (const r of rows) {
    const key = `${r.model}|${r.engine ?? ""}`;
    const a = (by[key] ??= {
      model: r.model,
      engine: r.engine,
      artifact: r.artifact,
      closed: r.closed,
      pages: 0,
      total_seconds: 0,
      total_chars: 0,
      visible_chars: 0,
      cost_usd: 0,
      credits: 0,
      _hasCost: false,
    });
    a.pages += r.pages;
    a.total_seconds += r.total_seconds;
    a.total_chars += r.total_chars;
    a.visible_chars += r.visible_chars ?? 0;
    if (r.cost_usd != null) { a.cost_usd += r.cost_usd; a._hasCost = true; }
    if (r.credits != null) a.credits += r.credits;
  }
  return Object.values(by).map((a) => ({
    ...a,
    total_seconds: +a.total_seconds.toFixed(1),
    seconds_per_page: +(a.total_seconds / a.pages).toFixed(2),
    visible_chars: a.visible_chars || undefined,
    pct_text: a.visible_chars
      ? +((100 * a.visible_chars) / a.total_chars).toFixed(1)
      : undefined,
    cost_usd: a._hasCost ? +a.cost_usd.toFixed(4) : undefined,
    credits: a.credits || undefined,
  }));
}

function BarCard({ title, note, rows, field, better, fmt }) {
  const withVal = rows.filter((r) => r[field] != null);
  if (!withVal.length) return null;
  const max = Math.max(...withVal.map((r) => r[field]));
  const sorted = [...withVal].sort((a, b) =>
    better === "low" ? a[field] - b[field] : b[field] - a[field]
  );
  return (
    <div className="card">
      <h3>{title}</h3>
      <p className="note">{note}</p>
      {sorted.map((r, i) => {
        const m = MODEL_BY_ID[r.model];
        return (
          <div className={"bar-row" + (i === 0 ? " best" : "")} key={r.model}>
            <span className="bar-label" title={m?.note}>
              {m?.label ?? r.model}
            </span>
            <div className="bar-track">
              <div
                className="bar"
                style={{ width: `${Math.max(2, (100 * r[field]) / max)}%`, background: m?.color }}
              />
            </div>
            <span className="bar-val">{fmt(r[field])}</span>
          </div>
        );
      })}
    </div>
  );
}

function MetricsTable({ rows }) {
  const [sortKey, setSortKey] = useState("seconds_per_page");
  const [asc, setAsc] = useState(true);
  const cols = [
    "model",
    "engine",
    "pages",
    "seconds_per_page",
    "total_seconds",
    "total_chars",
    "visible_chars",
    "pct_text",
    "cost_usd",
    "billed_pages",
    "credits",
    "max_gpu_mem_mb",
    "gpu_mem_mb",
    "mean_mean_confidence",
  ];
  const sorted = [...rows].sort((a, b) => {
    const av = a[sortKey] ?? -Infinity;
    const bv = b[sortKey] ?? -Infinity;
    if (av < bv) return asc ? -1 : 1;
    if (av > bv) return asc ? 1 : -1;
    return 0;
  });
  const sortBy = (k) => {
    if (k === sortKey) setAsc(!asc);
    else {
      setSortKey(k);
      setAsc(true);
    }
  };
  return (
    <div className="table-wrap">
      <table className="metrics">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c} onClick={() => sortBy(c)} className={c === sortKey ? "sorted" : ""}>
                {c} {c === sortKey ? (asc ? "▲" : "▼") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={`${r.model}|${r.engine}`} className={(r.artifact ? "artifact " : "") + (r.closed ? "closed-row" : "")}>
              {cols.map((c) => (
                <td key={c}>
                  {c === "model" ? (
                    <span className="cell-model" title={MODEL_BY_ID[r.model]?.note}>
                      <span className="dot" style={{ background: MODEL_BY_ID[r.model]?.color }} />
                      {MODEL_BY_ID[r.model]?.label ?? r.model}
                    </span>
                  ) : c === "cost_usd" ? (
                    r.cost_usd != null ? `$${r.cost_usd}` : "–"
                  ) : (
                    r[c]?.toLocaleString?.() ?? r[c] ?? "–"
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="table-footnote">
        greyed rows are kept engine artifacts, not benchmark rows. <strong>Bold rows are
        paid APIs</strong> — their s/page is whole-PDF wall-clock (not GPU decode) and
        cost_usd is an <em>estimate</em> from build-time rates (see closed_apis/run.py);
        billed_pages/credits are what each API actually reported. The two GPU-memory columns
        are different measurements (per-page peak vs single post-run sample), not merged.
      </p>
    </div>
  );
}

function PricingTable() {
  const [p, setP] = useState(null);
  useEffect(() => {
    fetch("/outputs/closed/pricing.json").then((r) => r.json()).then(setP).catch(() => {});
  }, []);
  if (!p) return null;
  const rows = Object.entries(p.runs);
  return (
    <div className="card">
      <h3>Pricing — paid APIs (verified {p.verified})</h3>
      <p className="note">
        Costs are <strong>measured</strong> wherever the provider reports them. Datalab's API returns
        its real charge; Landing AI's credits are metered; LlamaParse's free tier absorbed the run so
        its column is the published list rate.
      </p>
      <div className="table-wrap">
        <table className="metrics">
          <thead>
            <tr>
              <th>run</th>
              <th>$ / 1k pages</th>
              <th>$ / 68p</th>
              <th>cost source</th>
              <th>tier</th>
            </tr>
          </thead>
          <tbody>
            {rows
              .sort((a, b) => (a[1].usd_per_1k_pages || 0) - (b[1].usd_per_1k_pages || 0))
              .map(([run, r]) => (
                <tr key={run}>
                  <td>
                    <span className="cell-model">
                      <span className="dot" style={{ background: MODEL_BY_ID[run]?.color }} />
                      {MODEL_BY_ID[run]?.label ?? run}
                    </span>
                  </td>
                  <td>{r.usd_per_1k_pages ? `$${r.usd_per_1k_pages.toFixed(2)}` : "not run"}</td>
                  <td>{r.usd_per_1k_pages ? `$${r[`cost_68_pages`]?.toFixed(3)}` : "–"}</td>
                  <td>{r.cost_source}</td>
                  <td>{r.label}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <p className="table-footnote">
        Tiers not run: {Object.entries(p.unrun_tiers_usd_per_page)
          .map(([n, rate]) => `${n} = $${(rate * 1000).toFixed(2)}/1k`)
          .join(" · ")}
      </p>
    </div>
  );
}

function FieldNotes() {
  return (
    <div className="card notes-card">
      <h3>Field notes</h3>
      <p className="note">traps found during the benchmark — read before trusting any single number</p>
      {MODELS.map((m) => (
        <div className="field-note" key={m.id}>
          <span className="cell-model">
            <span className="dot" style={{ background: m.color }} />
            <strong>{m.label}</strong>
          </span>
          <p>{m.note}</p>
        </div>
      ))}
    </div>
  );
}

function MetricsView({ rows, category, setCategory }) {
  // benchmark rows carry their engine label from models.js
  const bench = rows.map((r) => ({
    ...r,
    engine: benchmarkVariant(MODEL_BY_ID[r.model] ?? { variants: [{}] }).label,
  }));
  // non-benchmark engine runs (lightonocr/mineru transformers) join as greyed rows
  const artifactVariants = MODELS.flatMap((m) =>
    m.variants.filter((v) => !v.benchmark).map((v) => ({ model: m.id, ...v }))
  );
  const summaries = useSummaries(artifactVariants.map((v) => v.dir));
  const artifactRows = artifactVariants.flatMap((v) =>
    (summaries[v.dir] ?? []).map((r) => ({
      ...r,
      model: v.model,
      engine: v.label,
      artifact: true,
    }))
  );

  const all = [...bench, ...artifactRows];
  const scoped =
    category === ALL ? aggregate(all) : all.filter((r) => r.pdf === `${category}.pdf`);
  const benchScoped = scoped.filter((r) => !r.artifact);

  return (
    <div className="metrics-view">
      <div className="scope-tabs">
        <button className={category === ALL ? "active" : ""} onClick={() => setCategory(ALL)}>
          All documents
        </button>
        {CATEGORIES.map((c) => (
          <button key={c} className={c === category ? "active" : ""} onClick={() => setCategory(c)}>
            {c.replaceAll("_", " ")}
          </button>
        ))}
      </div>

      <div className="cards">
        <BarCard
          title="Speed"
          note="seconds per page, benchmark engines only — engines differ per model; the table below shows the un-migrated engine runs"
          rows={benchScoped}
          field="seconds_per_page"
          better="low"
          fmt={(v) => `${v}s`}
        />
        <BarCard
          title="Visible text extracted"
          note="markup stripped — the honest ranking (total_chars rewards markup bloat)"
          rows={benchScoped}
          field="visible_chars"
          better="high"
          fmt={(v) => v.toLocaleString()}
        />
        <BarCard
          title="Text density"
          note="% of raw output that is actual text rather than markup (got_ocr's LaTeX isn't stripped, so its % is flattering)"
          rows={benchScoped}
          field="pct_text"
          better="high"
          fmt={(v) => `${v}%`}
        />
        <BarCard
          title="Cost (paid APIs only)"
          note={
            category === ALL
              ? "estimated USD to process all 68 pages — rates are build-time estimates, verify"
              : "estimated USD for this document — rates are build-time estimates, verify"
          }
          rows={benchScoped}
          field="cost_usd"
          better="low"
          fmt={(v) => `$${v.toFixed(category === ALL ? 3 : 4)}`}
        />
      </div>

      <MetricsTable rows={scoped} />
      <PricingTable />
      <FieldNotes />
    </div>
  );
}

/* ---------- app shell ---------- */

export default function App() {
  const [view, setView] = usePersistedState("ocr.view", "compare");
  const [category, setCategoryRaw] = usePersistedState("ocr.category", CATEGORIES[0]);
  const [active, setActive] = usePersistedState("ocr.models", ["lightonocr", "mineru"]);
  const [theme, setTheme] = usePersistedState("ocr.theme", "system"); // system | light | dark
  const rows = useComparison();

  // "system" removes the attribute so the prefers-color-scheme media query takes over
  useEffect(() => {
    const el = document.documentElement;
    if (theme === "system") el.removeAttribute("data-theme");
    else el.setAttribute("data-theme", theme);
  }, [theme]);
  const cycleTheme = () =>
    setTheme((t) => (t === "system" ? "light" : t === "light" ? "dark" : "system"));

  // metrics view allows an extra "all documents" scope; compare view cannot show it
  const compareCategory = CATEGORIES.includes(category) ? category : CATEGORIES[0];
  const setCategory = (c) => setCategoryRaw(c);

  const toggle = (id) =>
    setActive((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));

  useEffect(() => {
    const h = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey) return;
      const i = "12345".indexOf(e.key);
      if (i >= 0 && i < CATEGORIES.length) setCategoryRaw(CATEGORIES[i]);
      if (e.key === "m") setView((v) => (v === "compare" ? "metrics" : "compare"));
      if (e.key === "t") cycleTheme();
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [setCategoryRaw, setView, setTheme]);

  return (
    <div className="app">
      <header>
        <h1>OCR Bench</h1>
        <nav className="view-switch">
          <button className={view === "compare" ? "active" : ""} onClick={() => setView("compare")}>
            Compare
          </button>
          <button className={view === "metrics" ? "active" : ""} onClick={() => setView("metrics")}>
            Metrics
          </button>
        </nav>
        {view === "compare" && (
          <div className="category-tabs">
            {CATEGORIES.map((c, i) => (
              <button
                key={c}
                className={c === compareCategory ? "active" : ""}
                onClick={() => setCategory(c)}
                title={`shortcut: ${i + 1}`}
              >
                {c.replaceAll("_", " ")}
              </button>
            ))}
          </div>
        )}
        <button className="mini theme-toggle" onClick={cycleTheme} title={`theme: ${theme}`}>
          {theme === "dark" ? "🌙" : theme === "light" ? "☀️" : "🖥️"} {theme}
        </button>
        <span className="kbd-hint">1–5 docs · m metrics · t theme</span>
      </header>

      {view === "compare" && (
        <>
          <div className="model-picker">
            {[
              { key: "open", label: "Open source", hint: "self-hosted, GPU", closed: false },
              { key: "paid", label: "Paid API", hint: "hosted, per-page cost", closed: true },
            ].map((group) => {
              const items = MODELS.filter((m) => !!m.closed === group.closed);
              const on = items.filter((m) => active.includes(m.id)).length;
              return (
                <div className="picker-group" key={group.key}>
                  <div className="picker-head">
                    <span className="picker-label">{group.label}</span>
                    <span className="picker-hint">{group.hint}</span>
                    <button
                      className="picker-clear"
                      onClick={() =>
                        setActive((cur) =>
                          on
                            ? cur.filter((id) => !items.some((m) => m.id === id))
                            : [...cur, ...items.map((m) => m.id)]
                        )
                      }
                    >
                      {on ? `clear ${on}` : "all"}
                    </button>
                  </div>
                  <div className="chips">
                    {items.map((m) => (
                      <button
                        key={m.id}
                        className={
                          "chip" +
                          (m.closed ? " closed" : "") +
                          (active.includes(m.id) ? " checked" : "")
                        }
                        onClick={() => toggle(m.id)}
                        title={m.note}
                      >
                        <span className="dot" style={{ background: m.color }} />
                        {m.label}
                        {m.closed && <span className="chip-flag">$</span>}
                        {m.variants.length > 1 && <span className="chip-flag">2 engines</span>}
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          <CompareView category={compareCategory} active={active} toggle={toggle} />
        </>
      )}

      {view === "metrics" && (
        <MetricsView rows={rows} category={category} setCategory={setCategory} />
      )}
    </div>
  );
}
