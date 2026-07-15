import { useEffect, useMemo, useState } from "react";
import PdfViewer from "./PdfViewer";
import Markdown from "./Markdown";
import FolderPicker from "./FolderPicker";

// Point at an INPUT folder of pdfs and an OUTPUT folder of the .md an OCR run produced (either
// mirrored/nested or flat); browse the pairs and read the source pdf next to its markdown. All
// the filesystem work is done by the dev-server /api/* endpoints (see vite.config.js) — the
// pdf is rasterized on the fly with poppler, same as the experiment view.

function usePersisted(key, initial) {
  const [v, setV] = useState(() => localStorage.getItem(key) ?? initial);
  useEffect(() => localStorage.setItem(key, v), [key, v]);
  return [v, setV];
}

// resolve a pdf path -> its page image URLs (via the on-the-fly rasterizer)
function ViewerPdf({ pdfPath }) {
  const [state, setState] = useState({ loading: true, urls: null, error: null });
  useEffect(() => {
    let dead = false;
    setState({ loading: true, urls: null, error: null });
    fetch(`/api/pdf-info?path=${encodeURIComponent(pdfPath)}`)
      .then((r) => r.json())
      .then((m) => {
        if (dead) return;
        if (m.error) throw new Error(m.error);
        const urls = Array.from({ length: m.pages }, (_, i) =>
          `/api/pdf-page?path=${encodeURIComponent(pdfPath)}&n=${i}`
        );
        setState({ loading: false, urls, error: null });
      })
      .catch((e) => !dead && setState({ loading: false, urls: null, error: e.message }));
    return () => { dead = true; };
  }, [pdfPath]);
  return (
    <PdfViewer
      pageUrls={state.urls}
      openHref={`/api/pdf-file?path=${encodeURIComponent(pdfPath)}`}
      loading={state.loading}
      error={state.error}
    />
  );
}

function MdPane({ mdPath }) {
  const [state, setState] = useState({ loading: true, text: "", error: null });
  const [raw, setRaw] = useState(false);
  useEffect(() => {
    let dead = false;
    setState({ loading: true, text: "", error: null });
    if (!mdPath) return setState({ loading: false, text: "", error: "no matching .md in the output folder" });
    fetch(`/api/md?path=${encodeURIComponent(mdPath)}`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(r.statusText))))
      .then((t) => !dead && setState({ loading: false, text: t, error: null }))
      .catch((e) => !dead && setState({ loading: false, text: "", error: e.message }));
    return () => { dead = true; };
  }, [mdPath]);

  return (
    <div className="pane">
      <div className="pane-header">
        <strong>markdown</strong>
        <span className="muted-inline">{mdPath ? mdPath.split("/").pop() : "—"}</span>
        <div className="pane-actions">
          <button className="mini" disabled={!state.text} onClick={() => setRaw((v) => !v)} title="toggle raw source">
            {raw ? "rendered" : "raw"}
          </button>
        </div>
      </div>
      <div className="pane-body">
        {state.loading && <p className="muted">loading…</p>}
        {state.error && <p className="error">{state.error}</p>}
        {!state.loading && !state.error && (raw ? <pre className="raw">{state.text}</pre> : <Markdown text={state.text} />)}
      </div>
    </div>
  );
}

export default function MarkdownViewer() {
  const [input, setInput] = usePersisted("viewer.input", "");
  const [output, setOutput] = usePersisted("viewer.output", "");
  const [docs, setDocs] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(0);
  const [filter, setFilter] = useState("");
  const [picking, setPicking] = useState(null); // "input" | "output" | null

  const load = () => {
    setLoading(true);
    setError(null);
    setDocs(null);
    fetch(`/api/browse?input=${encodeURIComponent(input)}&output=${encodeURIComponent(output)}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.error) throw new Error(d.error);
        setDocs(d.docs);
        setSelected(0);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const shown = useMemo(
    () => (docs || []).filter((d) => d.rel.toLowerCase().includes(filter.toLowerCase())),
    [docs, filter]
  );
  const doc = shown[selected] ?? shown[0];
  const matched = (docs || []).filter((d) => d.md).length;

  return (
    <div className="viewer">
      {picking && (
        <FolderPicker
          start={picking === "input" ? input : output}
          onClose={() => setPicking(null)}
          onPick={(p) => {
            picking === "input" ? setInput(p) : setOutput(p);
            setPicking(null);
          }}
        />
      )}
      <div className="viewer-bar">
        <label>
          input pdfs
          <div className="path-input">
            <input value={input} onChange={(e) => setInput(e.target.value)}
                   placeholder="/abs/path/to/input_folder" onKeyDown={(e) => e.key === "Enter" && load()} />
            <button className="mini browse" onClick={() => setPicking("input")} title="browse folders">📁</button>
          </div>
        </label>
        <label>
          output markdown
          <div className="path-input">
            <input value={output} onChange={(e) => setOutput(e.target.value)}
                   placeholder="/abs/path/to/outputs/run" onKeyDown={(e) => e.key === "Enter" && load()} />
            <button className="mini browse" onClick={() => setPicking("output")} title="browse folders">📁</button>
          </div>
        </label>
        <button className="load-btn" onClick={load} disabled={!input || !output || loading}>
          {loading ? "loading…" : "Load"}
        </button>
        {docs && (
          <span className="viewer-count">
            {docs.length} pdfs · {matched} matched to markdown
          </span>
        )}
      </div>

      {error && <p className="error viewer-error">{error}</p>}

      {docs && (
        <div className="viewer-body">
          <div className="viewer-list">
            <input className="viewer-filter" placeholder="filter…" value={filter}
                   onChange={(e) => { setFilter(e.target.value); setSelected(0); }} />
            <div className="viewer-docs">
              {shown.map((d, i) => (
                <button key={d.rel} className={"viewer-doc" + (d === doc ? " active" : "")}
                        onClick={() => setSelected(i)} title={d.rel}>
                  <span className={"doc-dot" + (d.md ? " ok" : " missing")} />
                  {d.rel}
                </button>
              ))}
              {!shown.length && <p className="muted">no documents match</p>}
            </div>
          </div>

          <div className="viewer-panes">
            {doc ? (
              <>
                <div className="pane pdf-pane">
                  <div className="pane-header">
                    <strong>{doc.rel}</strong>
                    <span className="muted-inline">source pdf</span>
                  </div>
                  <ViewerPdf pdfPath={doc.pdf} />
                </div>
                <MdPane mdPath={doc.md} />
              </>
            ) : (
              <div className="empty-hint"><p>Pick a document from the left.</p></div>
            )}
          </div>
        </div>
      )}

      {!docs && !error && !loading && (
        <div className="viewer-hello">
          <p>Enter an <strong>input folder</strong> of PDFs and the <strong>output folder</strong> of
            an OCR run's <code>.md</code> files, then Load.</p>
          <p className="muted">Both may be nested or flat. Documents are matched by filename; a
            mirrored or <code>inference/</code>-style tree is preferred when names collide.</p>
        </div>
      )}
    </div>
  );
}
