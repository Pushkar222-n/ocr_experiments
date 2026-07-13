import { useEffect, useRef, useState } from "react";

// Shows pre-rasterized page PNGs (poppler, see rasterize.sh) instead of rendering the
// PDF client-side. pdf.js silently fails on the CCITT Group-4 fax scans these documents
// use — render "succeeds" but the page is blank white. Serving images sidesteps that and
// every other pdf.js quirk (canvas-memory limits, render races), and native
// loading="lazy" bounds memory for free.
let manifestPromise = null;
const loadManifest = () => (manifestPromise ??= fetch("/pdf-pages/manifest.json").then((r) => r.json()));

export default function PdfViewer({ stem }) {
  const scrollRef = useRef(null);
  const [pages, setPages] = useState(null);
  const [error, setError] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [current, setCurrent] = useState(1);

  useEffect(() => {
    let dead = false;
    setPages(null);
    setError(null);
    setCurrent(1);
    loadManifest()
      .then((m) => {
        if (dead) return;
        if (!m[stem]) throw new Error(`no rasterized pages for "${stem}" — run frontend/rasterize.sh`);
        setPages(m[stem]);
        if (scrollRef.current) scrollRef.current.scrollTop = 0;
      })
      .catch((e) => !dead && setError(e.message));
    return () => {
      dead = true;
    };
  }, [stem]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const mid = el.scrollTop + el.clientHeight / 3;
    let page = 1;
    for (const child of el.querySelectorAll(".pdf-page")) {
      if (child.offsetTop <= mid) page = +child.dataset.page;
      else break;
    }
    setCurrent(page);
  };

  const jumpTo = (n) => {
    const clamped = Math.min(Math.max(1, n), pages?.length ?? 1);
    scrollRef.current
      ?.querySelector(`.pdf-page[data-page="${clamped}"]`)
      ?.scrollIntoView({ block: "start" });
  };

  return (
    <div className="pdf-viewer">
      <div className="pdf-controls">
        <input
          type="number"
          value={current}
          min={1}
          max={pages?.length ?? 1}
          onChange={(e) => {
            const n = +e.target.value || 1;
            setCurrent(n);
            jumpTo(n);
          }}
        />
        <span className="pdf-total">/ {pages?.length ?? "…"}</span>
        <span className="spacer" />
        <button className="mini" onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}>
          −
        </button>
        <button className="mini" onClick={() => setZoom(1)} title="fit width">
          {Math.round(zoom * 100)}%
        </button>
        <button className="mini" onClick={() => setZoom((z) => Math.min(3, z + 0.25))}>
          +
        </button>
      </div>
      <div className="pdf-canvas-wrap" ref={scrollRef} onScroll={onScroll}>
        {error && <p className="error">{error}</p>}
        {!pages && !error && <p className="muted">loading pages…</p>}
        {pages &&
          pages.map((file, i) => (
            <div className="pdf-page" key={file} data-page={i + 1} style={{ width: `${zoom * 100}%` }}>
              <img src={`/pdf-pages/${file}`} loading="lazy" alt={`page ${i + 1}`} />
              <span className="pdf-pagenum">{i + 1}</span>
            </div>
          ))}
      </div>
    </div>
  );
}
