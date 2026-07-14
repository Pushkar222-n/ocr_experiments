import { useEffect, useRef, useState } from "react";

// Shows pre-rasterized page PNGs (poppler, see rasterize.sh) instead of rendering the PDF
// client-side. pdf.js silently fails on the CCITT Group-4 fax scans these documents use —
// render "succeeds" but the page is blank white. Serving images sidesteps that and every
// other pdf.js quirk (canvas-memory limits, render races); native loading="lazy" bounds
// memory for free.
let manifestPromise = null;
const loadManifest = () =>
  (manifestPromise ??= fetch("/pdf-pages/manifest.json").then((r) => r.json()));

const ZOOMS = [0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4];

// One page. Rotation needs the image's real aspect: a 90-degree turn swaps the box, and the
// pages here are NOT uniform (Complex_table_layouts mixes portrait scans with landscape
// tables), so we read each image's own naturalWidth/Height on load rather than assuming A4.
function Page({ file, num, width, rotate }) {
  const [aspect, setAspect] = useState(null); // h / w
  const turned = rotate === 90 || rotate === 270;
  const boxW = turned && aspect ? width * aspect : width;
  const boxH = aspect ? (turned ? width : width * aspect) : undefined;

  return (
    <div
      className="pdf-page"
      data-page={num}
      style={{ width: boxW, height: boxH, minHeight: aspect ? undefined : 200 }}
    >
      <img
        src={`/pdf-pages/${file}`}
        loading="lazy"
        alt={`page ${num}`}
        onLoad={(e) => setAspect(e.target.naturalHeight / e.target.naturalWidth)}
        style={{
          width: width,
          height: aspect ? width * aspect : "auto",
          transform: rotate ? `rotate(${rotate}deg)` : undefined,
        }}
      />
      <span className="pdf-pagenum">{num}</span>
    </div>
  );
}

export default function PdfViewer({ stem }) {
  const scrollRef = useRef(null);
  const [pages, setPages] = useState(null);
  const [error, setError] = useState(null);
  const [zoom, setZoom] = useState(1); // 1 = fit width
  const [rotate, setRotate] = useState(0);
  const [current, setCurrent] = useState(1);
  const [baseW, setBaseW] = useState(0); // pane width = the "fit width" reference

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

  // track the pane width so "fit width" stays correct as panes open/close
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => setBaseW(Math.max(160, el.clientWidth - 28));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [pages]);

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
    setCurrent(clamped);
    scrollRef.current
      ?.querySelector(`.pdf-page[data-page="${clamped}"]`)
      ?.scrollIntoView({ block: "start", behavior: "smooth" });
  };

  const step = (dir) =>
    setZoom((z) => {
      const i = ZOOMS.findIndex((v) => v >= z - 1e-6);
      return ZOOMS[Math.min(ZOOMS.length - 1, Math.max(0, i + dir))] ?? z;
    });

  const total = pages?.length ?? 0;
  const width = baseW * zoom;

  return (
    <div className="pdf-viewer">
      <div className="pdf-controls">
        <button className="mini" onClick={() => jumpTo(current - 1)} disabled={current <= 1} title="previous page">
          ‹
        </button>
        <input
          type="number"
          value={current}
          min={1}
          max={total || 1}
          onChange={(e) => jumpTo(+e.target.value || 1)}
          title="jump to page"
        />
        <span className="pdf-total">/ {total || "…"}</span>
        <button
          className="mini"
          onClick={() => jumpTo(current + 1)}
          disabled={!total || current >= total}
          title="next page"
        >
          ›
        </button>

        <span className="spacer" />

        <button className="mini" onClick={() => step(-1)} disabled={zoom <= ZOOMS[0]} title="zoom out">
          −
        </button>
        <button className="mini" onClick={() => setZoom(1)} title="reset to fit width">
          {Math.round(zoom * 100)}%
        </button>
        <button
          className="mini"
          onClick={() => step(1)}
          disabled={zoom >= ZOOMS[ZOOMS.length - 1]}
          title="zoom in"
        >
          +
        </button>
        <button
          className={"mini" + (rotate ? " on" : "")}
          onClick={() => setRotate((r) => (r + 90) % 360)}
          title="rotate 90° (these documents contain sideways scanned tables)"
        >
          ⟳
        </button>
        <a className="mini" href={`/pdfs/${stem}.pdf`} target="_blank" rel="noreferrer" title="open the original PDF">
          ↗
        </a>
      </div>

      <div className="pdf-canvas-wrap" ref={scrollRef} onScroll={onScroll}>
        {error && <p className="error">{error}</p>}
        {!pages && !error && <p className="muted">loading pages…</p>}
        {pages && baseW > 0 && (
          <div className="pdf-pages">
            {pages.map((file, i) => (
              <Page key={file} file={file} num={i + 1} width={width} rotate={rotate} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
