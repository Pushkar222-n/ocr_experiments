import { useEffect, useState } from "react";
import PdfViewer from "./PdfViewer";

// The experiment view's pages are poppler-rasterized at build time (frontend/rasterize.sh)
// and listed in pdf-pages/manifest.json. Resolve a category stem to its page URLs, then hand
// them to the shared PdfViewer.
let manifestPromise = null;
const loadManifest = () =>
  (manifestPromise ??= fetch("/pdf-pages/manifest.json").then((r) => r.json()));

export default function ExperimentPdf({ stem }) {
  const [state, setState] = useState({ loading: true, urls: null, error: null });

  useEffect(() => {
    let dead = false;
    setState({ loading: true, urls: null, error: null });
    loadManifest()
      .then((m) => {
        if (dead) return;
        if (!m[stem]) throw new Error(`no rasterized pages for "${stem}" — run frontend/rasterize.sh`);
        setState({ loading: false, urls: m[stem].map((f) => `/pdf-pages/${f}`), error: null });
      })
      .catch((e) => !dead && setState({ loading: false, urls: null, error: e.message }));
    return () => {
      dead = true;
    };
  }, [stem]);

  return (
    <PdfViewer
      pageUrls={state.urls}
      openHref={`/pdfs/${stem}.pdf`}
      loading={state.loading}
      error={state.error}
    />
  );
}
