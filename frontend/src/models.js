// Mirrors scripts/compare.py's engine-tag rule: lightonocr and mineru were re-run on
// vLLM into an output_vllm/ subfolder, and that tagged run is the benchmark row.
// The untagged transformers runs are kept as engine artifacts — worth comparing,
// especially for mineru where the engine changes the *output*, not just the speed.
// color: stable per-model hue used in chips, badges and metric bars.
// note: field notes distilled from CLAUDE.md — traps a reader needs before trusting numbers.
export const MODELS = [
  {
    id: "chandra",
    label: "Chandra (baseline)",
    color: "#e5484d",
    note:
      "THE MODEL THIS PROJECT SETTLED ON — but compare against 'Chandra oriented+tuned', which " +
      "is the config actually shipped in inference/ and beats this by +3.1% text for free. " +
      "Best diagram parser in the field: recovers flowchart topology as mermaid including the QC " +
      "feedback loops every other model drops (24 nodes / 55 edges / 10 loops vs MinerU's " +
      "21/40/0). It emits the graph in a bare code fence, so it won't auto-render — look at raw. " +
      "Weights are only 8.61 GiB; the ~39 GB on the card is KV-cache reservation, not demand.",
    variants: [{ id: "vllm", label: "vLLM", dir: "chandra", benchmark: true }],
  },
  {
    id: "chandra_oriented_optimized",
    label: "Chandra oriented+tuned",
    color: "#f43f5e",
    note:
      "THE PRODUCTION CONFIG (this is what inference/ runs). Chandra with page-rotation " +
      "correction on and the tuned vLLM flags. Best chandra run: 112,486 visible chars vs the " +
      "baseline's 109,101 (+3.1%) at the SAME speed (9.72 vs 9.80 s/page) — the rotation fix is " +
      "free. Isolating it with per-page token counts: +4.8% on the 7 rotated pages of " +
      "Complex_table_layouts against −0.7% on that same document's unrotated pages (control), so " +
      "the gain is real and attributable, not noise. Flowchart also improved: 36 → 46 mermaid edges.",
    variants: [{ id: "vllm", label: "vLLM", dir: "chandra_oriented_optimized", benchmark: true }],
  },
  {
    id: "chandra_dpi256",
    label: "Chandra DPI-256",
    color: "#9f1239",
    note:
      "A TESTED DEAD END — kept so nobody re-runs it. Raising the render DPI from chandra's " +
      "default 192 to 256 (which saturates the model's pixel cap, 74% more vision tokens) is " +
      "**worse**: 111,343 visible chars vs the oriented run's 112,486, and 35% slower (13.15 vs " +
      "9.72 s/page). Dense tables gain slightly (+480) but Flowchart — chandra's best document " +
      "class — COLLAPSES, losing 2,048 chars and 6 mermaid edges. More pixels is not more signal; " +
      "the vision encoder degrades at the edge of its budget. 192 is a tuned default, not a lazy one.",
    variants: [{ id: "vllm", label: "vLLM", dir: "chandra_dpi256", benchmark: true }],
  },
  {
    id: "dots_ocr",
    label: "dots.mocr",
    color: "#f76b15",
    note:
      "Layout-first pipeline; native output is layout JSON (bbox + category + text). " +
      "Declines to OCR inside diagrams — flowchart shapes become Picture elements with no " +
      "text at all (626 chars on Flowchart vs LightOnOCR's 4,846). Fine on tables and text.",
    variants: [{ id: "transformers", label: "transformers", dir: "dots_ocr", benchmark: true }],
  },
  {
    id: "glm_ocr",
    label: "GLM-OCR",
    color: "#ffb224",
    note:
      "Emits HTML tables, not markdown pipes — raw char counts are markup-inflated " +
      "(39–83% text). Skips diagrams: crops them out as images and never reads inside. " +
      "Heaviest VRAM in the set (38.3 GB). Its image links are dangling by design; text is intact.",
    variants: [{ id: "vllm", label: "vLLM", dir: "glm_ocr", benchmark: true }],
  },
  {
    id: "got_ocr",
    label: "GOT-OCR",
    color: "#46a758",
    note:
      "Native format is mathpix LaTeX despite the .md extension. Degenerates into token " +
      "loops on 5 of 68 pages (each runs to the 4096-token cap), so its char counts " +
      "overstate real extraction — and its 96% text ratio is an artifact: the tag-stripper " +
      "doesn't remove LaTeX. Collapses on printouts and flowcharts; competitive on formulas.",
    variants: [{ id: "transformers", label: "transformers", dir: "got_ocr", benchmark: true }],
  },
  {
    id: "lightonocr",
    label: "LightOnOCR",
    color: "#12a594",
    note:
      "Fastest model in the benchmark on vLLM (2.8 s/page). No layout stage — it reads the " +
      "whole page as pixels, so it never skips a diagram (though it returns diagram text as " +
      "loose prose, with no graph structure). Tables come out as HTML. The two engines here " +
      "produce the same output; only the speed differs (5.4x) — the control case for mineru's engine finding.",
    variants: [
      { id: "vllm", label: "vLLM", dir: "lightonocr/output_vllm", benchmark: true },
      { id: "transformers", label: "transformers", dir: "lightonocr" },
    ],
  },
  {
    id: "mineru",
    label: "MinerU",
    color: "#0091ff",
    note:
      "Extracts the most real text of all 9 models (128k visible chars, 17% clear of next " +
      "best) — but only on vLLM. The transformers engine silently drops MinerU's own " +
      "presence/frequency penalties: 6.4x slower AND worse output (gives up on flowcharts, " +
      "emitting prose summaries instead of mermaid). Toggle the engine here to see it. " +
      "Reads flowcharts as mermaid, though forward-only — it misses feedback loops Chandra catches.",
    variants: [
      { id: "vllm", label: "vLLM", dir: "mineru/output_vllm", benchmark: true },
      { id: "transformers", label: "transformers", dir: "mineru" },
    ],
  },
  {
    id: "paddleocr_vl",
    label: "PaddleOCR-VL",
    color: "#6e56cf",
    note:
      "The trap model: highest raw char count in the benchmark (423k) but 8th of 9 on " +
      "visible text — it puts inline CSS on every single <td>, so 79% of its output is " +
      "markup. Rank it on visible chars, never total. Skips diagrams (emits <img> refs). " +
      "Best-in-field on printouts.",
    variants: [{ id: "paddle", label: "paddle", dir: "paddleocr_vl", benchmark: true }],
  },
  {
    id: "surya",
    label: "Surya",
    color: "#d6409f",
    note:
      "Second-fastest (4.1 s/page) and the only model that self-reports confidence — but " +
      "that confidence is NOT a coverage metric: it's page-level (every block on a page " +
      "carries the identical value) and it scores only what surya chose to read. On " +
      "Flowchart it dropped the entire diagram and still reported 0.947. Never use it as a quality gate.",
    variants: [{ id: "vllm", label: "vLLM", dir: "surya", benchmark: true }],
  },
  {
    id: "unlimited_ocr",
    label: "Unlimited OCR",
    color: "#8d8d8d",
    note:
      "Native format is grounding tags + HTML tables. Needs a capped max_length (4096) or " +
      "some pages decode forever; 6 of 68 pages hit that cap (all in Complex_table_layouts) " +
      "and are UNDER-counted — the post-processor discards most of a degenerate span. " +
      "Skips diagrams (classifies them as pictures).",
    variants: [{ id: "transformers", label: "transformers", dir: "unlimited_ocr", benchmark: true }],
  },

  // ---- closed / paid APIs (balanced tier), stored under outputs/closed/<provider>/ ----
  // These report billed pages / credits and an estimated cost_usd (see closed_apis/run.py).
  {
    id: "mistral",
    label: "Mistral OCR",
    color: "#fa5305",
    closed: true,
    note:
      "mistral-ocr-latest, $4/1k pages ($0.27 for all 68). FASTEST thing in the benchmark, " +
      "open or closed: 0.49 s/page wall (whole-PDF in one call). Solid on text/tables/formulas " +
      "($-delim LaTeX). But NOT a diagram parser — flattened the flowchart to 724 chars with zero " +
      "arrows. Upgrade path: Document AI at $5/1k.",
    variants: [{ id: "api", label: "API", dir: "closed/mistral", benchmark: true }],
  },
  {
    id: "datalab",
    label: "Datalab (legacy /marker)",
    color: "#d13b8f",
    closed: true,
    note:
      "The DEPRECATED /api/v1/marker endpoint (superseded by /api/v1/convert — see the two " +
      "Datalab mode entries). Still the best value in the whole field: 149,941 visible chars, more " +
      "than any open model (MinerU 128k), at $4.26/1k pages and 0.96 s/page. It bills IDENTICALLY " +
      "to mode=balanced (13/2/5/6/3 cents on the same 5 pdfs) and extracts marginally more text — " +
      "so there is no cost reason to migrate, only an API-lifecycle one.",
    variants: [{ id: "api", label: "API", dir: "closed/datalab", benchmark: true }],
  },
  {
    id: "datalab_balanced",
    label: "Datalab balanced",
    color: "#ec4899",
    closed: true,
    note:
      "Current API: /api/v1/convert, mode=balanced. $4.26/1k pages — and that is MEASURED, not " +
      "estimated: this endpoint returns its own charge in cost_breakdown.final_cost_cents. " +
      "147,416 visible chars, 1.10 s/page. Recovers flowchart edges (39). Statistically tied with " +
      "the legacy endpoint on text, at the same price.",
    variants: [{ id: "api", label: "API", dir: "closed/datalab_balanced", benchmark: true }],
  },
  {
    id: "datalab_accurate",
    label: "Datalab accurate",
    color: "#be123c",
    closed: true,
    note:
      "Current API: /api/v1/convert, mode=accurate. 2.35x the price of balanced ($10/1k vs $4.26) " +
      "and it buys NOTHING measurable on this document set: 148,848 visible chars vs balanced's " +
      "147,416 (+1%) and the LEGACY endpoint's 149,941 (-0.7%). Same 39 flowchart edges. 2.2x slower. " +
      "On these 68 pages, paying for 'accurate' is not justified — verify on your own docs before " +
      "assuming the top tier is worth it.",
    variants: [{ id: "api", label: "API", dir: "closed/datalab_accurate", benchmark: true }],
  },
  {
    id: "llamaparse",
    label: "LlamaParse agentic",
    color: "#8b5cf6",
    closed: true,
    note:
      "tier=agentic (the 'Balanced' preset) = 10 credits/page = $12.50/1k — 3x Datalab for LESS text " +
      "(144k). BY FAR the slowest API: 14 s/page (568s on the 32-page doc). BUT it is the best " +
      "diagram parser of the paid set: 3 real ```mermaid blocks and 55 edges on Flowchart, the only " +
      "closed model that fences the graph properly. Free tier (10k cr/mo) absorbed the run, so the " +
      "API honestly reported 0 credits and the cost shown is list rate.",
    variants: [{ id: "api", label: "API", dir: "closed/llamaparse", benchmark: true }],
  },
  {
    id: "llamaparse_agentic_plus",
    label: "LlamaParse agentic+",
    color: "#6d28d9",
    closed: true,
    note:
      "tier=agentic_plus (premium) = 45 credits/page = $56.25/1k — the MOST EXPENSIVE run in this " +
      "repo ($3.83 for 68 pages, 19x Datalab). It does extract the most visible text of anything, " +
      "open or closed (167,108). BUT it is a WORSE diagram parser than the cheaper agentic tier: it " +
      "flattens the flowchart into 72 TABLE ROWS and emits ZERO graph syntax (0 mermaid, 0 arrows) " +
      "where plain agentic gave 55 edges. More money bought more text and destroyed the topology. " +
      "Worst cost-per-text of the set ($0.229/10k visible, tied with landing_ai).",
    variants: [{ id: "api", label: "API", dir: "closed/llamaparse_agentic_plus", benchmark: true }],
  },
  {
    id: "landing_ai",
    label: "Landing AI ADE",
    color: "#059669",
    closed: true,
    note:
      "Agentic Document Extraction (dpt-2-latest). THE WORST VALUE HERE BY A WIDE MARGIN: metered at " +
      "exactly 3 credits/page x $0.01 = $30/1k pages — 10x Datalab's price for 40% LESS text (90k " +
      "visible, lowest of the four). That's $0.226 per 10k visible chars vs Datalab's $0.014, a 16x gap. " +
      "Sparsest formula math; only partially recovers the flowchart. Does return grounded chunks with " +
      "per-block page refs. Team plan ($250/mo) only drops it to ~$27/1k.",
    variants: [{ id: "api", label: "API", dir: "closed/landing_ai", benchmark: true }],
  },
];

export const MODEL_BY_ID = Object.fromEntries(MODELS.map((m) => [m.id, m]));

export const CATEGORIES = [
  "Complex_table_layouts",
  "Flowchart",
  "Formulas_with_tables",
  "Handwritten",
  "printouts",
];
