// Mirrors scripts/compare.py's engine-tag rule: lightonocr and mineru were re-run on
// vLLM into an output_vllm/ subfolder, and that tagged run is the benchmark row.
// The untagged transformers runs are kept as engine artifacts — worth comparing,
// especially for mineru where the engine changes the *output*, not just the speed.
// color: stable per-model hue used in chips, badges and metric bars.
// note: field notes distilled from CLAUDE.md — traps a reader needs before trusting numbers.
export const MODELS = [
  {
    id: "chandra",
    label: "Chandra",
    color: "#e5484d",
    note:
      "Best diagram parser in the field: recovers flowchart topology as mermaid, including " +
      "the QC feedback loops every other model drops (24 nodes / 55 edges / 10 loops on " +
      "Flowchart vs MinerU's 21/40/0). But it emits the graph in a bare code fence, so it " +
      "won't auto-render — look at the raw view. Heaviest runtime: carries its own vLLM, ~38 GB VRAM.",
    variants: [{ id: "vllm", label: "vLLM", dir: "chandra", benchmark: true }],
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
    label: "Datalab Marker",
    color: "#d13b8f",
    closed: true,
    note:
      "Hosted Marker (use_llm=false, the non-LLM base tier, ~$3/1k pages). THE BEST VALUE IN THE " +
      "WHOLE FIELD: most visible text of anything here, open or closed (149,941 over 68 pages, vs " +
      "the best open model MinerU at 128k) AND the cheapest paid API ($0.20/68p) AND near-fastest " +
      "(0.96 s/page). Recovers flowchart edges. HTML tables inflate its raw count. Upgrade path: " +
      "High Accuracy (use_llm=true) at $6/1k. Note: base rate is the one unconfirmed number in the " +
      "price table — Datalab doesn't publish it fetchably.",
    variants: [{ id: "api", label: "API", dir: "closed/datalab", benchmark: true }],
  },
  {
    id: "llamaparse",
    label: "LlamaParse",
    color: "#8b5cf6",
    closed: true,
    note:
      "LlamaIndex, Balanced preset = parse_page_with_agent (gemini-2.5-flash) = 10 credits/page = " +
      "$12.50/1k — 4x Datalab for slightly LESS text. Second-highest visible text (144k) and the most " +
      "flowchart edges recovered (58), but BY FAR the slowest API: 14 s/page (568s on the 32-page doc). " +
      "The free tier (10k credits/mo) absorbed this run, so the API reported 0 credits and the cost " +
      "shown is the list rate. Cheaper tiers: Fast 1cr/pg ($1.25/1k), Cost-effective 3cr/pg ($3.75/1k). " +
      "Premium Agentic Plus is 45cr/pg = $56/1k.",
    variants: [{ id: "api", label: "API", dir: "closed/llamaparse", benchmark: true }],
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
