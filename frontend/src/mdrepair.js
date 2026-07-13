// Best-effort repair of broken markdown in OCR output, applied only to the
// rendered view (the raw view always shows the true file bytes).
// For well-formed files every pass is a no-op — verified against all 9 models.

export const DIAGRAM_START =
  /^\s*(graph\s+(TD|TB|LR|RL|BT)\b|flowchart\s+\w+|sequenceDiagram|stateDiagram)/;
const FENCE = /^\s*```(.*)$/;

const MERMAID_KEYWORD =
  /^\s*(subgraph\b|end\b|style\b|classDef\b|class\b|linkStyle\b|graph\b|flowchart\b|direction\b|%%)/;

// a line that clearly isn't part of a mermaid graph body
function isProse(line) {
  const t = line.trim();
  if (!t) return false; // blank lines occur inside graphs
  if (/^(#|---|\*\*|!|\|)/.test(t)) return true;
  if (MERMAID_KEYWORD.test(t)) return false;
  if (/-->|\[|\]/.test(t)) return false; // edges / node defs
  return /\s/.test(t); // multi-word line with no mermaid syntax
}

// lightonocr mangles fences three ways (all observed in outputs/lightonocr/*):
//   1. opener written as a bare "mermaid" word line: `mermaid\ngraph TD`
//   2. opener missing entirely: bare `graph TD` mid-prose
//   3. closer missing: graph body runs straight into prose
// plus orphan closing ``` whose opener was dropped — left alone it OPENS a code
// block that swallows the rest of the document.
export function repairFences(src) {
  const lines = src.split("\n");
  const out = [];
  const drop = new Set();
  // state: null | {type:"lang", lang} | {type:"bare", idx: index-in-out}
  let state = null;

  const openMermaid = () => {
    if (state?.type === "bare") drop.add(state.idx); // that fence was an orphan closer
    if (state?.type === "lang") out.push("```"); // close an unterminated block first
    out.push("```mermaid");
    state = { type: "lang", lang: "mermaid" };
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const m = line.match(FENCE);

    if (!m && /^\s*mermaid\s*$/.test(line) && DIAGRAM_START.test(lines[i + 1] ?? "")) {
      openMermaid(); // case 1: the word "mermaid" standing in for ```mermaid
      continue;
    }
    if (m) {
      const lang = m[1].trim();
      if (!state) {
        out.push(line);
        state = lang ? { type: "lang", lang } : { type: "bare", idx: out.length - 1 };
      } else if (lang) {
        // a definite ```lang opener arrived while a block was still open
        if (state.type === "bare") drop.add(state.idx); // orphan closer
        else out.push("```");
        out.push(line);
        state = { type: "lang", lang };
      } else {
        out.push(line); // proper close
        state = null;
      }
      continue;
    }
    if (!state && DIAGRAM_START.test(line) && !FENCE.test(out[out.length - 1] ?? "")) {
      openMermaid(); // case 2: unfenced diagram start
      out.push(line);
      continue;
    }
    if (state?.type === "lang" && state.lang === "mermaid" && isProse(line)) {
      out.push("```"); // case 3: graph ended without a closer
      state = null;
    }
    out.push(line);
  }
  if (state?.type === "bare") drop.add(state.idx);
  else if (state?.type === "lang") out.push("```");

  return out.filter((_, i) => !drop.has(i)).join("\n");
}

// got_ocr / unlimited_ocr emit LaTeX with \(...\) and \[...\] delimiters;
// remark-math only understands $...$ / $$...$$. Applied outside code fences.
//
// chandra is different: it wraps LaTeX in an HTML <math> tag (e.g. `<math>\pm</math>`),
// almost always *inside* an HTML <td>. The browser parses <math> as MathML, can't read
// the LaTeX, and shows nothing; remark-math never tokenizes inside raw HTML either. So
// rewrite <math> to a <span class="math-inline"> — rehype-katex renders any element with
// that class, and it runs after rehype-raw has parsed the span into a real node, so it
// works even inside a table cell.
export function normalizeMathDelims(src) {
  const out = [];
  let buf = [];
  let inFence = false;
  const flush = () => {
    if (!buf.length) return;
    let t = buf.join("\n");
    if (!inFence) {
      t = t
        .replace(/<math\s+display\s*=\s*["']?block["']?[^>]*>([\s\S]*?)<\/math>/gi,
                 (_, x) => `<div class="math-display">${x}</div>`)
        .replace(/<math\b[^>]*>([\s\S]*?)<\/math>/gi,
                 (_, x) => `<span class="math-inline">${x}</span>`)
        .replace(/\\\[([\s\S]+?)\\\]/g, (_, x) => `\n$$\n${x}\n$$\n`)
        .replace(/\\\(([\s\S]+?)\\\)/g, (_, x) => `$${x}$`);
    }
    out.push(t);
    buf = [];
  };
  for (const l of src.split("\n")) {
    if (FENCE.test(l)) {
      flush();
      inFence = !inFence;
      out.push(l);
    } else buf.push(l);
  }
  flush();
  return out.join("\n");
}

export const prepareMarkdown = (src) => normalizeMathDelims(repairFences(src));
