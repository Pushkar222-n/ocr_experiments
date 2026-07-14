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

// ---------------------------------------------------------------- LaTeX documents
// got_ocr's native format is mathpix LaTeX, not markdown, despite the .md extension:
// \title{}, \section*{}, and \begin{tabular} (266 of them in one file). KaTeX only does
// *math* — `tabular` is a document environment it will never render — so without this the
// whole model reads as raw LaTeX noise. Convert the document structure to markdown/HTML and
// leave the math to KaTeX.

const looksLikeLatexDoc = (s) =>
  /\\begin\{tabular\}|\\title\{|\\section\*?\{/.test(s);

// split on a delimiter that isn't backslash-escaped (\& is a literal ampersand)
const splitUnescaped = (s, ch) =>
  s.split(new RegExp(`(?<!\\\\)\\${ch}`));

// read a {...} group starting at i (which must be '{'), respecting nesting.
// returns [content, indexAfterClosingBrace] — a regex can't do this: cell content like
// \multicolumn{2}{|c|}{ Total {net} } has nested braces and greedy/lazy both get it wrong.
function readBraced(s, i) {
  if (s[i] !== "{") return [null, i];
  let depth = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === "{" && s[j - 1] !== "\\") depth++;
    else if (s[j] === "}" && s[j - 1] !== "\\") {
      depth--;
      if (depth === 0) return [s.slice(i + 1, j), j + 1];
    }
  }
  return [s.slice(i + 1), s.length]; // unterminated — take the rest
}

// \multicolumn{n}{spec}{content} -> colspan, \multirow{n}{*}{content} -> rowspan
function parseCell(raw) {
  let cell = raw.trim();
  let attrs = "";
  for (let guard = 0; guard < 4; guard++) {
    const m = cell.match(/^\\(multicolumn|multirow)\s*/);
    if (!m) break;
    let i = m[0].length;
    const [n, i1] = readBraced(cell, i);
    const [, i2] = readBraced(cell, i1);      // the spec — discarded
    const [content, i3] = readBraced(cell, i2);
    if (n === null || content === null) break;
    attrs += m[1] === "multicolumn" ? ` colspan="${n.trim()}"` : ` rowspan="${n.trim()}"`;
    cell = (content + cell.slice(i3)).trim();
  }
  return `<td${attrs}>${cell}</td>`;
}

function latexRowsToHtml(body) {
  const rows = body
    .split(/\\\\/) // rows end with \\
    .map((r) => r.replace(/\\hline|\\cline\s*\{[^}]*\}|\\toprule|\\midrule|\\bottomrule/g, "").trim())
    .filter((r) => r.length);
  const html = rows.map(
    (row) => `<tr>${splitUnescaped(row, "&").map(parseCell).join("")}</tr>`
  );
  return `<table>${html.join("")}</table>`;
}

export function latexToMarkdown(src) {
  if (!looksLikeLatexDoc(src)) return src;
  let t = src;

  // math first, as span/div (works inside the HTML tables we are about to build — remark-math
  // would never tokenize $...$ in there, same reason as chandra's <math> tags)
  t = t
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, x) => `\n<div class="math-display">${x.trim()}</div>\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, x) => `<span class="math-inline">${x.trim()}</span>`);

  // tabular -> html table, innermost first so nested tabulars resolve bottom-up
  const INNERMOST = /\\begin\{tabular\}\s*(?:\{[^}]*\})?([\s\S]*?)\\end\{tabular\}/;
  for (let guard = 0; guard < 2000 && INNERMOST.test(t); guard++) {
    t = t.replace(INNERMOST, (_, body) => latexRowsToHtml(body));
  }
  // got_ocr degenerates into token loops on 5/68 pages and gets cut at the 4096-token cap
  // mid-table, leaving \begin{tabular} with no \end (168 vs 166 in Complex_table_layouts).
  // Render what it did emit rather than dumping raw latex at the reader.
  t = t.replace(/\\begin\{tabular\}\s*(?:\{[^}]*\})?([\s\S]*)$/,
                (_, body) => latexRowsToHtml(body) +
                  `\n\n> *(table truncated — the model hit its token cap here)*\n`);

  // document structure -> markdown headings
  t = t
    .replace(/\\title\s*\{([\s\S]*?)\}/g, (_, x) => `\n# ${x.trim()}\n`)
    .replace(/\\(?:sub){2}section\*?\s*\{([\s\S]*?)\}/g, (_, x) => `\n#### ${x.trim()}\n`)
    .replace(/\\subsection\*?\s*\{([\s\S]*?)\}/g, (_, x) => `\n### ${x.trim()}\n`)
    .replace(/\\section\*?\s*\{([\s\S]*?)\}/g, (_, x) => `\n## ${x.trim()}\n`)
    .replace(/\\(?:author|date|maketitle)\s*(\{[\s\S]*?\})?/g, "")
    .replace(/\\footnotetext\s*\{([\s\S]*?)\}/g, (_, x) => `\n> ${x.trim()}\n`)
    .replace(/\\(?:begin|end)\{(?:document|center|abstract)\}/g, "");

  // unescape the characters latex requires escaping (outside math, which is already fenced
  // into spans/divs — KaTeX handles its own escapes)
  t = t.replace(/\\([&%_#$])/g, "$1");
  return t;
}

export const prepareMarkdown = (src) =>
  normalizeMathDelims(latexToMarkdown(repairFences(src)));
