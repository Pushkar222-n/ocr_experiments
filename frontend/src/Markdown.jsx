import { memo, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css";
import { prepareMarkdown, DIAGRAM_START } from "./mdrepair";

function MermaidBlock({ code }) {
  const [svg, setSvg] = useState(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "loose",
          suppressErrorRendering: true,
          theme: matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "default",
        });
        const { svg } = await mermaid.render(
          "mmd" + Math.random().toString(36).slice(2),
          code
        );
        if (!dead) setSvg(svg);
      } catch {
        if (!dead) setFailed(true); // not valid mermaid after all — show as code
      }
    })();
    return () => {
      dead = true;
    };
  }, [code]);
  if (failed)
    return (
      <pre>
        <code>{code}</code>
      </pre>
    );
  if (!svg) return <pre className="muted">rendering diagram…</pre>;
  return <div className="mermaid-diagram" dangerouslySetInnerHTML={{ __html: svg }} />;
}

// intercept <pre> so fenced blocks that are mermaid (declared or sniffed — chandra
// emits its graphs in bare fences) render as diagrams instead of code
function Pre({ children, node, ...props }) {
  const child = Array.isArray(children) ? children[0] : children;
  const cls = child?.props?.className ?? "";
  const text = String(child?.props?.children ?? "").replace(/\n$/, "");
  if (/language-mermaid/.test(cls) || DIAGRAM_START.test(text)) {
    return <MermaidBlock code={text} />;
  }
  return <pre {...props}>{children}</pre>;
}

const COMPONENTS = { pre: Pre };
const REMARK = [remarkGfm, remarkMath];
const REHYPE = [rehypeRaw, rehypeKatex];

export default memo(function Markdown({ text }) {
  const prepared = useMemo(() => prepareMarkdown(text), [text]);
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={REMARK} rehypePlugins={REHYPE} components={COMPONENTS}>
        {prepared}
      </ReactMarkdown>
    </div>
  );
});
