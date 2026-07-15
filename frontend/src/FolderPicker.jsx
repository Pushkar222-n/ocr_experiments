import { useEffect, useState } from "react";

// A native-feeling folder chooser that navigates the REAL filesystem through the dev server's
// /api/dirs endpoint. A browser can't return an absolute path from the OS file dialog (for
// security), and the OCR API needs real paths — so we browse server-side and hand back the
// path the user lands on.
export default function FolderPicker({ start, onPick, onClose }) {
  const [cwd, setCwd] = useState(start || null);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const go = (path) => {
    setError(null);
    fetch(`/api/dirs${path ? `?path=${encodeURIComponent(path)}` : ""}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.error) throw new Error(d.error);
        setData(d);
        setCwd(d.path);
      })
      .catch((e) => setError(e.message));
  };

  useEffect(() => { go(start || null); /* eslint-disable-next-line */ }, []);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <strong>Choose a folder</strong>
          <button className="mini" onClick={onClose}>×</button>
        </div>

        <div className="fp-path">
          <button className="mini" disabled={!data?.parent} onClick={() => go(data.parent)} title="up one level">↑</button>
          <input
            value={cwd || ""}
            onChange={(e) => setCwd(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && go(cwd)}
            spellCheck={false}
          />
          <button className="mini" onClick={() => go(cwd)} title="go">go</button>
          {data?.home && <button className="mini" onClick={() => go(data.home)} title="home">~</button>}
        </div>

        {error && <p className="error fp-error">{error}</p>}

        {data && (
          <>
            <div className="fp-counts">
              {data.pdfs} pdf · {data.mds} md <span className="muted">(in this folder)</span>
            </div>
            <div className="fp-list">
              {data.dirs.length === 0 && <p className="muted">no sub-folders</p>}
              {data.dirs.map((name) => (
                <button key={name} className="fp-dir" onDoubleClick={() => go(`${data.path}/${name}`)}
                        onClick={() => go(`${data.path}/${name}`)} title={name}>
                  📁 {name}
                </button>
              ))}
            </div>
          </>
        )}

        <div className="modal-foot">
          <span className="muted fp-hint">click a folder to open it</span>
          <button className="load-btn" disabled={!data} onClick={() => onPick(data.path)}>
            Select this folder
          </button>
        </div>
      </div>
    </div>
  );
}
