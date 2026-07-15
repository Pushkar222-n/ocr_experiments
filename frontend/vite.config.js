import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import fs from 'node:fs'
import os from 'node:os'
import crypto from 'node:crypto'
import { execFileSync } from 'node:child_process'

const OUT_ROOT = path.resolve(__dirname, '../outputs')

// Dev-server-only save endpoint: PUT /outputs/**.md overwrites the file on disk.
// Guards: resolved path must stay inside outputs/, must be .md, and must already
// exist (no creating new files). First overwrite snapshots the original to
// <file>.md.orig so a benchmark artifact is never destroyed irreversibly.
const mdSave = {
  name: 'md-save',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.method !== 'PUT' || !req.url.startsWith('/outputs/')) return next()
      const rel = decodeURIComponent(req.url.split('?')[0])
      const target = path.resolve(__dirname, '..', '.' + rel)
      if (!target.startsWith(OUT_ROOT + path.sep) || !target.endsWith('.md') || !fs.existsSync(target)) {
        res.statusCode = 403
        return res.end('forbidden: only existing .md files under outputs/ are writable')
      }
      let body = ''
      req.on('data', (c) => (body += c))
      req.on('end', () => {
        try {
          const orig = target + '.orig'
          if (!fs.existsSync(orig)) fs.copyFileSync(target, orig)
          fs.writeFileSync(target, body)
          res.end('ok')
        } catch (e) {
          res.statusCode = 500
          res.end(String(e))
        }
      })
    })
  },
}

// ---------------------------------------------------------------------------
// Advanced markdown viewer API (dev server only).
//
// The viewer lets the user point at an arbitrary INPUT folder of pdfs and an OUTPUT folder
// of the .md files an OCR run produced. A static frontend can't read arbitrary local paths,
// so these endpoints do it server-side. PDFs are rasterized on the fly with poppler
// (pdftoppm) — the same reason the experiment view uses images: pdf.js silently fails on the
// CCITT fax scans these documents use. Rasterized pages are cached under the OS temp dir keyed
// by (path, mtime), so re-opening a document is instant.
//
// This runs only under `vite dev` on the user's own machine, reading paths the user typed —
// it is not a hardening boundary, just a convenience. It never writes outside its temp cache.
const CACHE = path.join(os.tmpdir(), 'ocr-viewer-cache')

const send = (res, code, obj) => {
  res.statusCode = code
  res.setHeader('content-type', 'application/json')
  res.end(JSON.stringify(obj))
}

const MD_EXT = '.md'
const PDF_EXT = '.pdf'

function walk(dir, exts, base = dir, out = []) {
  let entries
  try { entries = fs.readdirSync(dir, { withFileTypes: true }) } catch { return out }
  for (const e of entries) {
    if (e.name.startsWith('.')) continue
    const full = path.join(dir, e.name)
    if (e.isDirectory()) walk(full, exts, base, out)
    else if (exts.includes(path.extname(e.name).toLowerCase()))
      out.push({ full, rel: path.relative(base, full) })
  }
  return out
}

// Match each input pdf to an output .md. Output may mirror the input tree, use the
// inference/ layout (<stem>/<stem>.md), or be flat. Strategy: index every .md by stem, then
// for a pdf prefer the md whose relative path shares the most leading path with the pdf's.
function pairDocs(inputDir, outputDir) {
  const pdfs = walk(inputDir, [PDF_EXT])
  const mds = walk(outputDir, [MD_EXT])
  const byStem = new Map()
  for (const m of mds) {
    const stem = path.basename(m.rel, MD_EXT)
    ;(byStem.get(stem) || byStem.set(stem, []).get(stem)).push(m)
  }
  const sharedDepth = (a, b) => {
    const pa = a.split(path.sep), pb = b.split(path.sep)
    let i = 0
    while (i < pa.length - 1 && i < pb.length - 1 && pa[i] === pb[i]) i++
    return i
  }
  return pdfs
    .map((p) => {
      const stem = path.basename(p.rel, PDF_EXT)
      const candidates = byStem.get(stem) || []
      candidates.sort((a, b) => sharedDepth(b.rel, p.rel) - sharedDepth(a.rel, p.rel))
      const md = candidates[0]
      return { rel: p.rel, stem, pdf: p.full, md: md ? md.full : null }
    })
    .sort((a, b) => a.rel.localeCompare(b.rel))
}

function rasterize(pdfPath) {
  const st = fs.statSync(pdfPath)
  const key = crypto.createHash('md5').update(`${pdfPath}:${st.mtimeMs}:${st.size}`).digest('hex')
  const dir = path.join(CACHE, key)
  const donefile = path.join(dir, 'done.json')
  if (fs.existsSync(donefile)) return JSON.parse(fs.readFileSync(donefile, 'utf8'))
  fs.mkdirSync(dir, { recursive: true })
  // 120 dpi is plenty on screen and keeps pages small; poppler handles CCITT correctly
  execFileSync('pdftoppm', ['-png', '-r', '120', pdfPath, path.join(dir, 'page')])
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.png')).sort()
  const meta = { pages: files.length, files, key }
  fs.writeFileSync(donefile, JSON.stringify(meta))
  return meta
}

const mdViewer = {
  name: 'md-viewer-api',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const u = new URL(req.url, 'http://localhost')
      if (!u.pathname.startsWith('/api/')) return next()
      try {
        if (u.pathname === '/api/dirs') {
          // list subdirectories for the in-app folder picker. Defaults to $HOME.
          const raw = u.searchParams.get('path')
          const dir = raw ? path.resolve(raw) : os.homedir()
          if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory())
            return send(res, 400, { error: `not a directory: ${dir}` })
          let entries
          try {
            entries = fs.readdirSync(dir, { withFileTypes: true })
          } catch (e) { return send(res, 400, { error: String(e.message || e) }) }
          const dirs = entries.filter((e) => e.isDirectory() && !e.name.startsWith('.'))
            .map((e) => e.name).sort((a, b) => a.localeCompare(b))
          // shallow counts only — a recursive walk of $HOME would be far too slow just to
          // render a folder list. This is a hint, not the authoritative pairing.
          const ext = (n) => path.extname(n).toLowerCase()
          const pdfs = entries.filter((e) => e.isFile() && ext(e.name) === PDF_EXT).length
          const mds = entries.filter((e) => e.isFile() && ext(e.name) === MD_EXT).length
          const parent = path.dirname(dir)
          return send(res, 200, { path: dir, parent: parent === dir ? null : parent, dirs, pdfs, mds, home: os.homedir() })
        }
        if (u.pathname === '/api/browse') {
          const input = u.searchParams.get('input') || ''
          const output = u.searchParams.get('output') || ''
          if (!fs.existsSync(input)) return send(res, 400, { error: `input not found: ${input}` })
          if (!fs.existsSync(output)) return send(res, 400, { error: `output not found: ${output}` })
          const docs = pairDocs(path.resolve(input), path.resolve(output))
          return send(res, 200, { docs, input: path.resolve(input), output: path.resolve(output) })
        }
        if (u.pathname === '/api/pdf-info') {
          const p = u.searchParams.get('path')
          if (!p || !fs.existsSync(p)) return send(res, 404, { error: 'pdf not found' })
          return send(res, 200, rasterize(p))
        }
        if (u.pathname === '/api/pdf-page') {
          const p = u.searchParams.get('path')
          const n = parseInt(u.searchParams.get('n') || '0', 10)
          if (!p || !fs.existsSync(p)) { res.statusCode = 404; return res.end('not found') }
          const meta = rasterize(p)
          const file = meta.files[n]
          if (!file) { res.statusCode = 404; return res.end('no such page') }
          res.setHeader('content-type', 'image/png')
          res.setHeader('cache-control', 'max-age=3600')
          return fs.createReadStream(path.join(CACHE, meta.key, file)).pipe(res)
        }
        if (u.pathname === '/api/md' || u.pathname === '/api/pdf-file') {
          const p = u.searchParams.get('path')
          if (!p || !fs.existsSync(p)) { res.statusCode = 404; return res.end('not found') }
          res.setHeader('content-type', u.pathname === '/api/md' ? 'text/plain; charset=utf-8' : 'application/pdf')
          return fs.createReadStream(p).pipe(res)
        }
        return send(res, 404, { error: 'unknown endpoint' })
      } catch (e) {
        return send(res, 500, { error: String(e && e.message || e) })
      }
    })
  },
}

// public/outputs and public/pdfs are symlinks to ../outputs and ../data — both
// live one level above this project root, so fs.allow must include the repo root.
export default defineConfig({
  plugins: [react(), mdSave, mdViewer],
  server: {
    fs: { allow: [path.resolve(__dirname, '..')] },
  },
})
