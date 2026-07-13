import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import fs from 'node:fs'

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

// public/outputs and public/pdfs are symlinks to ../outputs and ../data — both
// live one level above this project root, so fs.allow must include the repo root.
export default defineConfig({
  plugins: [react(), mdSave],
  server: {
    fs: { allow: [path.resolve(__dirname, '..')] },
  },
})
