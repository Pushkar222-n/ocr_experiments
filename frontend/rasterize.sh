#!/usr/bin/env bash
# Pre-rasterize the sample PDFs to PNGs with poppler (pdftoppm) and write a manifest.
# We serve these images instead of rendering PDFs client-side: pdf.js silently fails
# to decode the CCITT Group-4 fax scans these documents use (render "succeeds" but the
# page is all white). Poppler decodes them correctly. Output is gitignored — rerun this
# after changing the sample set. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

SRC="../data/Evaluation set/sample_set"
OUT="public/pdf-pages"
DPI="${DPI:-120}"

rm -rf "$OUT"
mkdir -p "$OUT"

manifest="{"
first=1
for pdf in "$SRC"/*.pdf; do
  stem="$(basename "$pdf" .pdf)"
  mkdir -p "$OUT/$stem"
  pdftoppm -png -r "$DPI" "$pdf" "$OUT/$stem/page"
  files=$(cd "$OUT/$stem" && ls page-*.png | sort -V | sed "s|^|\"$stem/|;s|$|\"|" | paste -sd,)
  [ $first -eq 1 ] || manifest+=","
  manifest+="\"$stem\":[$files]"
  first=0
  echo "  $stem: $(ls "$OUT/$stem" | wc -l) pages"
done
manifest+="}"
echo "$manifest" > "$OUT/manifest.json"
echo "wrote $OUT/manifest.json"
