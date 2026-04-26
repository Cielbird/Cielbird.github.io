#!/usr/bin/env bash
set -e

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
OUT="$SITE_ROOT/static/demos/hello-wasm"

cd "$DEMO_DIR"
wasm-pack build --target web --out-dir pkg

mkdir -p "$OUT"
cp -r pkg "$OUT/"
cp index.html "$OUT/"

echo "Deployed to $OUT"
