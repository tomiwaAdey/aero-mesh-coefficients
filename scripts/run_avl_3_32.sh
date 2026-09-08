#!/usr/bin/env bash
set -euo pipefail

geometry="${1:?Pass an AVL geometry file.}"
case_dir="$(cd "$(dirname "$geometry")" && pwd)"
case_name="$(basename "$geometry")"

exec docker run --rm -i --platform linux/amd64 \
  --volume "$case_dir:/work" \
  --workdir /work \
  aero-mesh-avl:3.32 "$case_name"
