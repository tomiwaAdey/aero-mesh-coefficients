#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec docker run --rm --platform linux/amd64 \
  --volume "$root:$root" \
  --workdir "$PWD" \
  aero-mesh-gmsh:2.7.0 "$@"
