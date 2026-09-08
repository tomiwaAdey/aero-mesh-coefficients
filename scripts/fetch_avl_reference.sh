#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="aero-mesh-avl:3.32"
archive="$root/vendor/avl3.32.tgz"
url="https://web.mit.edu/drela/Public/web/avl/avl3.32.tgz"
sha256="dcc0a6ae7ee2f5b3d6a46e730eecab1a1e2a4472a0cd3314bc7fd464f7e091f0"

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required to build AVL 3.32 reproducibly." >&2
  exit 1
}

mkdir -p "$(dirname "$archive")"
curl --fail --location --silent --show-error "$url" --output "$archive"
printf '%s  %s\n' "$sha256" "$archive" | shasum -a 256 -c -

docker build --platform linux/amd64 -t "$image" "$root/containers/avl-3.32"
banner="$(printf 'QUIT\n' | docker run --rm -i --platform linux/amd64 "$image" 2>&1)"
printf '%s\n' "$banner"
grep -Eq 'Athena Vortex Lattice[[:space:]]+Program[[:space:]]+Version[[:space:]]+3\.32([[:space:]]|$)' <<<"$banner" || {
  echo "AVL image did not report version 3.32." >&2
  exit 1
}
