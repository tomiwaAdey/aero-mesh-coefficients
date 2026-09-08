#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="aero-mesh-gmsh:2.7.0"
archive="$root/vendor/gmsh-2.7.0-Linux64.tgz"
url="https://gmsh.info/bin/Linux/gmsh-2.7.0-Linux64.tgz"
sha256="f2303b791cf2e05efa6850de1eac4c70a7586dd1b19db49df133d2b7291e9a4d"

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required to run Gmsh 2.7.0 reproducibly." >&2
  exit 1
}

mkdir -p "$(dirname "$archive")"
curl --fail --location --silent --show-error "$url" --output "$archive"
printf '%s  %s\n' "$sha256" "$archive" | shasum -a 256 -c -

docker build --platform linux/amd64 -t "$image" "$root/containers/gmsh-2.7.0"
version="$(docker run --rm --platform linux/amd64 "$image" -version 2>&1 | grep -E '^2\.7\.0$' | tail -1)"
printf '%s\n' "$version"
test "$version" = "2.7.0" || {
  echo "Gmsh image did not report version 2.7.0." >&2
  exit 1
}
