#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$root/vendor/onera-m6"

mkdir -p "$destination"

download() {
  local url="$1"
  local output="$2"
  local expected="$3"
  curl --fail --location --silent --show-error "$url" --output "$output"
  printf '%s  %s\n' "$expected" "$output" | shasum -a 256 -c -
}

download \
  "https://www.grc.nasa.gov/www/wind/valid/m6wing/airfoil.txt" \
  "$destination/airfoil.txt" \
  "66b2a7bcd5a0cab274396de1c5c55d0f5cad90911a1e19ddae4e77db16834ab7"
download \
  "https://www.grc.nasa.gov/www/wind/valid/m6wing/page01.pdf" \
  "$destination/schmitt-charpin-page01.pdf" \
  "34fa0be78d766c1e048ced09c4e1aa3ddc8605b99b4155d394449e55f0f10a27"
download \
  "https://www.grc.nasa.gov/www/wind/valid/m6wing/page38.pdf" \
  "$destination/schmitt-charpin-page38.pdf" \
  "8836b7b90175032b5e5988daaf0bc14b29633d19c8d1736082cd2ed7d6575860"

python="${PYTHON:-$root/.venv/bin/python}"
if [[ ! -x "$python" ]]; then
  python=python3
fi
PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python" "$root/scripts/generate_onera_m6_reference.py"
