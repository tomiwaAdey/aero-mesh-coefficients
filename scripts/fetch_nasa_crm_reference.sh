#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
target="$root/vendor/nasa-crm-high-speed"
archive="$target/DPW4_wb_no_tail_v03.igs.gz"
geometry="$target/DPW4_wb_no_tail_v03.igs"
url="https://web.archive.org/web/20090320225007id_/http://aaac.larc.nasa.gov/tsab/cfdlarc/aiaa-dpw/Workshop4/Geometry/2008-11-07/DPW4_wb_no_tail_v03.igs.gz"
expected_sha256="5033db500c09a468b1dfb8c5e5142ea2a04e2500af459ae84c004b458b75c037"
paper="$target/AIAA-2008-6919-Vassberg.pdf"
paper_url="https://www.aiaa-dpw.org/Workshop4/AIAA-2008-6919-Vassberg.pdf"
paper_sha256="82ef59b7d612d0a8a104e472b2f5db3f1023c6279fe4775a1b0a1185100e8732"

mkdir -p "$target"
curl --fail --location --retry 3 --silent --show-error --output "$archive" "$url"
printf '%s  %s\n' "$expected_sha256" "$archive" | shasum -a 256 -c -
gzip -dc "$archive" > "$geometry"
curl --fail --location --retry 3 --silent --show-error --output "$paper" "$paper_url"
printf '%s  %s\n' "$paper_sha256" "$paper" | shasum -a 256 -c -
printf 'Downloaded the November 2008 DPW4 high-speed CRM geometry to %s\n' "$geometry"
