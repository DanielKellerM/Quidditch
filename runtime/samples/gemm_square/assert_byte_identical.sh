#!/usr/bin/env bash
# CI gate: assert two built device objects are byte-identical (normalized .text).
# Usage: assert_byte_identical.sh <object-a> <object-b>. Needs QUIDDITCH_OBJDUMP.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/gate_lib.sh"

if diff <(norm "$1") <(norm "$2") >/dev/null; then
  echo "PASS: $(basename "$1") byte-identical to $(basename "$2")"
else
  echo "FAIL: $(basename "$1") diverged from $(basename "$2")" >&2
  diff <(norm "$1") <(norm "$2") | head >&2
  exit 1
fi
