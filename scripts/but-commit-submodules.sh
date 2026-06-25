#!/usr/bin/env bash
#
# Commit git submodule gitlink bumps under a GitButler workspace.
#
# GitButler's commit engine cannot author a submodule gitlink: the pointer is a
# Mode::COMMIT tree entry with no worktree hunks for its hunk-replay model, so
# `but commit --changes <submodule>` rejects it ("Some selected changes could
# not be committed") and emits an empty commit. The supported fix is to briefly
# leave the workspace, record the bump with plain git on the applied branch, and
# re-enter. This script does exactly that, safely, in one command.
#
# Usage:  scripts/but-commit-submodules.sh <applied-branch> [message] [submodule...]
#   <applied-branch>  GitButler branch to record the bump on (e.g. autotuner-design)
#   [message]         commit message (default: "build: Bump submodule gitlinks")
#   [submodule...]    paths to commit (default: every submodule whose checkout
#                     differs from the recorded pointer, per `git submodule status`)
#
set -euo pipefail

BRANCH="${1:-}"
if [[ -z "$BRANCH" ]]; then
  echo "usage: $0 <applied-branch> [message] [submodule...]" >&2
  exit 2
fi
shift
MSG="${1:-build: Bump submodule gitlinks}"
[[ $# -gt 0 ]] && shift || true
SUBS=("$@")

# Must run from inside the GitButler workspace.
cur="$(git symbolic-ref --short -q HEAD || true)"
if [[ "$cur" != "gitbutler/workspace" ]]; then
  echo "error: HEAD is '$cur', expected gitbutler/workspace -- run from the workspace." >&2
  exit 1
fi

# Default to every submodule whose checkout is ahead of the recorded gitlink ('+').
if [[ ${#SUBS[@]} -eq 0 ]]; then
  mapfile -t SUBS < <(git submodule status | awk '/^\+/{print $2}')
fi
if [[ ${#SUBS[@]} -eq 0 ]]; then
  echo "no submodule gitlink changes to commit."
  exit 0
fi
echo ">> recording gitlink bumps for: ${SUBS[*]}"

# Re-enter the workspace no matter how we exit (success, failure, or interrupt).
restore() {
  echo ">> re-entering GitButler workspace (but setup)"
  but setup >/dev/null 2>&1 || \
    echo "!! 'but setup' failed -- run it manually to restore the workspace." >&2
}
trap restore EXIT

git checkout "$BRANCH"
# Partial commit of ONLY the submodule paths; any other staged/working changes
# are left untouched and carried back into the workspace.
git commit -m "$MSG" -- "${SUBS[@]}"
echo ">> committed on $BRANCH ($(git rev-parse --short HEAD)):"
for s in "${SUBS[@]}"; do
  echo "   $s -> $(git rev-parse --short "HEAD:$s" 2>/dev/null || echo '?')"
done
