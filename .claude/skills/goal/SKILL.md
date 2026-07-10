---
name: goal
description: >-
  Track and align work against the project's north-star goal. `/goal` shows the
  goal hierarchy + status from GOAL.md; `/goal align <task|plan|diff>` judges
  whether a thing advances a sub-goal, drifts, is scope-creep, or is not-to-
  standard; `/goal update` records progress. Use when checking what the goal is,
  whether current or proposed work advances it, whether something is a detour, or
  to mark a milestone. Also: "are we on track", "does this fit the goal", "should
  we do this now", "update the goal", "what's the north star".
---

# /goal — track + align against the north star

The project's goal lives in **`GOAL.md` at the repo root** — a north star, sub-goals
with status markers, and an explicitly-deferred list. This skill keeps work
anchored to it. `GOAL.md` is the source of truth; if it's missing, offer to seed
it (don't invent a goal silently).

## Modes

**`/goal` (no args) — show.**
Read `GOAL.md` and print a scannable briefing: the north star, the sub-goals with
their markers (`[x]` done · `[~]` in progress · `[ ]` planned), and the deferred
list. If a sub-goal's status looks stale versus what's actually happened recently,
flag it and offer to `update`. A briefing, not a re-derivation.

**`/goal align <task | plan | diff | idea>` — judge.**
Decide, against `GOAL.md`, whether the thing is worth doing *now* and *this way*.
Give a verdict, not a vibe:
1. **Which sub-goal (if any)** does it serve?
2. **Verdict** — one of:
   - **ADVANCE** — directly moves a sub-goal forward.
   - **DRIFT** — serves nothing on the list; it's a detour. Name what it displaces.
   - **SCOPE-CREEP** — expands beyond the goal or into a deferred item; push back.
   - **NOT-TO-STANDARD** — may touch a sub-goal, but as a hack: it would fail
     `principal-review` or isn't upstreamable (see the `principal-review` skill's
     `references/upstreamability.md`). Advancing the goal the wrong way is not
     advancing it.
3. **One line why**, plus a **redirect** if it's off — the smaller or cleaner thing
   that *would* advance the goal.

Be willing to say "not now." That gate is the whole point — if every answer is
ADVANCE, you're not looking hard enough at drift and scope-creep.

**`/goal update <what happened>` — record.**
Edit `GOAL.md`: flip a status marker, add a sub-goal, or re-status one. Keep the
**north star stable** — change it only on a deliberate pivot, and say so. Terse:
the file is a living index, not a changelog.

## Notes

- "Advances the goal" always includes "to the standard." A change a principal
  review would send back does not count as progress, however much code it moves.
- Cross-check with the actual repo state (branches, recent commits, open review
  findings) before claiming a sub-goal is done — don't trust a stale marker.
- Keep `GOAL.md` short. Sub-goals are outcomes, not task lists; the detailed steps
  live in plan files and the task list, not here.
