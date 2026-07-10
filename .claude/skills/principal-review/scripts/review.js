// Principal-review workflow: fan out one reviewer per convention dimension over
// the scope, adversarially verify every finding (real violation vs style
// opinion), dedup, and synthesize one report.
//
// Invoke via the Workflow tool:
//   Workflow({ scriptPath: ".claude/skills/principal-review/scripts/review.js",
//              args: { scope: "<label>", repo: "<abs repo root>", files: ["a.c", ...] } })
// The caller (SKILL.md) determines `files` first (changed files for a diff, or a
// glob for a subsystem) and passes them in. A file may match several dimensions;
// each reviews it through its own lens on purpose.

export const meta = {
  name: 'principal-review',
  description: 'Multi-agent principal-grade code + docs review with adversarial verification',
  phases: [
    { title: 'Review', detail: 'one reviewer per convention dimension present' },
    { title: 'Verify', detail: 'adversarially confirm each finding is real, not opinion' },
    { title: 'Synthesize', detail: 'dedup + one report with a merge/send-back verdict' },
  ],
}

// args may arrive as an object or (defensively) as a JSON string.
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = {} } }
const repo = (A && A.repo) || '/home/dankeller/Projects/Quidditch'
const scope = (A && A.scope) || 'uncommitted diff'
const files = (A && A.files) || []
const commentsOnly = !!(A && A.only === 'comments')
log(`review: args=${typeof args}, resolved ${files.length} files${commentsOnly ? ' (comments-only)' : ''}`)
const SKILL = `${repo}/.claude/skills/principal-review`

if (!files.length) {
  log('No files passed in args.files — nothing to review. The caller must enumerate the scope first.')
  return { confirmed: [], report: 'No files provided to review.' }
}

// Dimension routing. `match` decides which files each reviewer sees; a file may
// land in several dimensions (e.g. an ABI header gets both IREE-C and ABI lenses).
const DIMS = [
  { key: 'iree-c', ref: 'references/iree-runtime-c.md',
    neighbor: 'iree/runtime/src/iree/hal/drivers/null/ and iree/runtime/src/iree/hal/buffer.h',
    match: f => /runtime\/(host|samples|runtime)\/.*\.(c|h)$/.test(f) },
  { key: 'mlir-cpp', ref: 'references/mlir-compiler-cpp.md',
    neighbor: 'codegen/compiler/src/Quidditch/Conversion/*.cpp and Target/Passes.td',
    match: f => /codegen\/.*\.(cpp|cc|h|td)$/.test(f) },
  { key: 'xdsl', ref: 'references/xdsl-passes.md',
    neighbor: 'xdsl/xdsl/transforms/convert_riscv_scf_for_to_frep.py',
    match: f => /\.py$/.test(f) && /(xdsl|transform|pass)/.test(f) },
  { key: 'abi', ref: 'references/abi-and-versioning.md',
    neighbor: 'runtime/host/transport/cluster_command_stream.h and shared_region.h',
    match: f => /(qcs|abi|command_stream|shared_region|executable|descriptor)/i.test(f) && /\.(c|h)$/.test(f) },
  { key: 'docs', ref: 'references/documentation.md',
    neighbor: 'docs/host-device-split.md and runtime/host/transport/cluster_command_stream.h banner',
    match: f => /\.md$/.test(f) || /README/i.test(f) },
  { key: 'build', ref: 'references/conventions.md',
    neighbor: 'the CMake cache-var pattern in runtime/snitch_cluster/CMakeLists.txt',
    match: f => /\.(sh|cmake)$/.test(f) || /CMakeLists\.txt$/.test(f) || /tools\/.*\.py$/.test(f)
             || /(pyproject\.toml|requirements[\w-]*\.txt)$/.test(f) },
]

// Chunk each dimension's files into batches so a reviewer reads a digestible set
// (a single agent can't deeply review 50 files). One reviewer per batch.
const BATCH = 7
const chunk = (a, n) => { const o = []; for (let i = 0; i < a.length; i += n) o.push(a.slice(i, i + n)); return o }
const tasks = []
if (!commentsOnly) for (const d of DIMS) {
  const df = files.filter(d.match)
  if (!df.length) continue
  const batches = chunk(df, BATCH)
  batches.forEach((b, i) => tasks.push({ ...d, kind: 'ecosystem', effort: 'medium', files: b,
    label: batches.length > 1 ? `${d.key}#${i + 1}/${batches.length}` : d.key }))
}

// Comment-discipline: a mandatory LIGHT pass over ALL implementation code (not
// docs). Cheap agent, larger batches — it enforces one mechanical rule: no
// multi-line / added / debugging / restating comments in code bodies.
const isCode = f => /\.(c|h|cpp|cc|td|py|sh|cmake)$/.test(f) || /CMakeLists\.txt$/.test(f)
const codeFiles = files.filter(isCode)
chunk(codeFiles, 10).forEach((b, i, all) => tasks.push({
  key: 'comments', kind: 'comment', effort: 'low', files: b,
  label: all.length > 1 ? `comments#${i + 1}/${all.length}` : 'comments' }))

// Upstreamability: every repo here is a fork. Classify each changed file as
// upstreamable / fork-only-justified / unjustified-divergence. Needs judgment.
if (!commentsOnly) chunk(codeFiles, 8).forEach((b, i, all) => tasks.push({
  key: 'upstream', kind: 'upstream', effort: 'medium', files: b,
  label: all.length > 1 ? `upstream#${i + 1}/${all.length}` : 'upstream' }))

if (!tasks.length) {
  log(`None of the ${files.length} files matched a review dimension.`)
  return { confirmed: [], report: 'No files matched a review dimension.' }
}
const dimsPresent = [...DIMS.filter(d => files.some(d.match)).map(d => d.key),
                     codeFiles.length ? 'comments' : '',
                     (codeFiles.length && !commentsOnly) ? 'upstream' : ''].filter(Boolean).join(', ')
log(`Reviewing ${files.length} files across ${tasks.length} batches (dimensions: ${dimsPresent}).`)

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: {
        file: { type: 'string' }, line: { type: 'string' },
        severity: { type: 'string', enum: ['BLOCKER', 'TASTE', 'NIT'] },
        rule: { type: 'string', description: 'the convention/idiom violated, one line' },
        idiomaticFix: { type: 'string' },
        neighborCited: { type: 'string', description: 'the upstream/neighbor file:line it should match' },
      }, required: ['file', 'line', 'severity', 'rule', 'idiomaticFix', 'neighborCited'] } },
    notes: { type: 'string' },
  }, required: ['findings', 'notes'],
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    real: { type: 'boolean', description: 'true only if a genuine idiom violation or real risk, not a style opinion' },
    severity: { type: 'string', enum: ['BLOCKER', 'TASTE', 'NIT', 'FALSE-POSITIVE'] },
    reason: { type: 'string' },
  }, required: ['real', 'severity', 'reason'],
}

const reviewPrompt = p => `You are an extremely opinionated PRINCIPAL compiler engineer reviewing for TASTE, not bugs. Repo root: ${repo}.
1. Read the convention reference: ${SKILL}/${p.ref}
2. Read the gold-standard neighbor(s) to calibrate the bar: ${p.neighbor}
3. Review ONLY these files against that bar (the "${p.key}" dimension):
${p.files.map(f => '   - ' + f).join('\n')}
Report findings where the code fails what a principal would accept: foreign idiom, hardcoded/absolute paths, magic constants/addresses, vendored-ABI drift, missing error propagation, non-idiomatic naming, rotting or constant-restating docs. For each: file, line, severity (BLOCKER=correctness/ABI/portability/fails-review, TASTE=works-but-send-back, NIT=optional), the rule violated, the idiomatic fix, and the neighbor file:line it should match. Be concrete and cite the neighbor. DO NOT bikeshed — every finding ties to a real idiom or risk. Read only.`

const verifyPrompt = f => `You are a skeptical principal engineer auditing another reviewer's finding for false positives. Repo root: ${repo}.
Finding: [${f.severity}] ${f.file}:${f.line} — ${f.rule}
Proposed fix: ${f.idiomaticFix}
Claimed neighbor to match: ${f.neighborCited}
Open ${f.file} at that location and the cited neighbor. Decide: is this a REAL convention violation or genuine risk, or is it a style opinion / false positive / already-idiomatic? Default to real=false if it is merely taste-preference with no idiom or risk behind it. Return real, the (possibly adjusted) severity, and a one-line reason. Read only.`

const commentPrompt = p => `You are a LIGHT, fast reviewer enforcing ONE rule: COMMENT DISCIPLINE in implementation code. Repo root: ${repo}. Project rule: ${SKILL}/references/conventions.md (§0 comments). Scan ONLY these files:
${p.files.map(f => '   - ' + f).join('\n')}
Flag as findings (file:line):
- ANY multi-line (>1 line) comment BLOCK inside implementation code (function/logic bodies) — the "explain the change / debugging journey" narrative. Fix: condense to <=1 terse line, or delete; the rationale belongs in the commit message / PR, not the code.
- Commented-out code, leftover debug prints / logging, or debugging narrative left in comments.
- A comment that restates WHAT the code does, or was clearly added just to explain a change (a comment where none is warranted).
- A comment whose style/tone does not match the surrounding code.
DO NOT flag the file-header license/contract banner or a single-line public-API doc comment — those are the docs dimension. Focus on inline body comments. Severity: a multi-line block or debug-in-code = TASTE; an egregious multi-paragraph dump or committed debug spew = BLOCKER. For each: file, line, the rule violated, and the fix ("condense to 1 line" / "delete, move to commit"). Read only.`

const upstreamPrompt = p => `You are a principal engineer reviewing FORK CHANGES for UPSTREAMABILITY. Repo root: ${repo}. Read the framework first: ${SKILL}/references/upstreamability.md (the fork map + the three verdicts). Every repo here is a fork (Quidditch<-opencompl, iree<-iree-org, xdsl<-xdslproject, snitch_cluster<-pulp-platform); a future Nimbus repo owns deployment code. Judge each of these CHANGED files against its upstream:
${p.files.map(f => '   - ' + f).join('\n')}
Emit a finding for: (a) an UPSTREAMABLE fix sitting fork-only with no tracked PR (TASTE); (b) UNJUSTIFIED fork divergence — a hack that could/should be general (BLOCKER); (c) a fork-only change with no documented rationale (TASTE, or BLOCKER if high-divergence like editing an IREE ABI header); (d) an INVASIVE rewrite of an upstream code path where an additive / cfg-gated change would rebase cleanly (BLOCKER). Each finding: file, line, severity, rule (state the verdict — upstreamable / fork-only-justified / unjustified), the fix (generalize + PR / document the rationale / make it additive), and cite the upstream. Deployment-only code legitimately bound for Nimbus is NOT a finding — say so, don't flag it. Read only.`

phase('Review')
const reviewed = await pipeline(
  tasks,
  p => agent(p.kind === 'comment' ? commentPrompt(p) : p.kind === 'upstream' ? upstreamPrompt(p) : reviewPrompt(p),
    { label: `review:${p.label}`, phase: 'Review', schema: FINDINGS_SCHEMA, agentType: 'Explore', effort: p.effort }),
  (r, p) => parallel(((r && r.findings) || []).map(f => () =>
    agent(verifyPrompt(f), { label: `verify:${p.label}`, phase: 'Verify', schema: VERDICT_SCHEMA, agentType: 'Explore' })
      .then(v => ({ ...f, dimension: p.key, verdict: v })))),
)

const confirmed = reviewed
  .flat().filter(Boolean)
  .filter(f => f.verdict && f.verdict.real)
  .map(f => ({ ...f, severity: (f.verdict.severity && f.verdict.severity !== 'FALSE-POSITIVE') ? f.verdict.severity : f.severity }))

// dedup identical file:line:rule surfaced by multiple dimensions
const seen = new Set()
const deduped = confirmed.filter(f => {
  const k = `${f.file}:${f.line}:${f.rule}`
  if (seen.has(k)) return false
  seen.add(k); return true
})
log(`${deduped.length} confirmed findings after verification + dedup (from ${reviewed.flat().filter(Boolean).length} raw).`)

phase('Synthesize')
const bySev = s => deduped.filter(f => f.severity === s)
const fmt = f => `- ${f.file}:${f.line} — ${f.rule} → ${f.idiomaticFix} (match: ${f.neighborCited})`
const report = await agent(
  `You are the principal engineer writing the final review verdict for scope "${scope}". Below are the VERIFIED findings (already confirmed real + deduped). Write the report in exactly this format:

## Principal review — ${scope}

Verdict: <MERGE AS-IS | SEND BACK> — <one line>

### Blockers
${bySev('BLOCKER').map(fmt).join('\n') || '- (none)'}

### Taste
${bySev('TASTE').map(fmt).join('\n') || '- (none)'}

### Nits (optional)
${bySev('NIT').map(fmt).join('\n') || '- (none)'}

### Top 3 to fix first
<pick the 3 highest-leverage fixes across the above; if fewer, list what there is>

Keep it tight. The verdict is SEND BACK if there is any blocker, or if the taste findings collectively mean a principal wouldn't merge. Do not invent findings beyond those given.`,
  { label: 'synthesize', phase: 'Synthesize' })

return { confirmed: deduped, report }
