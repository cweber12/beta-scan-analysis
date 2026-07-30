# Scanner handoffs

The Beta Scanner is a **separate repo** with its own agents. This harness cannot change
it, and no agent here should try. Work that lands on the scanner side travels as a
**handoff doc** in [`docs/handoffs/`](../handoffs/), tracked by a GitHub issue labelled
`ready-for-human`.

The docs in `docs/handoffs/` are the prose half of the cross-repo contract; the
machine-readable half is `GET /api/contract` (see ADR / issue #63). Neither replaces
the other.

## When to write one

Whenever the fix — or half of it — has to happen in the scanner: a new or changed field
in `setup.json` / `ground-truth.json` / pose `diagnostics`, a change in what an endpoint
sends, a detection-behaviour ask, or a sequencing decision the scanner has to make.

An issue alone is not a handoff. Issues state the problem and the acceptance criteria
for *this* repo's tracker; the handoff doc is the spec the scanner agent implements
from. Findings-only messages ("here is what the corpus says about your detector") are
handoffs too — the `scanner-detection-improvements*` series is that shape.

## The flow

1. **Write `docs/handoffs/scanner-<topic>.md`** — see the shape below.
2. **Commit it `docs:` on a branch and open the PR.** One handoff per branch; it is a
   concern of its own. Never mix it with pipeline code or `analysis/` data.
3. **Comment the path on the tracking issue**, on the branch it lives on until it
   merges — e.g. *"Scanner handoff written: `docs/handoffs/scanner-foo.md` (on
   `docs/130-…`)"*. The issue stays the tracker; the doc is the spec. **Do not inline
   the spec into the issue body** — it will drift from the file.
4. **Label the issue `ready-for-human`** if it is not already. The human carries it
   across to the scanner repo; agents here never push to that repo.
5. **If the change is a contract decision rather than a request, write the ADR too** and
   link it from the handoff. ADRs 0005–0008 all have a paired handoff.

## Shape of a good handoff

The existing docs converge on this, and the ones that get implemented cleanly all have
it. `scanner-truth-scaffold-provenance.md` is the reference example.

- **`**Audience:**` line first** — name the repo, and say whether they need this repo
  open. Usually they don't, and saying so removes a blocker.
- **Companion docs + harness refs** — link the handoffs this one extends, the issue, and
  the PRD if it is a slice. Say if it gates other work.
- **The defect, concretely, with measured numbers.** Not "runs may be misattributed" —
  *67 runs stamped `c305954`*. Per CLAUDE.md, prove it against the real corpus. A handoff
  asserting a problem it hasn't measured is how inert asks get shipped.
- **What to change** — the exact JSON shape, the exact field, where it goes. State
  which parts are *suggestions* (field names usually are) versus *requirements* (the
  properties the field must satisfy). The scanner agent knows its codebase; over-
  specifying its internals wastes both sides' time.
- **Fail-open on absence, explicitly.** Records written before the change must degrade
  to today's behaviour, never to an error or a false positive. This is the established
  tradition on every field in this contract; state the null guards.
- **What the harness does with it** — what already ships, what is waiting. Marks clearly
  which half is *not* the scanner's work.
- **An acceptance procedure, not just acceptance criteria.** The concrete steps that
  demonstrate the change works against reality, and — where they exist — the steps that
  distinguish a real fix from one that passes tests while being useless. Same rule as
  CLAUDE.md's "green tests do not demonstrate that a change did anything", applied across
  the repo boundary. This is the section most often missing, and the one that decides
  whether what comes back is actually the thing you asked for.
- **What to report back**, and why the harness needs it. Usually the final field name and
  semantics, so the reader can be written against what actually shipped rather than what
  was proposed. Say which parts of the spec are open to pushback.
- **Not in scope** — the adjacent asks you deliberately excluded, and why. Prevents the
  handoff growing scope in the scanner's hands.

**Write it to be self-sufficient.** The repo is public, so a complete handoff can be
handed to a scanner-side agent as a one-line prompt — *"fetch \<raw URL\> and implement it
in this repo"* — with nothing re-pasted and nothing to drift. If a prompt has to carry
context the doc omits, that context belongs in the doc.

## Replies are handoffs too

When the scanner asks a question, answer in `docs/handoffs/scanner-<topic>-reply.md`
rather than only in a comment thread — the answer usually corrects a premise and needs
to be citable later. `scanner-reset-sequencing-reply.md` held a batch from being spent
on a wrong premise; that only worked because it was a file.

If a handoff's advice is later superseded, **edit the doc** and note it — a stale
handoff on disk is worse than none, because it reads as current.
