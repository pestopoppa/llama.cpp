# KPF-62 G3 corpus and task binding

The required B7-derived corpus is **not supplied by this tool**. Select actual task data
and an appropriate metric before freezing; generated fixtures are not B7 traffic. This
implementation currently supports deterministic `exact_token_match` targets only. A
different task metric needs its own implementation and prospective declaration.

For each target role, prepare two UTF-8 JSONL files with the same unique `id`s:

* Corpus: `{"id":"…","tokens":[...],"chunks":[{"offset":0,"n_tokens":64},...]}`.
  Token IDs must use that role's tokenizer; chunks are sorted, non-overlapping and
  leave a suffix. Each chunk's identical token sequence must appear at two or more
  distinct offsets or predecessor histories. Length and gate-bucket floors come from
  P0; split is `sha256(id) mod 5 == 0` for development, otherwise gate.
* Tasks: `{"id":"…","metric":"exact_token_match","target_tokens":[...]}`.
  The target is an independently sourced answer, not a model response or fixture.
  Record the real source, role, selection and license as part of corpus preparation.

Freeze with `python3 g3_manifest.py --corpus CORPUS --tasks TASKS --role ROLE
--task-source SOURCE --out MANIFEST`. Exclusive creation forbids replacing a prior
manifest. The tool checks raw byte hashes, structural constraints, every chunk's
placement diversity, and the held-out gate split's >=200 requests and >=50 in each
length bucket. Commit the immutable manifest and record its SHA in P0 section 8
before collection, following the existing measurement authority. Bind the tokenizer,
model, binary, argv, device and observation window separately in the run capsule;
this tool does not establish their provenance or decide which task source is B7.

The task outcome file passed as `g3_gate.py gate --retention OUTCOMES` is JSONL
with exactly one `{id, arm: "full"|"reuse", output_tokens: [...], complete: true}`
per held-out task. Retain the raw producer outputs and custody with that file. The
gate verifies the frozen corpus and task bytes, exact held-out IDs and token/chunk
shapes in all native arms, every suffix position plus 64 continuation positions,
and computes per-task exact-match scores and their ratio from the paired outputs.
It reports a deterministic 10,000-resample paired bootstrap CI; this CI does not
change the pre-existing threshold. Missing, duplicated, failed, unknown or
unmatched records refuse. A standalone aggregate `score_full`/`score_reuse` value
is not evidence and is not accepted. Other request capture and model/DSO provenance
remain the runner's responsibility; a score table alone is not a qualified run.

## Optional pinned project text scorer

For prospectively declared deterministic text tasks, the tasks JSONL may instead bind
`metric: "orchestrator_text_v1"`, a nonempty `expected` string, explicit
`scoring_method` (`substring`, `exact_match`, or `f1`), explicit `scoring_config`, and
`scorer_sha256` of the exact orchestrator `scripts/benchmark/debug_scorer.py` bytes. The
full tasks-file digest freezes all of these fields. Both outcome arms must carry original
`output_text` strings and `complete: true`; errors, missing pairs, unsupported methods,
multiple scorer digests, and an absent or changed scorer refuse. Pass the exact scorer
file via `g3_gate.py gate --scorer-module PATH`; the gate verifies and executes those
pinned bytes, returns per-item full/reuse scores, and computes the original retention
ratio and paired CI without introducing a new threshold. Original generated output,
token-to-text decoding and native run custody remain separately required.

This implements an instrument interface, not corpus or scoring-policy admission. The
project's current ZeroSCROLLS QA adapter selects substring; SQuALITY and open-ended
L-Eval use `llm_judge`. This deterministic path deliberately refuses `llm_judge` and
must not silently substitute a different score for those tasks. Freeze any task-specific
methods and scorer identities under the existing G3 measurement process before running a
decision-grade gate; the prospective mixed corpus is not thereby admitted as B7.
