# L4 Repo-Readiness Audit (2026-07-06)

Scope: `epyc-llama` (`/mnt/raid0/llm/llama.cpp`)

## Blockers checked

- L4.health_automation
- L4.analysis_reports
- L4.security_audit
- L4.replay_analysis

## Audit summary

- Date run: 2026-07-06
- Result: `L4.health_automation` and `L4.security_audit` have partial evidence in-repo; `L4.analysis_reports` and `L4.replay_analysis` are missing a concrete docs/automation artifact.

## Evidence review

- `L4.health_automation`
  - `tools/server/README.md`: health endpoint documented (`GET /health`, `GET /v1/health`) and public behavior described.
  - `tools/server/tests/unit/test_basic.py`: `test_server_start_simple` validates `GET /health`.
  - `tools/server/tests/unit/test_sleep.py`: health endpoint remains responsive during sleep lifecycle.
  - `tools/server/tests/unit/test_security.py`: `test_access_public_endpoint` includes `/health`.
  - Gap: no dedicated readiness automation script/checklist in this repo for periodic health-health drift reporting.

- `L4.analysis_reports`
  - `docs/autoparser.md`: analysis pipeline and artifact references for `tools/parser/template-analysis.cpp`.
  - `tools/parser/template-analysis.cpp`: CLI analysis tool exists and writes structured analysis output.
  - Gap: no "analysis report" landing doc/checklist that tracks report generation cadence, retention, or ownership for L4 readiness scoring.

- `L4.security_audit`
  - `tools/server/README-dev.md`: explicit security guidance for default-off dangerous features and file boundaries.
  - `tools/server/tests/unit/test_security.py`: endpoint auth, CORS, and proxy header policy tests.
  - `SECURITY.md`: repository security reporting policy.
  - `.github/workflows/server.yml`: server test workflow includes unit test execution.
  - Gap: no repo-local security-audit checklist artifact that maps the above checks to L4 acceptance criteria.

- `L4.replay_analysis`
  - `tools/completion/completion.cpp`: references replay of last token internally.
  - Gap: no replay-analysis reporting harness, replay dataset/replay CLI, or readiness-oriented replay report.

## Action items for these blockers

- [ ] Add a repo-local `docs/` checklist for L4 blocker evidence and required cadence:
  - one entry per L4 criterion,
  - owner + target file paths,
  - verification command(s),
  - date last verified.
- [ ] Add dedicated automation/docs artifact for `L4.replay_analysis` (if this blocker is still expected by governance),
  and explicitly link it from `docs/epyc-llama-readiness-index.md` + `handoffs/active/master-handoff-index.md`.

## Commands run (read-only)

- `gitnexus status` (repo showed stale index)
- `gitnexus impact AGENTS --repo /mnt/raid0/llm/epyc-root --direction upstream` (LOW risk)
- `rg -n "L4\\.health_automation|L4\\.analysis_reports|L4\\.security_audit|L4\\.replay_analysis" ...`
- `rg -n "GET /health|/health|test_access_public_endpoint|test_server_start_simple|template-analysis|security|Replay|replay" ...`
- `python3 -m pytest ...` (not run: `No module named pytest`)
