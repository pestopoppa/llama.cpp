#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

status="ok"
missing_critical=0

check_path() {
    local path="$1"
    if [[ -e "${path}" ]]; then
        printf 'ok\t%s\n' "${path}"
    else
        printf 'missing\t%s\n' "${path}"
        status="attention"
        missing_critical=1
    fi
}

printf 'repo\t%s\n' "${repo_root}"
printf 'branch\t%s\n' "$(git branch --show-current)"
printf 'commit\t%s\n' "$(git rev-parse --short HEAD)"
check_path "CMakePresets.json"
check_path "docs/build.md"
check_path "docs/epyc-llama-readiness-index.md"
check_path "tools/server/README.md"
check_path "tools/server/tests/unit/test_basic.py"

if git diff --quiet -- . ':(exclude)_libomp_src' ':(exclude)merged.profdata'; then
    printf 'worktree\tclean_or_only_untracked\n'
else
    printf 'worktree\tmodified\n'
    status="attention"
fi

printf 'status\t%s\n' "${status}"
[[ "${missing_critical}" -eq 0 ]]
