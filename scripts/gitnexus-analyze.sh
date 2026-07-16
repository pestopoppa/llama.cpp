#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
repo_name="${GITNEXUS_REPO_NAME:-epyc-llama}"
lock_path="/tmp/gitnexus-${repo_name}-analyze.lock"

exec 9>"${lock_path}"
if ! flock -n 9; then
    echo "gitnexus analyze already running for ${repo_name}" >&2
    exit 75
fi

cd "${repo_root}"

if ! command -v gitnexus >/dev/null 2>&1; then
    echo "gitnexus not found on PATH" >&2
    exit 127
fi

exec gitnexus analyze "$@"
