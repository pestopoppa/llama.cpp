#!/usr/bin/env python3
"""Compile the actual C5 candidate warning branches with stubbed inputs; no model."""
import json
import re
import subprocess
import tempfile
from pathlib import Path

candidate = Path(__file__).resolve().parents[1]
root = Path(tempfile.mkdtemp(prefix='r23-2-warning-test-'))
spec = (candidate / 'common/speculative.cpp').read_text()
start = spec.index('        int32_t n_embd_tgt_expected = n_embd_dec;')
end = spec.index('        if (target_layer_ids_n >', start)
branch = spec[start:end]
loader = (candidate / 'src/models/dflash.cpp').read_text()
start = loader.index('    if (!has_target_hidden_size) {')
end = loader.index('    if (target_layer_ids.size()', start)
loader_branch = loader[start:end]
server = (candidate / 'tools/server/server-context.cpp').read_text()
match = re.search(r'\s*(SRV_WRN\("\[spec\] failed to measure %s memory:[\s\S]*?e\.what\(\)\);)', server)
assert match
harness = r'''#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#define LOG_INF(...) do { std::printf("INFO "); std::printf(__VA_ARGS__); } while (0)
#define LLAMA_LOG_WARN(...) do { std::printf("WARN "); std::printf(__VA_ARGS__); } while (0)
#define SRV_WRN(...) do { std::printf("WARN "); std::printf(__VA_ARGS__); } while (0)
static const char * metadata = nullptr;
static int dspark = 0;
static int llama_model_meta_val_str(void *, const char *, char * dst, size_t n) {
    if (!metadata) return -1;
    std::snprintf(dst, n, "%s", metadata);
    return 0;
}
static int llama_model_dspark_target_hidden_size(void *) { return dspark; }
static void compare(int32_t n_embd_dec, int32_t n_embd_tgt) {
    void * model_dft = nullptr;
''' + branch + r'''
}
static void loader(bool has_target_hidden_size, uint32_t n_embd_tgt) {
''' + loader_branch + r'''
}
static void memory(bool has_draft, const char * text) {
    try { throw std::runtime_error(text); }
    catch (const std::exception & e) {
''' + match.group(1) + r'''
    }
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    std::string which(argv[1]);
    try {
        if (which == "legacy_match") compare(5120, 5120);
        else if (which == "legacy_mismatch") compare(5120, 4096);
        else if (which == "unknown_zero") compare(0, 0);
        else if (which == "metadata_match") { metadata = "4096"; compare(5120, 4096); }
        else if (which == "metadata_invalid") { metadata = "garbage"; compare(5120, 4096); }
        else if (which == "derived_match") { dspark = 4096; compare(5120, 4096); }
        else if (which == "loader_missing") loader(false, 5120);
        else if (which == "loader_present") loader(true, 5120);
        else if (which == "memory_ctx") memory(true, "failed to create llama_context from model");
        else if (which == "memory_other") memory(true, "I/O error or allocation failure");
        else return 2;
    } catch (const std::exception & e) { std::printf("ERROR %s\n", e.what()); return 1; }
    return 0;
}
'''
(root / 'warning_fixture.cpp').write_text(harness)
subprocess.run(['g++', '-std=c++17', '-O0', '-Wall', '-Wextra', '-Werror', str(root / 'warning_fixture.cpp'), '-o', str(root / 'warning_fixture')], check=True)
cases = {
 'legacy_match': (0, ['INFO ', 'verified matching target hidden size 5120'], ['WARN', 'ERROR']),
 'legacy_mismatch': (1, ['ERROR ', 'does not match'], ['verified matching']),
 'unknown_zero': (1, ['ERROR ', 'unknown or nonpositive'], ['verified matching']),
 'metadata_match': (0, [], ['legacy', 'ERROR', 'WARN']),
 'metadata_invalid': (1, ['ERROR ', 'invalid dflash.target_hidden_size'], ['verified matching']),
 'derived_match': (0, ['INFO ', 'derived from'], ['legacy', 'ERROR']),
 'loader_missing': (0, ['WARN ', 'unverified', 'check pending'], ['verified matching']),
 'loader_present': (0, [], ['legacy', 'WARN']),
 'memory_ctx': (0, ['WARN ', 'failed to create llama_context', 'reservation estimate unavailable', 'compatibility unverified'], ['verified matching']),
 'memory_other': (0, ['WARN ', 'I/O error or allocation failure', 'reservation estimate unavailable', 'compatibility unverified'], ['verified matching']),
}
results = {}
for case, (expected_rc, required, forbidden) in cases.items():
    p = subprocess.run([str(root / 'warning_fixture'), case], capture_output=True, text=True)
    assert p.returncode == expected_rc, (case, p.returncode, p.stdout, p.stderr)
    assert all(word in p.stdout for word in required), (case, p.stdout)
    assert not any(word in p.stdout for word in forbidden), (case, p.stdout)
    results[case] = {'rc': p.returncode, 'output': p.stdout.strip()}
(root / 'test-results.json').write_text(json.dumps(results, indent=2) + '\n')
print(f'{len(results)} compiled branch controls passed; no model or inference')
