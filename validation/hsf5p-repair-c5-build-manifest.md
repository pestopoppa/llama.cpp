# HSF-5P-REPAIR-C5 private review candidate

Full champion base: `c72ec82f5417bb2937310326120f149b665e8400` (C5), including C4 graph fixes and C5 fused-decode default/layout fixes. Host mirror reference: `14d0f7fa2b61c26f8edfacf91c300c37962e1d20`, ported onto fresh full C5. Production and kernel store were not changed. Root CLAUDE overlay matched the pinned origin/main object; upstream contribution policy remains in AGENTS.md.

The mirror remains opt-in through `--kv-host-mirror`, default OFF. Enabled saves drain issued copies, advance the recurrent checkpoint to the live state, flush dirty KV and retain every token. C5 context checkpoints and live draft/spec appendices remain unconditional. Both legacy `mode=lag` and `mode=fresh` now save the live boundary; no replay tail is omitted. New code changes ELF hashes; OFF equivalence below concerns behavior and state bytes in the tested scope.

## Build source and artifacts

Validated build-source manifest SHA256: `e87acfc5ee0fb855eab9e821d8201d1bfa6bdc21c97d078a63f6757b6b042083` (3336 files). Builds ran from that sealed C5-plus-repair source before this metadata-only manifest and publication commit were created. The publication claim checks every sealed file byte before staging, and stages only the enumerated repair files plus this manifest. The commit therefore maps the accepted source bytes; this added manifest is not claimed to have been a build input.

| Artifact | SHA256 |
| --- | --- |
| CPU llama-server | `bfab92dffa31ad24f65ca53ec082ef8f1751b54d56efdc0f106bef797a72ee2e` |
| CPU libllama.so | `1406c96710ed2d0d4f5fbc0378ec2118f0cd7f8011925387bc5b5a8a0d6c0f49` |
| HIP llama-server | `802a3e0c7bcfcb23c661dd879ec6b12d76048a4525ac8a2aa3b336f561bfa6f4` |
| HIP libllama.so | `4b6e69b4a86866e4440aaa412b0c1e42d15f0fff153e599e8266fa58d483b41b` |

CPU and HIP full builds completed under owner quarter grants on 2026-10-27, with whole-job affinity and `-j24`. CPU used GCC15/native/OpenMP; HIP used gfx90a/rocWMMA/native. Each build/test set its own LD_LIBRARY_PATH and passed verify_ggml_linkage. Compiled mirror symbols and both C5 fused-default/layout strings were checked. Actual library target mtimes were rechecked with stat -L against the repaired source; earlier symlink-only mtime records were superseded.

## Accepted model-free validation

- Host mirror unit tests passed.
- Actual hybrid-memory/binding save/erase/restore fixture, OFF/lag/fresh: saved43 = erased43 = restored43, replay0. The stale periodic checkpoint has a 37-token tail; enabled save advances it. Full target and live draft memory bytes plus nonempty opaque speculative appendix bytes equal an independently linked pinned-C5 reference. The opaque spec payload tests production appendix serialization, not model drafter behavior.
- CPU seeded backend ops: C5 OFF, candidate OFF and actual mirror ON CSVs identical; ON performed D2H, poisoned destination and verified H2D for 143 cases / 463875872 bytes.
- Native GPU seeded ADD/CPY/SET_ROWS subset: all three CSV SHA256 values `442f99d366d79206d6cdfa7e0c0e082e2475e8f0f59972eb60de9ba50cd454a6`; ON verified 429 cases / 1391627616 bytes. Positive KFD plus VRAM samples taken during the three processes: 690 / 686 / 734. GPU job ended 08:11:16Z rc0. MAIN accepted this scope.

GitNexus blast radius remains UNKNOWN for the unindexed private tree, not LOW. Manual caller/source review and these checks do not replace the pending current-C5 graph/impact gate; HIGH/CRITICAL remains a fold stop for the owner.

This is a private review branch, not a champion fold or production promotion. No model inference, attribution or performance benchmark was run. Full CPU/GPU model regressions and any promotion require later explicit operator authorization.

Evidence scratch root: `/mnt/raid0/llm/tmp/codex-ni-20261027-hsf5p-repair-c5`. CPU/HIP/gpu-ops logs, source/script seals, linkage, compiled symbols and raw receipts are retained there. The expected GPU-TERMINAL.flag was absent and its watcher was absent at closeout; completion was established from canonical ended/complete records and actual result files, not that watcher.
