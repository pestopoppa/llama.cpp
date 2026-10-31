# R23-2 DFlash logging controls

Base is the full C5 champion commit `c72ec82f5417bb2937310326120f149b665e8400`; frozen production branch tip `ffc1bac82eeca6f9099e1ccd9ba49703c460a115` is its ancestor (`git merge-base --is-ancestor`, exit 0). The source patch was reviewed as private commit `ce6918b404b817d4d935ac1c549a89eda4995819` and this branch preserves its file contents.

The DFlash model-free control extracts the actual changed source sections, compiles them with mock APIs, and executes match, mismatch, invalid and explicit metadata, DSpark, and draft/MTP estimation error cases. The exact test script is `tests/test-dflash-warning-controls.py`; its SHA-256 is listed below. It is not a full loader integration, a full CPU/GPU build, a model run, or a regression or deployment gate. No claim about live logs or production is made.

On 2026-10-31 the source fixture was run using the original 26-27 mutex and the R6 supervisor/admission, with SMT CPUs 122,123 after the physical choices were occupied, `OMP_NUM_THREADS=1`, deadline 120 seconds. Raw output and sampled owned-process-tree RSS are in `native-controls.log`; exit was 0 and children drained. RSS is sampled and can miss transients. The admission on 26,27, 50,51, 74,75 and 2,3 refused overlapping grants; no test ran on those pairs.

Command (fixture `files/` held exact copies of the three Git blobs):

```sh
flock -w 10 /mnt/raid0/llm/tmp/compute-flags/prep-locks/26-27.lock env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 R232_SOURCE_ROOT=/mnt/raid0/llm/tmp/codex3-r232-warnings-20261031/files taskset -c 122,123 /usr/bin/python3 /mnt/raid0/llm/tmp/codex3-k4a-20261031/capsule/prep_supervisor.py 120 -- /usr/bin/python3 /mnt/raid0/llm/tmp/codex3-r232-warnings-20261031/native_controls.py
```

SHA-256 of tested inputs and receipt:

```text
cc98b5f34e2ad01940990723b94175820a7514d2fcfec98198ce3432439283f6  common/speculative.cpp
516daa364e8a2516bb5120ebbb0623e05430432f3355285464d8a05dad12b07e  src/models/dflash.cpp
3e2e170db7dcba307c1f7bbd7cf0bf2276c9452f493602c2504a9a857b51fe63  tools/server/server-context.cpp
d8972817f020539f7514c24390fb5c1a7f2b90c1334e59f20b3241201e125e94  tests/test-dflash-warning-controls.py
b74c35c6a6cbe8e4537ad59a5bd3b2cec310a9431552bd174882c605391259d4  evidence/r23-2/native-controls.log
63222eb38697a0d15a7cee063d6e26fcbb9355c7e0c8701647a0dd9b7009ff65  prep_supervisor.py (approved external capsule)
6e0340cd380b7d268e0b68c18b35adeb5f71e7afaf5d7a6b408fd27d35abc2c0  prep_admission.py (approved external capsule)
```
