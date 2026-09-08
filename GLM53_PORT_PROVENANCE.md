# GLM-5.3-Flash text and native-MTP candidate

- Champion base: `ef81196d5bdd4190b46dff4ae7eecc333a46c8ce`
- Text/model support source: upstream PR #27773 head `8134115f88ed8018474e7db69afcfe97fb097fc4`
- Native-MTP source: upstream PR #27917 head `5b8593b5451ec45fd4a81fb844efb6be9b45fd36`
- Scope: private experimental text-only port. Conversion and `tools/mtmd` vision changes were excluded.
- Adaptation: retained the champion's model APIs, recurrent rollback, hybrid-index state serialization, and kernel optimizations; added GLM5Next schema/model graphs, k-pool index memory, native MTP selection reuse, dual `glm5-next`/`glm5next` metadata handling, and official true defaults for absent tail-selection/index-sharing keys.
- Regression scope: retained champion Qwen hybrid-index filtering and legacy `tokenizer_pre=glm4` merge behavior; `ignore_merges` is enabled only for GLM5Next using `glm4`, or for the explicit `glm5` tokenizer label.
- Test scope: `test-llama-archs` contains naming/prefix assertions, while its generic synthetic model fixture explicitly skips GLM5Next and is not GLM coverage. `tests/glm53` provides deterministic KDA/k-pool/mHC/NextN GGUF schema fixtures. `test-glm5next-mtp-rollback` checks target suffix rollback and full restore into an already-used target context through public/common APIs; it does not prove native draft dispatch or native-MTP state restore.
- Pending: CPU compile and artifact load, native MTP dispatch/state-restore and pool-boundary correctness, then CPU/HIP champion regression and performance measurement. No inference result or performance claim exists yet.
