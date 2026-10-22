# Model 1 Stage 9: real-data one-batch smoke test

## Scope and result

This stage validates only the author-source baseline training path on verified real-data arrays. It is not a reproduction of the paper's reported AUC. The successful run compiled the author's two Theano training functions, executed exactly one labeled and one unlabeled Adam update, and then called the author's original calibrated evaluation function. No epoch loop, pretrained weights, optimizer replacement, or hyperparameter search was used.

- Baseline: `5304bec5034fcf35e04893e73b21a58696561f45`
- Execution commit: `59f3a9b85a4de91c9529e62837df704a3aa4c3fb`
- Successful Actions run: [39775848902](https://github.com/1258529809/reject_inference/actions/runs/39775848902)
- Result: training smoke test passed; original evaluation passed; `epochs_run=0`

## Reproducibility and isolation

- Fixed image: `rogelioandrade/theano_setup@sha256:69c334b33ae5df566d8f17363428bb5a7030743f7bcc7ba27ab29e616f7b848f` (Linux/amd64).
- Environment: Python 2 author image, Theano 1.0.4, NumPy 1.15.2, CPU, `floatX=float32`.
- Seeds: Python, NumPy, and `PYTHONHASHSEED` set to `9009`; latent noise was generated from a fixed derived seed.
- Static inspection allowed exactly `numpy.dtype`, `numpy.ndarray`, and `numpy.core.multiarray._reconstruct`; both files had no unexpected globals. The executable-global allowlist was not broadened.
- First object construction used the exact-global restricted unpickler in a network-disabled, non-root container with read-only root/source/data mounts, dropped capabilities, `no-new-privileges`, CPU/memory/PID limits, and a `noexec` `/tmp`.
- Theano compilation used a separate executable temporary directory, matching the constraint established in Stages 3-6; data remained read-only and `/output` was not mounted.
- No customer-level rows or original data contents were printed or uploaded.

## Data identity and aggregate contract

| File | Bytes | Git blob SHA-1 | SHA-256 |
|---|---:|---|---|
| `data/supervised_lc_paper.pk.gz` | 854,027 | `63e87ab586d25f6109310eb71206267d6897c739` | `2492824dc943e67ee6777895dfa4d30caea8cce8b94d13814f7965dc9df0a187` |
| `data/unsupervised_lc_paper.pk.gz` | 3,011,684 | `ffe372541611db4a66504fc49186401aef642123` | `a5ef7158dd561917b09d8d714b1b7137cbc3625cd2d268a4a153f5e018620a75` |

Both checksum checks passed in the successful run. The Stage 8 exact contract was retained, including `accepted_flg` in both files:

| Aggregate | Accepted/supervised | Rejected/unsupervised |
|---|---:|---:|
| Full rows | 149,125 | 536,459 |
| `data` | `(149125, 8)` float32 | `(536459, 8)` float32 |
| `accepted_flg` | `(149125, 2)` float32 | `(536459, 2)` float32 |
| Feature columns after date removal | 7 | 7 |
| Encoded date range | 20090100-20131200 | 20090100-20120930 |
| NaN / Inf in validated arrays | 0 / 0 | 0 / 0 |

Supervised labels are `(149125, 2)` float32; first-column class counts are 131,115 and 18,010. The fixed-seed source-baseline selection supplied 3,104 labeled rows, 30,997 rejected rows, 16,110 test rows, and 2,000 calibration rows to the smoke-test wrapper. All selected model inputs were float32 and finite.

## Model configuration and tensor contract

| Setting | Value |
|---|---|
| Dataset route | `paper` |
| Input / class / latent dimensions | 7 / 2 / 10 |
| Batch size | 100 |
| Classifier hidden dimensions | `[70]` |
| Encoder / decoder hidden dimensions | `[10, 10]` / `[10, 10]` |
| Conditional-prior hidden dimensions | `[10]` |
| Decoder | Gaussian |
| Activation / dropout | softplus / 0.01 |
| `beta`, derived `alpha` | 1.1 / 12.084761597938146 |
| Labeled / unlabeled learning rates | 0.0001 / 0.0001 |
| Pretrained weights | none |

The real minibatch input was `(100, 7)`. Class probabilities and the author's one-hot predictions were both `(100, 2)` float32 and finite.

The full Route A architecture has **30** parameter arrays: classifier 4, encoder 8, decoder 8, and conditional prior 10. The earlier expectation of 26 arrays came from the reduced one-hidden-layer synthetic configuration used in Stages 5-6 and does not apply to the author's two-layer Route A encoder/decoder.

## One-batch update results

Both `labeled_training()` and `unlabeled_training()` compiled. Gradient and parameter checks covered all 30 arrays.

| Path | Returned loss | Deterministic probe before | Probe after | Probe delta | Gradient norm | Total parameter delta norm | Unchanged arrays |
|---|---:|---:|---:|---:|---:|---:|---:|
| Labeled | 19.201488973012882 | 19.201121594432614 | 19.175608000738421 | -0.025513593694193 | 20.87276005897894 | 0.001886481143405 | 0 / 30 |
| Unlabeled | 10.006204035494738 | 10.006185631312494 | 10.005277668784785 | -0.000907962527709 | 0.500000006813209 | 0.001218501253746 | 0 / 30 |

The labeled returned components included classifier loss 8.364740371704102 and mean reconstruction cost 10.143546503413342. The unlabeled gradient norm reflects the author's clipping path. Every logged gradient, loss, post-update parameter, probability, and prediction was finite.

Combined update norms by source component:

| Component (array count) | Labeled delta norm | Unlabeled delta norm |
|---|---:|---:|
| Classifier (4) | 0.001340298063287 | 0.000382710830711 |
| Encoder (8) | 0.000503451176199 | 0.000212876571205 |
| Decoder (8) | 0.001134615438759 | 0.001120972336241 |
| Conditional prior (10) | 0.000470740826022 | 0.000190741480155 |

These values show that the two original Adam paths updated all Route A arrays on one fixed real minibatch. They do not establish convergence or mathematical equivalence to the paper.

## Evaluation

After both update gates passed, the wrapper called the author's unmodified `SS_M1.get_performance()` with 16,110 test rows and 2,000 calibration rows. The installed `betacal`, `rpy2`, R, and `hmeasure` path completed successfully. The returned score vector was `(16110,)`, all values were finite, and the reported AUC was **0.50344892906491379**.

This AUC is an environment and end-to-end evaluation smoke test after only two minibatch updates. It was not used for tuning and must not be compared with the paper's final result.

## Preserved source-baseline behavior and wrapper differences

Preserved and explicitly logged:

- scaling before the train/test time split (known preprocessing leakage);
- independent accepted/rejected scaler fits;
- computation but non-use of the rejected 70/30 split;
- sampling the rejected subset from the full pre-cutoff rejected pool;
- fixed compiled learning rate and alpha behavior documented in Stage 8;
- no VAE pretraining and `zdim=10`;
- no correction to the original tail-batch or scheduling behavior.

The wrapper differs from invoking `training_model1.py` in these controlled ways:

- it uses an exact-global restricted unpickler and mirrors the author's transformations instead of calling unrestricted `cPickle.load`;
- it selects fixed-seed arrays and executes only batch index 0 for each original training function;
- it compiles diagnostic gradient and deterministic inference probes without updates;
- it does not enter the author's epoch/minibatch scheduler and writes no weights, pickle results, plots, arguments, or other output files.

Therefore this is a safe minimal training-chain validation, not an equivalent execution of the complete training entry point.

## Immutability evidence and diagnostic history

The successful workflow ended with `POST_RUN_AUTHOR_SOURCE_DATA_WEIGHTS_OUTPUT_UNCHANGED=PASS`. Comparison from the baseline contains only Stage 9-specific wrapper/workflow/report files; author Model 1 source, the two data blobs, pretrained weights, and historical output are unchanged.

Before the successful run, three bounded diagnostic attempts identified only wrapper/workflow issues: checksum-file line construction, Python 2 C-unpickler subclassing, and the one-hot prediction shape assertion. None required changing author source/data, broadening the pickle allowlist, or running training beyond the two permitted updates.

