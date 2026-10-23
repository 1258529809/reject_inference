# Model 1 Stage 10: one complete real-data epoch

## Scope and outcome

Stage 10 validates exactly one complete epoch of the author's public Model 1 source baseline. It preserves the original alternating minibatch rule and original Theano training functions while retaining the Stage 9 restricted data loader and container isolation. It does not enter epoch 1, run 401 epochs, load pretrained weights, tune parameters, or claim reproduction of the paper's final result.

- Stage 9 baseline: `98757a3f1683c553f606340bd6fe75e85345f779`
- Execution commit: `962baf8338ded51ddc4d5dccc90006d257221408`
- Successful Actions run: [39925544573](https://github.com/1258529809/reject_inference/actions/runs/39925544573)
- Result artifact: `model1-stage10-aggregate-result`, ID `12489316968`, SHA-256 digest `3b02ed48f5b4780ec003d006fd602a602542975a5f63a9c3aacc59cde5554a01`
- Outcome: one epoch completed; original calibrated evaluation completed; no numerical or scheduling failure

## Configuration

| Setting | Value |
|---|---|
| `epochs` / `n_cv` | 1 / 1 |
| `batch_size` | 100 |
| `n_sup` / selected labeled rows | 1,552 per class / 3,104 total |
| `n_unsup` / selected rejected rows | 30,997 |
| Test / calibration rows | 16,110 / 2,000 |
| Input / classes / latent dimensions | 7 / 2 / 10 |
| Classifier hidden dimensions | `[70]` |
| Encoder / decoder hidden dimensions | `[10, 10]` / `[10, 10]` |
| Conditional-prior hidden dimensions | `[10]` |
| `beta` / derived `alpha` | 1.1 / 12.084761597938146 |
| Labeled / unlabeled learning rates | 0.0001 / 0.0001 |
| Activation / dropout / decoder | softplus / 0.01 / Gaussian |
| Seed | 9009 for Python, NumPy, and `PYTHONHASHSEED` |
| Environment | CPU, Theano 1.0.4, NumPy 1.15.2, `floatX=float32` |

No pretrained weights were loaded.

## Exact author scheduling check

The wrapper mirrors the source expressions:

```text
labeled_batches   = 3104 // 100  = 31
unlabeled_batches = 30997 // 100 = 309
interval           = int((3104 + 30997) / 3104.0) = 10
batches_per_epoch  = 31 + 309 = 340
is_labeled         = (iteration % 10 == 1) and labeled_counter < 31
```

The schedule was simulated before model construction and checked again after execution. Actual counts were exactly 31 labeled and 309 unlabeled updates. Batch indices were labeled 0-30 and unlabeled 0-308. As in the original source, tail rows were omitted: 4 labeled and 97 unlabeled rows. The only executed epoch index was 0.

## Training diagnostics

- Both original `labeled_training()` and `unlabeled_training()` functions compiled.
- Compilation plus diagnostic monitor construction took 64.684184 seconds.
- The 340 training updates took 0.996652 seconds in the fixed GitHub Actions CPU environment.
- Every returned loss was finite.
- All 30 model arrays were checked for NaN/Inf after every update and remained finite.
- All 30 arrays differed between the start and end of the epoch; total combined parameter delta norm was 0.031103522195583648.

| Loss | Count | First | Last | Minimum | Maximum | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Labeled total | 31 | 19.2055337908 | 19.0353492531 | 18.5701756441 | 19.2894139359 | 19.0052683955 |
| Classifier component | 31 | 8.3705120087 | 8.3757886887 | 8.3705120087 | 8.3787040710 | 8.3760096027 |
| Reconstruction mean | 31 | 10.1418179295 | 9.9664132074 | 9.5007551796 | 10.2203464644 | 9.9361092042 |
| Unlabeled total | 309 | 10.0037011112 | 9.5984180109 | 9.2523982039 | 11.2021209730 | 9.9265574051 |

Losses are minibatch observations under stochastic latent sampling, so first-to-last movement is not treated as a monotonic convergence proof.

### Gradients and parameter groups

All 30 monitored gradients were finite before and after the epoch.

| Gradient path | Pre-epoch norm | Post-epoch norm |
|---|---:|---:|
| Labeled | 20.9137243307 | 21.0234855061 |
| Unlabeled | 0.4999999506 | 0.5000000270 |

The unlabeled value reflects the author's gradient-clipping path. Combined epoch parameter changes were:

| Group | Arrays | Delta norm |
|---|---:|---:|
| Classifier | 4 | 0.0196014842 |
| Encoder | 8 | 0.0037558971 |
| Decoder | 8 | 0.0233311218 |
| Conditional prior | 10 | 0.0049762339 |

These checks establish finite executable gradients and actual updates, not mathematical correctness of the gradients.

## Evaluation phase

Evaluation began only after all 340 training updates and numerical gates passed. The wrapper called the unmodified `SS_M1.get_performance()` with the real test and calibration arrays.

- Beta-calibrated evaluation: successful
- Score shape: `(16110,)`
- Test AUC: **0.62022562853270857**
- Evaluation time: 0.152256 seconds

This is a single fixed-seed, one-epoch source-baseline observation. It is not the paper's reported cross-validation result and was not used to tune the model.

## Safety and immutability

- The same pinned image digest and exact Stage 9 static pickle/global allowlist were used.
- The non-constructing static scan block was byte-identical to Stage 9 and passed for both files.
- Controlled parsing and training ran without network access, as non-root, with read-only source/data/root mounts, dropped capabilities, `no-new-privileges`, CPU/memory/PID limits, `noexec` `/tmp`, and a separate executable `/theano` compilation directory.
- Only `/results` was writable. It contained one 651-byte aggregate JSON summary; no customer-level records, model weights, data arrays, or predictions were uploaded.
- The final job check reported `POST_RUN_AUTHOR_SOURCE_DATA_WEIGHTS_OUTPUT_UNCHANGED=PASS`.

## Differences from direct `training_model1.py`

The update order, batch counters, epsilon shape, original losses, original Adam functions, and epoch-zero learning-rate/alpha scheduling calls were preserved. The independent wrapper differs intentionally:

- restricted unpickling replaces unrestricted `cPickle.load`;
- all controllable random seeds are fixed;
- the one-epoch boundary and 31/309 counts are asserted explicitly;
- finite gradient monitors run before and after the epoch without updates;
- parameters are checked after every update;
- `model.summary()`, latent-space plotting, weight saving, result pickle writing, and historical output-directory creation are omitted;
- the original `get_performance()` is called directly after the training gate;
- aggregate diagnostics are written only to the dedicated Stage 10 artifact.

Known source-baseline behaviors remain unchanged: preprocessing leakage, separate accepted/rejected scaler fits, ignored rejected 70/30 split, omitted tail batches, and ineffective post-compilation learning-rate/alpha mutations. No author source behavior was silently repaired.

