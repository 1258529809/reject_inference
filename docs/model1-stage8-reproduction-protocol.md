# Model 1 Stage 8 Reproduction Protocol

Baseline source commit: `648665b5f11490ddd28895dcc4ee452bf643246a`

This record deliberately keeps two reproduction routes separate. Results from one route must not be described as results from the other.

## A. Public source baseline

This route reproduces the behavior of the public repository and README command without correcting known methodological or implementation issues.

| Item | Baseline setting |
|---|---|
| Runtime | Author Theano Docker image, CPU mode, `floatX=float32` |
| Entry point | `python/model1/training_model1.py` |
| Dataset selector | `paper` |
| Accepted minority count | `n_sup=1552`; loader selects the same number from class 0, producing 3,104 labeled rows |
| Rejected count | `n_unsup=30997` |
| Batch size | 100 |
| Epochs | 401 |
| README repetitions | `n_cv=1` |
| Encoder hidden layers | `[10, 10]` |
| Decoder hidden layers | `[10, 10]` |
| Conditional prior hidden layers | `[10]` |
| Classifier hidden layers | `[70]` |
| Latent dimension | `zdim=10` |
| Components/classes | 2 |
| Decoder | Gaussian |
| Hidden activation | softplus |
| Dropout | 0.01 |
| Labeled/unlabeled learning rates | 0.0001 / 0.0001 |
| Beta | 1.1 |
| Alpha at graph construction | `1.1 * (30997 + 3104) / 3104 = 12.084761597938146` |
| VAE pretraining in training entry | None; `SS_M1` is constructed without pretrained weights |
| Data processing | Preserve `utils.load_dataset()` behavior, including pre-split scaling and separate accepted/rejected scalers |
| Update logic | Preserve the two original Adam instances and minibatch scheduling |

Known source-baseline behavior:

- The README command uses `zdim=10` because it does not override the source default.
- Learning-rate decay and alpha annealing are called after Theano functions are compiled. Stage 8 checks whether those Python attribute changes alter the compiled graph.
- `save_every` is parsed but unused; weights are saved only after all epochs for a repetition.
- The baseline route does not claim to reproduce the paper's reported metrics.

## B. Paper experiment protocol

This route is a later, separate target. It must not silently inherit source-baseline settings where the paper differs.

| Item | Paper requirement or description |
|---|---|
| Lending Club period | January 2009 through September 2012, 36-month loans |
| Input features | Seven variables listed in Table A1 |
| Missing/range processing | Remove rows with missing values and apply Table A1 ranges before modeling |
| Accepted split | 70% pool / 30% test; test retains the original default rate |
| Training accepted set | Down-sample class 0 to balance class 1 |
| Calibration | Separate post-September-2012 accepted sample; beta calibration |
| Model 1 encoder | `[10, 10]` |
| Model 1 decoder | `[10, 10]` |
| Conditional prior | `[10]` |
| Classifier | `[70]` |
| Latent dimension | 50 for the selected Lending Club Model 1 architecture in the paper table |
| Beta | 1.1 |
| Optimizer | Adam, learning rate `1e-4` |
| Monte Carlo samples | `L=1` |
| Pretraining | VAE pretraining for `q(z|x,y)` and `p(x|z)` |
| Model selection | Paper describes grid search with 10-fold cross-validation |
| Evaluation | Beta-calibrated probabilities; AUC, GINI, H-measure, recall and precision |

## Unresolved differences

1. Public source and saved public M1 arguments use `zdim=10`; the paper-selected architecture uses `zdim=50`.
2. The paper requires VAE pretraining, but `training_model1.py` does not load pretrained encoder/decoder weights. `VAE.py` imports a missing `mlp.py` module.
3. Source `n_cv` performs repeated random holdouts, not disjoint K-fold cross-validation.
4. Accepted standardization is fitted before the train/test/time split, so test and calibration feature statistics influence preprocessing.
5. Rejected data are standardized independently from accepted data.
6. The rejected 70/30 split is computed but ignored; sampling is performed from the full pre-cutoff rejected matrix.
7. The paper figure and table/README differ by one rejected observation (`30,996` versus `30,997`).
8. The source has no complete dependency lock file; the pinned author container remains the executable environment reference.
9. `--batch_size` lacks `type=int`, and the default dataset selector `cards` has no implementation in `load_dataset()`.
10. Random seeds are not fixed across Python, NumPy, sklearn or Theano.

## Data safety gate

The workflow `.github/workflows/model1-data-safety-check.yml`:

1. checks out the two data files from the immutable baseline commit;
2. verifies byte sizes and Git blob IDs;
3. reports SHA-256 and SHA-512 hashes, gzip type/integrity and file history;
4. performs a non-constructing `pickletools` opcode scan;
5. stops if an unexpected GLOBAL or dangerous construction/persistence opcode is present;
6. only after the gate passes, uses a restricted unpickler in a non-root, no-network, read-only, resource-limited container;
7. emits aggregate schema/statistics only and never prints customer records.

The security and aggregate results will be appended after the isolated workflow has completed.

