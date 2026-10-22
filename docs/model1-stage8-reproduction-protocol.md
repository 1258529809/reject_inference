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

## Stage 8 verified results

Workflow run: [39771806324](https://github.com/1258529809/reject_inference/actions/runs/39771806324), source commit `20a2ae847341b6a875a317dee8ef422b46ab1090`.

### File identity and provenance

| File | Compressed bytes | Git blob SHA-1 | SHA-256 | SHA-512 |
|---|---:|---|---|---|
| `data/supervised_lc_paper.pk.gz` | 854,027 | `63e87ab586d25f6109310eb71206267d6897c739` | `2492824dc943e67ee6777895dfa4d30caea8cce8b94d13814f7965dc9df0a187` | `240023d43257cfdcc4994d566a7ba35a68bab21f04067f523d3cd37a7cff41ed011acf63ccc6ffc89e4f9aaf38c35a93e5765e30e4eb800f069c91196a2c5ca4` |
| `data/unsupervised_lc_paper.pk.gz` | 3,011,684 | `ffe372541611db4a66504fc49186401aef642123` | `a5ef7158dd561917b09d8d714b1b7137cbc3625cd2d268a4a153f5e018620a75` | `c27dfd8d5e7005b36d989d5ce17c71df7ab269739a98c333d2bd991db0655e103c07edb80dd0e822bc055557c73dd94a06026adbd91de8f9ae9870a517cabd01` |

Both files have gzip magic `1f8b` and passed `gzip -t`. Their uncompressed pickle sizes are 7,158,284 and 21,458,598 bytes. Git history contains one unsigned introduction commit, `2061e614452a09e8336e7a8c8462fa50f4d92fda`, authored by `rogelioandrade` on 2021-09-24. This verifies repository history and byte identity, but does not independently authenticate the original Lending Club extraction.

### Static safety gate and isolated parsing

- Both streams referenced only `numpy.core.multiarray._reconstruct`, `numpy.ndarray`, and `numpy.dtype`.
- Supervised: 5 `REDUCE`, 5 `BUILD`; unsupervised: 4 `REDUCE`, 4 `BUILD`.
- Forbidden construction/persistence opcodes: none. Unexpected globals: none. Trailing bytes after `STOP`: none.
- Controlled parsing used an exact-global restricted unpickler in a network-disabled, non-root container with read-only data/root filesystems, dropped capabilities, `no-new-privileges`, 2 CPUs, 2 GiB memory, 64 PIDs, and a `noexec` temporary filesystem.
- The allowlist was not broadened after schema discovery. Exact keys are supervised `accepted_flg`, `data`, `labels`; unsupervised `accepted_flg`, `data`.

### Aggregate data contract

| Item | Supervised/accepted file | Unsupervised/rejected file |
|---|---:|---:|
| Rows | 149,125 | 536,459 |
| `data` | `(149125, 8)`, float32 | `(536459, 8)`, float32 |
| Feature columns excluding date | 7 | 7 |
| Encoded date range | 20090100–20131200 | 20090100–20120930 |
| `accepted_flg` | `(149125, 2)`, float32 | `(536459, 2)`, float32 |
| NaN / Inf in `data` | 0 / 0 | 0 / 0 |
| NaN / Inf in `accepted_flg` | 0 / 0 | 0 / 0 |

For each two-column `accepted_flg`, flattened values are exactly 0 and 1, each occurring once per row (149,125 each or 536,459 each). Supervised `labels` is `(149125, 2)` float32 with no NaN/Inf; its first column has values 0/1 with counts 131,115/18,010 (two classes). No customer-level records were printed or uploaded.

### Compiled schedule behavior

The data-free probe uses `/tmp/theano` on an explicitly executable tmpfs, matching the compilation arrangement used by the successful Stage 3–6 workflows. Inputs and random seed are fixed and evaluation uses inference mode.

| Check | Before/controls | After Python-side schedule call | Result |
|---|---:|---:|---|
| Adam learning rate | compiled at `0.0001`: parameter delta `0.000100016593933`; precompile `0.5` control delta `0.499994158745` | `optimizer.lr` became `0.5`, but compiled-function delta remained `0.000100016593933` | Post-compile `learning_decay()` does not alter the compiled update step |
| Alpha | compiled with `12.0847615979`; labeled loss `15.6555790817` | `model.alpha` became `1.02523253766`; loss remained `15.6555790817` | Post-compile `alpha_annealing()` does not alter the compiled loss |

These checks establish compiled-graph behavior, not mathematical correctness or paper-result reproduction.

## Training blockers before any real run

1. Decide explicitly whether to run the public source baseline (`zdim=10`, no VAE pretraining) or implement the separate paper protocol (`zdim=50`, VAE pretraining); do not mix them.
2. The public VAE path is incomplete because `VAE.py` imports missing `mlp.py`.
3. The README command must explicitly pass `--dset paper`; the source default `cards` is unsupported. If passing `--batch_size`, argparse leaves a command-line value as a string because no `type=int` is declared.
4. Treat learning-rate decay and alpha annealing as ineffective after graph compilation unless a separately approved compatibility change is designed; the source-baseline route must record this behavior rather than silently fix it.
5. Record/fix seeds before claiming repeatability, and separately resolve preprocessing leakage, rejected-split use, repeated-holdout-versus-K-fold behavior, and beta calibration for a paper-protocol experiment.

