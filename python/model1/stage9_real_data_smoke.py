from __future__ import print_function

import gzip
import os
import random
import traceback

import pickle
import numpy as np
import theano
import theano.tensor as T
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from SS_M1 import SS_M1
from utils import floatX, shared_dataset


SEED = 9009
BATCH_SIZE = 100
N_SUP_MINORITY = 1552
N_UNSUP = 30997
NUM_CLASSES = 2

EXPECTED_SHA256 = {
    "supervised_lc_paper.pk.gz": "2492824dc943e67ee6777895dfa4d30caea8cce8b94d13814f7965dc9df0a187",
    "unsupervised_lc_paper.pk.gz": "a5ef7158dd561917b09d8d714b1b7137cbc3625cd2d268a4a153f5e018620a75",
}

ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = (module, name)
        if key not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                "Forbidden global: {}.{}".format(module, name)
            )
        return ALLOWED_GLOBALS[key]

    def find_global(self, module, name):
        return self.find_class(module, name)

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent IDs are forbidden")


def restricted_load(path, expected_keys):
    with gzip.open(path, "rb") as handle:
        obj = RestrictedUnpickler(handle).load()
    if type(obj) is not dict:
        raise TypeError("Top-level object must be an exact dict")
    if sorted(obj.keys()) != sorted(expected_keys):
        raise ValueError("Unexpected keys for {}".format(os.path.basename(path)))
    for key, value in obj.items():
        if type(value) is not np.ndarray or value.dtype.kind not in "biufc":
            raise TypeError("{} must be an exact numeric ndarray".format(key))
        if not np.isfinite(value).all():
            raise ValueError("{} contains NaN/Inf".format(key))
    return obj


def load_author_baseline_data(data_dir):
    supervised = restricted_load(
        os.path.join(data_dir, "supervised_lc_paper.pk.gz"),
        ["accepted_flg", "data", "labels"],
    )
    unsupervised = restricted_load(
        os.path.join(data_dir, "unsupervised_lc_paper.pk.gz"),
        ["accepted_flg", "data"],
    )

    x_a_all = supervised["data"]
    y_a_all = supervised["labels"][:, 0]
    x_r_all = unsupervised["data"]
    if x_a_all.shape != (149125, 8) or x_r_all.shape != (536459, 8):
        raise ValueError("Stage 8 data shape contract changed")

    date_a = x_a_all[:, 0]
    x_a = x_a_all[:, 1:]
    accepted_scaler = preprocessing.StandardScaler()
    x_a = accepted_scaler.fit_transform(x_a)

    x_cal = x_a[date_a >= 20121001, :]
    y_cal = y_a_all[date_a >= 20121001]
    idx_c = random.sample(range(0, x_cal.shape[0]), 2000)
    x_cal = x_cal[idx_c]
    y_cal = y_cal[idx_c]

    date_r = x_r_all[:, 0]
    x_r = x_r_all[:, 1:]
    y_r = np.asarray([-1] * x_r.shape[0])
    rejected_scaler = preprocessing.StandardScaler()
    x_r = rejected_scaler.fit_transform(x_r)

    x_a = x_a[date_a < 20121001, :]
    y_a = y_a_all[date_a < 20121001]
    x_r = x_r[date_r < 20121001, :]
    y_r = y_r[date_r < 20121001]

    accepted_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
    for train_index_a, test_index_a in accepted_split.split(x_a, y_a):
        x_tr_a, x_te_a = x_a[train_index_a], x_a[test_index_a]
        y_tr_a, y_te_a = y_a[train_index_a], y_a[test_index_a]

    # Preserve the source's computed-but-unused rejected 70/30 split.
    rejected_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
    for train_index_r, test_index_r in rejected_split.split(x_r, y_r):
        unused_x_tr_r = x_r[train_index_r]
        unused_x_te_r = x_r[test_index_r]
    if unused_x_tr_r.shape[0] + unused_x_te_r.shape[0] != x_r.shape[0]:
        raise AssertionError("Rejected split accounting failed")

    x_a_1 = x_tr_a[y_tr_a == 1, :]
    y_a_1 = y_tr_a[y_tr_a == 1]
    x_a_0 = x_tr_a[y_tr_a == 0, :]
    y_a_0 = y_tr_a[y_tr_a == 0]
    if N_SUP_MINORITY > y_a_1.sum():
        raise RuntimeError("Requested minority sample exceeds available rows")

    idx_a_0 = random.sample(range(0, x_a_0.shape[0]), N_SUP_MINORITY)
    idx_a_1 = random.sample(range(0, x_a_1.shape[0]), N_SUP_MINORITY)
    # Deliberately sample from the full pre-cutoff rejected matrix, as source does.
    idx_r = random.sample(range(0, x_r.shape[0]), N_UNSUP)

    x_tr_a = np.r_[x_a_1[idx_a_1, :], x_a_0[idx_a_0, :]]
    y_tr_a = np.r_[y_a_1[idx_a_1], y_a_0[idx_a_0]]
    x_tr_r = x_r[idx_r, :]
    x_tr_a, y_tr_a = shuffle(x_tr_a, y_tr_a)

    encoder = OneHotEncoder(n_values=NUM_CLASSES)
    y_tr_ohe = encoder.fit_transform(y_tr_a.reshape(y_tr_a.shape[0], 1)).toarray()

    arrays = [x_tr_a, y_tr_ohe, x_tr_r, x_te_a, y_te_a, x_cal, y_cal]
    arrays = [np.asarray(value, dtype=floatX) for value in arrays]
    for value in arrays:
        if not np.isfinite(value).all():
            raise ValueError("Transformed data contains NaN/Inf")

    print(
        "DATA selected_labeled={} selected_unlabeled={} test={} calibration={} "
        "features={} seed={}".format(
            arrays[0].shape[0], arrays[2].shape[0], arrays[3].shape[0],
            arrays[5].shape[0], arrays[0].shape[1], SEED
        )
    )
    print(
        "DATA_DTYPES labeled_x={} labeled_y={} unlabeled_x={} test_x={} "
        "all_finite=True".format(
            arrays[0].dtype, arrays[1].dtype, arrays[2].dtype, arrays[3].dtype
        )
    )
    print(
        "SOURCE_BEHAVIOR preprocessing_leakage_preserved=True "
        "separate_scalers_preserved=True rejected_split_ignored_preserved=True"
    )
    return arrays


class Args(object):
    pass


def make_args():
    args = Args()
    args.lr_u = 0.0001
    args.lr_l = 0.0001
    args.batch_size = BATCH_SIZE
    args.hdim_cls = [70]
    args.hdim_enc = [10, 10]
    args.hdim_dec = [10, 10]
    args.hdim_gmm = [10]
    args.zdim = 10
    args.dec = "gaussian"
    args.num_k = 2
    args.nlayers_enc = 2
    args.nlayers_dec = 2
    args.nlayers_cls = 1
    args.nlayers_gmm = 1
    args.dropout_rate = 0.01
    args.hl_activation = "softplus"
    return args


PARAMETER_NAMES = {
    "classifier": ["hidden_W", "hidden_b", "softmax_W", "softmax_b"],
    "encoder": [
        "hidden0_W", "hidden0_b", "hidden1_W", "hidden1_b",
        "mu_W", "mu_b", "logvar_W", "logvar_b",
    ],
    "decoder": [
        "hidden0_W", "hidden0_b", "hidden1_W", "hidden1_b",
        "mu_W", "mu_b", "logvar_W", "logvar_b",
    ],
    "conditional_prior": [
        "shared_hidden_W", "shared_hidden_b",
        "class0_mu_W", "class0_mu_b", "class0_logvar_W", "class0_logvar_b",
        "class1_mu_W", "class1_mu_b", "class1_logvar_W", "class1_logvar_b",
    ],
}


def parameter_layout(model):
    groups = [
        ("classifier", list(model.cls_mlp.params)),
        ("encoder", list(model.enc_mlp.params)),
        ("decoder", list(model.dec_mlp.params)),
        ("conditional_prior", list(model.gmm_weights)),
    ]
    expected = {"classifier": 4, "encoder": 8, "decoder": 8, "conditional_prior": 10}
    flat = []
    for group, params in groups:
        if len(params) != expected[group]:
            raise AssertionError("Unexpected {} parameter count".format(group))
        for index, parameter in enumerate(params):
            flat.append((group, PARAMETER_NAMES[group][index], parameter))
    if len(flat) != 30:
        raise AssertionError("Expected 30 arrays for the README two-layer architecture")
    if sorted(id(item[2]) for item in flat) != sorted(id(p) for p in model.params):
        raise AssertionError("Parameter grouping differs from model.params")
    return flat


def snapshot(flat):
    values = [np.array(item[2].get_value(borrow=True), copy=True) for item in flat]
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("Model parameter contains NaN/Inf")
    return values


def gradients_in_flat_order(model, gradients, flat):
    by_parameter_id = dict(
        (id(parameter), gradient)
        for parameter, gradient in zip(model.params, gradients)
    )
    ordered = [by_parameter_id[id(item[2])] for item in flat]
    if len(ordered) != len(flat):
        raise AssertionError("Could not align gradients with grouped parameters")
    return ordered


def report_gradients(path, flat, gradients):
    if len(gradients) != len(flat):
        raise AssertionError("Gradient count mismatch")
    total_sq = 0.0
    for index, ((group, name, parameter), gradient) in enumerate(zip(flat, gradients)):
        value = np.asarray(gradient)
        if value.shape != parameter.get_value(borrow=True).shape:
            raise AssertionError("Gradient shape mismatch")
        if not np.isfinite(value).all():
            raise ValueError("Gradient contains NaN/Inf")
        norm = float(np.linalg.norm(value.astype("float64")))
        total_sq += norm * norm
        print(
            "GRAD path={} index={} group={} name={} shape={} dtype={} norm={:.17g} finite=True".format(
                path, index, group, name, value.shape, value.dtype, norm
            )
        )
    print("GRAD_SUMMARY path={} count={} norm={:.17g} all_finite=True".format(
        path, len(gradients), float(np.sqrt(total_sq))
    ))


def report_update(path, flat, before, after):
    unchanged = 0
    total_sq = 0.0
    for index, ((group, name, parameter), old, new) in enumerate(zip(flat, before, after)):
        delta = new.astype("float64") - old.astype("float64")
        changed = not np.array_equal(old, new)
        unchanged += int(not changed)
        norm = float(np.linalg.norm(delta))
        total_sq += norm * norm
        print(
            "PARAM_UPDATE path={} index={} group={} name={} shape={} dtype={} "
            "delta_norm={:.17g} changed={} finite=True".format(
                path, index, group, name, new.shape, new.dtype, norm, changed
            )
        )
    print(
        "PARAM_UPDATE_SUMMARY path={} count={} unchanged={} total_delta_norm={:.17g} "
        "all_finite=True".format(path, len(flat), unchanged, float(np.sqrt(total_sq)))
    )
    if unchanged != 0:
        raise AssertionError("{} parameters did not all update".format(path))


def compile_gradient_function(model, gradients, path, data_x, data_y=None):
    index = T.lscalar("{}_gradient_index".format(path))
    givens = {model.x: data_x[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]}
    if data_y is not None:
        batch_y = data_y[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
        givens.update({model.labels: batch_y, model.y_in_enc: batch_y, model.y_in_gmm: batch_y})
    return theano.function(
        inputs=[index, model.eps, model.training_mode],
        outputs=gradients,
        givens=givens,
        no_default_updates=True,
    )


def compile_loss_probe(model, cost, path, data_x, data_y=None):
    index = T.lscalar("{}_loss_index".format(path))
    givens = {model.x: data_x[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]}
    if data_y is not None:
        batch_y = data_y[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
        givens.update({model.labels: batch_y, model.y_in_enc: batch_y, model.y_in_gmm: batch_y})
    return theano.function(
        inputs=[index, model.eps, model.training_mode],
        outputs=cost,
        givens=givens,
        no_default_updates=True,
    )


def main():
    if os.getuid() == 0:
        raise RuntimeError("Stage 9 must not run as root")
    if os.path.exists("/work/output") or os.path.exists("/work/model1/output"):
        raise RuntimeError("Historical output directory must not be mounted")
    if theano.config.device != "cpu" or theano.config.floatX != "float32":
        raise RuntimeError("Expected Theano CPU float32 mode")

    random.seed(SEED)
    np.random.seed(SEED)
    print(
        "ENV python_seed={} numpy_seed={} PYTHONHASHSEED={} Theano={} NumPy={} "
        "device={} floatX={}".format(
            SEED, SEED, os.environ.get("PYTHONHASHSEED"), theano.__version__,
            np.__version__, theano.config.device, theano.config.floatX
        )
    )
    x_l, y_l, x_u, x_te, y_te, x_cal, y_cal = load_author_baseline_data("/data")
    if x_l.shape != (3104, 7) or y_l.shape != (3104, 2) or x_u.shape != (30997, 7):
        raise AssertionError("Selected source-baseline data sizes changed")

    args = make_args()
    alpha = 1.1 * (N_UNSUP + 2 * N_SUP_MINORITY) / float(2 * N_SUP_MINORITY)
    model = SS_M1(7, NUM_CLASSES, alpha, args)
    flat = parameter_layout(model)
    print(
        "MODEL input_dim=7 classes=2 zdim=10 alpha={:.17g} "
        "parameter_arrays={} pretrained=False".format(alpha, len(flat))
    )

    shared_x_l, shared_y_l = shared_dataset((x_l, y_l))
    shared_x_u = shared_dataset((x_u,))
    train_labeled = model.labeled_training(shared_x_l, shared_y_l, BATCH_SIZE)
    train_unlabeled = model.unlabeled_training(shared_x_u, BATCH_SIZE)
    grad_labeled = compile_gradient_function(
        model, gradients_in_flat_order(model, model.gparams_l, flat),
        "labeled", shared_x_l, shared_y_l
    )
    grad_unlabeled = compile_gradient_function(
        model, gradients_in_flat_order(model, model.gparams_u, flat),
        "unlabeled", shared_x_u
    )
    labeled_loss_probe = compile_loss_probe(
        model, model.labeled_cost, "labeled", shared_x_l, shared_y_l
    )
    unlabeled_loss_probe = compile_loss_probe(
        model, model.unlabeled_cost, "unlabeled", shared_x_u
    )
    print("COMPILE labeled_training=True unlabeled_training=True")

    rng = np.random.RandomState(SEED + 1)
    eps_l = rng.randn(BATCH_SIZE, args.zdim).astype(floatX)
    eps_u = rng.randn(BATCH_SIZE, args.zdim).astype(floatX)
    training_mode = np.asarray(1.0, dtype=floatX)
    inference_mode = np.asarray(0.0, dtype=floatX)

    gradients_l = grad_labeled(0, eps_l, training_mode)
    report_gradients("labeled", flat, gradients_l)
    labeled_loss_before = float(labeled_loss_probe(0, eps_l, inference_mode))
    before_l = snapshot(flat)
    labeled_output = train_labeled(0, eps_l, training_mode)
    after_l = snapshot(flat)
    labeled_loss_after = float(labeled_loss_probe(0, eps_l, inference_mode))
    labeled_loss = float(np.asarray(labeled_output[0]))
    if not all(np.isfinite(value) for value in [
        labeled_loss, labeled_loss_before, labeled_loss_after
    ]):
        raise ValueError("Labeled loss is NaN/Inf")
    report_update("labeled", flat, before_l, after_l)
    print(
        "LOSS path=labeled total={:.17g} classifier={:.17g} "
        "reconstruction_mean={:.17g} probe_before={:.17g} probe_after={:.17g} "
        "probe_delta={:.17g} finite=True".format(
            labeled_loss,
            float(np.asarray(labeled_output[1])),
            float(np.mean(np.asarray(labeled_output[2]))),
            labeled_loss_before,
            labeled_loss_after,
            labeled_loss_after - labeled_loss_before,
        )
    )

    gradients_u = grad_unlabeled(0, eps_u, training_mode)
    report_gradients("unlabeled", flat, gradients_u)
    unlabeled_loss_before = float(unlabeled_loss_probe(0, eps_u, inference_mode))
    before_u = snapshot(flat)
    unlabeled_output = train_unlabeled(0, eps_u, training_mode)
    after_u = snapshot(flat)
    unlabeled_loss_after = float(unlabeled_loss_probe(0, eps_u, inference_mode))
    unlabeled_loss = float(np.asarray(unlabeled_output))
    if not all(np.isfinite(value) for value in [
        unlabeled_loss, unlabeled_loss_before, unlabeled_loss_after
    ]):
        raise ValueError("Unlabeled loss is NaN/Inf")
    report_update("unlabeled", flat, before_u, after_u)
    print(
        "LOSS path=unlabeled total={:.17g} probe_before={:.17g} "
        "probe_after={:.17g} probe_delta={:.17g} finite=True".format(
            unlabeled_loss, unlabeled_loss_before, unlabeled_loss_after,
            unlabeled_loss_after - unlabeled_loss_before
        )
    )

    batch_pi = np.asarray(model.draw_pi(x_l[:BATCH_SIZE], np.asarray(0.0, dtype=floatX)))
    batch_pred = np.asarray(model.predict_label(x_l[:BATCH_SIZE], np.asarray(0.0, dtype=floatX)))
    if batch_pi.shape != (BATCH_SIZE, 2) or batch_pred.shape != (BATCH_SIZE,):
        raise AssertionError("Classifier output shape mismatch")
    if not np.isfinite(batch_pi).all():
        raise ValueError("Classifier output contains NaN/Inf")
    print(
        "FORWARD input_shape={} pi_shape={} prediction_shape={} dtype={} finite=True".format(
            x_l[:BATCH_SIZE].shape, batch_pi.shape, batch_pred.shape, batch_pi.dtype
        )
    )

    evaluation_success = False
    try:
        performance = model.get_performance(x_te, y_te, x_cal, y_cal)
        auc = float(performance["AUC"])
        scores = np.asarray(performance["scores"])
        if not np.isfinite(auc) or not np.isfinite(scores).all():
            raise ValueError("Evaluation produced NaN/Inf")
        evaluation_success = True
        print(
            "EVALUATION success=True method=original_get_performance calibrated=True "
            "test_rows={} calibration_rows={} score_shape={} AUC={:.17g}".format(
                x_te.shape[0], x_cal.shape[0], scores.shape, auc
            )
        )
    except Exception as exc:
        print(
            "EVALUATION success=False method=original_get_performance "
            "error_type={} error_message={!r}".format(type(exc).__name__, str(exc))
        )
        print("EVALUATION_TRACEBACK_BEGIN")
        print(traceback.format_exc())
        print("EVALUATION_TRACEBACK_END")

    print(
        "STAGE9_RESULT training_success=True evaluation_success={} epochs_run=0 "
        "labeled_updates=1 unlabeled_updates=1 output_files_written=0".format(
            evaluation_success
        )
    )


if __name__ == "__main__":
    main()

