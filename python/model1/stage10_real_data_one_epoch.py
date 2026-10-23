from __future__ import print_function

import json
import os
import random
import time

import numpy as np
import theano

from SS_M1 import SS_M1
from utils import floatX, shared_dataset
from stage9_real_data_smoke import (
    BATCH_SIZE,
    N_SUP_MINORITY,
    N_UNSUP,
    NUM_CLASSES,
    SEED,
    compile_gradient_function,
    gradients_in_flat_order,
    load_author_baseline_data,
    make_args,
    parameter_layout,
    report_gradients,
    report_update,
    snapshot,
)


EXPECTED_LABELED_UPDATES = 31
EXPECTED_UNLABELED_UPDATES = 309
EXPECTED_TOTAL_UPDATES = 340
RESULT_PATH = "/results/stage10-summary.txt"


def build_schedule(n_labeled, n_unlabeled):
    labeled_batches = n_labeled // BATCH_SIZE
    unlabeled_batches = n_unlabeled // BATCH_SIZE
    periodic_interval = int((n_labeled + n_unlabeled) / (1.0 * n_labeled))
    batches_per_epoch = labeled_batches + unlabeled_batches
    schedule = []
    labeled_index = 0
    unlabeled_index = 0

    for iteration in range(batches_per_epoch):
        is_labeled = (
            iteration % periodic_interval == 1
            and labeled_index < labeled_batches
        )
        if is_labeled:
            schedule.append(("labeled", labeled_index))
            labeled_index += 1
        else:
            schedule.append(("unlabeled", unlabeled_index))
            unlabeled_index += 1

    observed = (labeled_index, unlabeled_index, len(schedule))
    expected = (
        EXPECTED_LABELED_UPDATES,
        EXPECTED_UNLABELED_UPDATES,
        EXPECTED_TOTAL_UPDATES,
    )
    if observed != expected:
        raise RuntimeError(
            "Author schedule mismatch: observed={} expected={}".format(
                observed, expected
            )
        )
    if schedule[-1] != ("unlabeled", EXPECTED_UNLABELED_UPDATES - 1):
        raise RuntimeError("Unexpected final minibatch in author schedule")

    print(
        "SCHEDULE_CHECK labeled_batches={} unlabeled_batches={} interval={} "
        "total={} actual_labeled={} actual_unlabeled={} status=PASS".format(
            labeled_batches,
            unlabeled_batches,
            periodic_interval,
            batches_per_epoch,
            labeled_index,
            unlabeled_index,
        )
    )
    return schedule, periodic_interval


def assert_parameters_finite(flat, iteration):
    for group, name, parameter in flat:
        value = parameter.get_value(borrow=True)
        if not np.isfinite(value).all():
            raise ValueError(
                "Parameter contains NaN/Inf after iteration {}: {}.{}".format(
                    iteration, group, name
                )
            )


def finite_scalar(value, name, iteration):
    result = float(np.asarray(value))
    if not np.isfinite(result):
        raise ValueError(
            "{} is NaN/Inf at iteration {}".format(name, iteration)
        )
    return result


def summarize(values):
    array = np.asarray(values, dtype="float64")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty or non-finite losses")
    return {
        "count": int(array.size),
        "first": float(array[0]),
        "last": float(array[-1]),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def write_summary(summary):
    if not os.path.isdir("/results"):
        raise RuntimeError("Dedicated results directory is not mounted")
    with open(RESULT_PATH, "w") as handle:
        json.dump(summary, handle, sort_keys=True, indent=2)
        handle.write("\n")


def main():
    if os.getuid() == 0:
        raise RuntimeError("Stage 10 must not run as root")
    if os.path.exists("/work/output") or os.path.exists("/work/model1/output"):
        raise RuntimeError("Historical output directory must not be mounted")
    if theano.config.device != "cpu" or theano.config.floatX != "float32":
        raise RuntimeError("Expected Theano CPU float32 mode")

    random.seed(SEED)
    np.random.seed(SEED)
    print(
        "ENV python_seed={} numpy_seed={} PYTHONHASHSEED={} Theano={} NumPy={} "
        "device={} floatX={}".format(
            SEED,
            SEED,
            os.environ.get("PYTHONHASHSEED"),
            theano.__version__,
            np.__version__,
            theano.config.device,
            theano.config.floatX,
        )
    )

    x_l, y_l, x_u, x_te, y_te, x_cal, y_cal = load_author_baseline_data("/data")
    if x_l.shape != (3104, 7) or y_l.shape != (3104, 2):
        raise AssertionError("Labeled source-baseline data contract changed")
    if x_u.shape != (30997, 7):
        raise AssertionError("Unlabeled source-baseline data contract changed")

    schedule, periodic_interval = build_schedule(x_l.shape[0], x_u.shape[0])

    args = make_args()
    expected_args = {
        "batch_size": 100,
        "lr_l": 0.0001,
        "lr_u": 0.0001,
        "zdim": 10,
        "dropout_rate": 0.01,
        "hl_activation": "softplus",
        "hdim_cls": [70],
        "hdim_enc": [10, 10],
        "hdim_dec": [10, 10],
        "hdim_gmm": [10],
    }
    for name, expected in expected_args.items():
        if getattr(args, name) != expected:
            raise AssertionError("Unexpected Route A setting: {}".format(name))

    alpha = 1.1 * (N_UNSUP + 2 * N_SUP_MINORITY) / float(2 * N_SUP_MINORITY)
    model = SS_M1(7, NUM_CLASSES, alpha, args)
    flat = parameter_layout(model)
    initial_parameters = snapshot(flat)
    print(
        "MODEL input_dim=7 classes=2 zdim=10 alpha={:.17g} "
        "parameter_arrays={} pretrained=False epochs_requested=1".format(
            alpha, len(flat)
        )
    )

    shared_x_l, shared_y_l = shared_dataset((x_l, y_l))
    shared_x_u = shared_dataset((x_u,))
    compile_start = time.time()
    train_labeled = model.labeled_training(shared_x_l, shared_y_l, BATCH_SIZE)
    train_unlabeled = model.unlabeled_training(shared_x_u, BATCH_SIZE)
    grad_labeled = compile_gradient_function(
        model,
        gradients_in_flat_order(model, model.gparams_l, flat),
        "stage10_labeled",
        shared_x_l,
        shared_y_l,
    )
    grad_unlabeled = compile_gradient_function(
        model,
        gradients_in_flat_order(model, model.gparams_u, flat),
        "stage10_unlabeled",
        shared_x_u,
    )
    compile_seconds = time.time() - compile_start
    print(
        "COMPILE labeled_training=True unlabeled_training=True "
        "gradient_monitors=True seconds={:.6f}".format(compile_seconds)
    )

    training_mode = np.asarray(1.0, dtype=floatX)
    monitor_rng = np.random.RandomState(SEED + 1)
    pre_eps_l = monitor_rng.randn(BATCH_SIZE, args.zdim).astype(floatX)
    pre_eps_u = monitor_rng.randn(BATCH_SIZE, args.zdim).astype(floatX)
    report_gradients(
        "labeled_pre_epoch", flat, grad_labeled(0, pre_eps_l, training_mode)
    )
    report_gradients(
        "unlabeled_pre_epoch", flat, grad_unlabeled(0, pre_eps_u, training_mode)
    )

    labeled_losses = []
    classifier_losses = []
    reconstruction_losses = []
    unlabeled_losses = []
    labeled_updates = 0
    unlabeled_updates = 0

    epoch = 0
    if epoch != 0:
        raise RuntimeError("Stage 10 may execute epoch index 0 only")
    model.optimizer_l.learning_decay(
        decay_rate=0.9, every_epoch=5, epoch=epoch, min_lr=0.0000001
    )
    model.optimizer_u.learning_decay(
        decay_rate=0.9, every_epoch=10, epoch=epoch, min_lr=0.0000001
    )
    model.alpha_annealing(
        decay_rate=0,
        every_epoch=1,
        epoch=epoch,
        max_epoch=100,
        min_alpha=0.0,
        max_alpha=alpha,
        mode="increase",
    )

    print(
        "TRAINING_START epoch=0 total_iterations={} interval={} "
        "expected_labeled={} expected_unlabeled={}".format(
            len(schedule),
            periodic_interval,
            EXPECTED_LABELED_UPDATES,
            EXPECTED_UNLABELED_UPDATES,
        )
    )
    training_start = time.time()
    for iteration, (kind, batch_index) in enumerate(schedule):
        eps = np.random.randn(BATCH_SIZE, args.zdim).astype(floatX)
        if kind == "labeled":
            output = train_labeled(batch_index, eps, training_mode)
            labeled_losses.append(
                finite_scalar(output[0], "labeled_total_loss", iteration)
            )
            classifier_losses.append(
                finite_scalar(output[1], "classifier_loss", iteration)
            )
            reconstruction = np.asarray(output[2])
            if not np.isfinite(reconstruction).all():
                raise ValueError(
                    "Reconstruction loss contains NaN/Inf at iteration {}".format(
                        iteration
                    )
                )
            reconstruction_losses.append(float(reconstruction.mean()))
            labeled_updates += 1
        else:
            output = train_unlabeled(batch_index, eps, training_mode)
            unlabeled_losses.append(
                finite_scalar(output, "unlabeled_total_loss", iteration)
            )
            unlabeled_updates += 1

        assert_parameters_finite(flat, iteration)
        if iteration == 0 or (iteration + 1) % 50 == 0 or iteration + 1 == len(schedule):
            print(
                "TRAINING_PROGRESS epoch=0 iteration={}/{} labeled_updates={} "
                "unlabeled_updates={} parameters_finite=True".format(
                    iteration + 1,
                    len(schedule),
                    labeled_updates,
                    unlabeled_updates,
                )
            )

    training_seconds = time.time() - training_start
    if (labeled_updates, unlabeled_updates) != (
        EXPECTED_LABELED_UPDATES,
        EXPECTED_UNLABELED_UPDATES,
    ):
        raise RuntimeError("Executed update counts differ from the author schedule")

    final_parameters = snapshot(flat)
    report_update("one_epoch", flat, initial_parameters, final_parameters)

    post_eps_l = monitor_rng.randn(BATCH_SIZE, args.zdim).astype(floatX)
    post_eps_u = monitor_rng.randn(BATCH_SIZE, args.zdim).astype(floatX)
    report_gradients(
        "labeled_post_epoch", flat, grad_labeled(0, post_eps_l, training_mode)
    )
    report_gradients(
        "unlabeled_post_epoch", flat, grad_unlabeled(0, post_eps_u, training_mode)
    )

    labeled_summary = summarize(labeled_losses)
    classifier_summary = summarize(classifier_losses)
    reconstruction_summary = summarize(reconstruction_losses)
    unlabeled_summary = summarize(unlabeled_losses)
    print(
        "TRAINING_RESULT epochs_completed=1 max_epoch_index=0 total_updates={} "
        "labeled_updates={} unlabeled_updates={} seconds={:.6f} "
        "losses_finite=True gradients_finite=True parameters_finite=True".format(
            labeled_updates + unlabeled_updates,
            labeled_updates,
            unlabeled_updates,
            training_seconds,
        )
    )
    print("LOSS_SUMMARY labeled={}".format(json.dumps(labeled_summary, sort_keys=True)))
    print("LOSS_SUMMARY classifier={}".format(json.dumps(classifier_summary, sort_keys=True)))
    print(
        "LOSS_SUMMARY reconstruction={}".format(
            json.dumps(reconstruction_summary, sort_keys=True)
        )
    )
    print(
        "LOSS_SUMMARY unlabeled={}".format(
            json.dumps(unlabeled_summary, sort_keys=True)
        )
    )

    print(
        "EVALUATION_START method=original_get_performance test_rows={} "
        "calibration_rows={}".format(x_te.shape[0], x_cal.shape[0])
    )
    evaluation_start = time.time()
    performance = model.get_performance(x_te, y_te, x_cal, y_cal)
    evaluation_seconds = time.time() - evaluation_start
    auc = float(performance["AUC"])
    scores = np.asarray(performance["scores"])
    if not np.isfinite(auc) or not np.isfinite(scores).all():
        raise ValueError("Evaluation produced NaN/Inf")
    print(
        "EVALUATION_RESULT success=True calibrated=True score_shape={} "
        "AUC={:.17g} seconds={:.6f}".format(
            scores.shape, auc, evaluation_seconds
        )
    )

    summary = {
        "auc": auc,
        "batch_size": BATCH_SIZE,
        "compile_seconds": compile_seconds,
        "epochs_completed": 1,
        "evaluation_seconds": evaluation_seconds,
        "labeled_loss": labeled_summary,
        "labeled_updates": labeled_updates,
        "classifier_loss": classifier_summary,
        "reconstruction_loss": reconstruction_summary,
        "max_epoch_index": 0,
        "parameter_arrays": len(flat),
        "seed": SEED,
        "training_seconds": training_seconds,
        "unlabeled_loss": unlabeled_summary,
        "unlabeled_updates": unlabeled_updates,
    }
    write_summary(summary)
    print(
        "STAGE10_RESULT success=True epochs_completed=1 max_epoch_index=0 "
        "labeled_updates={} unlabeled_updates={} evaluation_success=True "
        "output_path={}".format(
            labeled_updates, unlabeled_updates, RESULT_PATH
        )
    )


if __name__ == "__main__":
    main()

