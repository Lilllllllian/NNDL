"""End-to-end experiment runner that produces all materials needed by the
two-column report. Six experiment packs are supported and each pack can be
resumed individually:

    --pack main      Part A and Part B main comparison (MLP 30ep + CNN 25ep)
    --pack optim     Direction 1: 5 optimisation settings on CNN
    --pack reg       Direction 2: 6 regularisation settings on CNN
    --pack aug       Direction 3: 8 augmentation settings on CNN
    --pack robust    Direction 4: 9 perturbation scenarios + sigma sweep
    --pack error     Direction 5: error / calibration / filter / t-SNE / saliency
    --pack all       run main, optim, reg, aug, robust, error in this order

Each pack writes ``results/full/<pack>/summary.json`` and reusable models go
into ``results/full/models/<name>/best_model.pickle``. The master aggregator
then glues everything together for the LaTeX report."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import mynn as nn
from study_utils import (apply_perturbation_set, confusion_matrix, count_params,
                         dump_json, evaluate_acc, expected_calibration_error,
                         feature_extractor_cnn, gaussian_noise, load_mnist,
                         make_augmenter, make_cnn, make_mlp, per_class_metrics,
                         plot_bar_comparison, plot_confusion_matrix, plot_curves,
                         plot_filters, plot_line_with_markers,
                         plot_misclassified, plot_mlp_weights, plot_reliability,
                         plot_saliency, predict_logits, saliency_cnn,
                         softmax_np, split_train_valid, top_confusion_pairs)


# -----------------------------------------------------------------------------
# Configuration block
# -----------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'dataset', 'MNIST')
OUT_DIR = os.path.join(ROOT, 'results', 'full')
MODEL_DIR = os.path.join(OUT_DIR, 'models')
FIG_DIR = os.path.join(OUT_DIR, 'figs')
SEED = 309
BATCH = 128
EVAL_BATCH = 256
WD = 1e-4
LR_BASE = 0.05


# -----------------------------------------------------------------------------
# Generic training routine with optional on-the-fly augmentation
# -----------------------------------------------------------------------------

def _make_optimizer(name: str, lr: float, model, momentum: float = 0.9):
    if name == 'sgd':
        return nn.optimizer.SGD(init_lr=lr, model=model)
    if name == 'momentum':
        return nn.optimizer.MomentGD(init_lr=lr, model=model, mu=momentum)
    raise ValueError(name)


def _make_scheduler(name: str, optimizer, milestones, gamma):
    if name in (None, 'none'):
        return None
    if name == 'multistep':
        return nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
    if name == 'exponential':
        return nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    raise ValueError(name)


def _save_history(history, path):
    """Serialise the training history. Each value can either be a list of
    floats (e.g. ``train_loss``) or a single scalar (e.g. ``best_valid_acc``)."""
    out = {}
    for k, v in history.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            out[k] = [float(x) for x in v]
        else:
            out[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
    dump_json(out, path)


def _train_one(name: str, model, train_set, valid_set, epochs: int, lr: float,
               optimizer_name: str = 'sgd', scheduler_name: Optional[str] = None,
               milestones: Sequence[int] = (), gamma: float = 0.5,
               augment_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               early_stop_patience: Optional[int] = None,
               momentum: float = 0.9):
    """Train ``model`` for ``epochs`` epochs and return the recorded history.

    Best checkpoint (by validation accuracy) is saved under
    ``MODEL_DIR/<name>/best_model.pickle`` and the history JSON under
    ``OUT_DIR/history/<name>.json``. If the checkpoint already exists the
    function loads it and skips training, enabling resume."""
    save_dir = os.path.join(MODEL_DIR, name)
    history_path = os.path.join(OUT_DIR, 'history', f'{name}.json')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    best_ckpt = os.path.join(save_dir, 'best_model.pickle')
    if os.path.exists(best_ckpt) and os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        model.load_model(best_ckpt)
        return history, best_ckpt

    optimizer = _make_optimizer(optimizer_name, lr, model, momentum=momentum)
    scheduler = _make_scheduler(scheduler_name, optimizer, milestones, gamma)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    train_X, train_y = train_set
    valid_X, valid_y = valid_set
    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': [],
               'lr': [], 'time': []}
    best_score = 0.0
    epochs_since_improve = 0
    rng = np.random.default_rng(SEED + 1)

    n_iters_per_epoch = max(1, train_X.shape[0] // BATCH)
    eval_iters = max(50, n_iters_per_epoch // 4)
    log_iters = eval_iters
    global_iter = 0
    t_start = time.time()
    for epoch in range(epochs):
        idx = rng.permutation(train_X.shape[0])
        Xs = train_X[idx]
        ys = train_y[idx]
        for it in range(n_iters_per_epoch + 1):
            xb = Xs[it * BATCH:(it + 1) * BATCH]
            yb = ys[it * BATCH:(it + 1) * BATCH]
            if xb.shape[0] == 0:
                continue
            if augment_fn is not None:
                xb = augment_fn(xb)
            if hasattr(model, 'set_training'):
                model.set_training(True)
            logits = model(xb)
            loss_v = loss_fn(logits, yb)
            preds = np.argmax(logits, axis=1)
            acc = float(np.mean(preds == yb))
            history['train_loss'].append(float(loss_v))
            history['train_acc'].append(acc)
            loss_fn.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            history['lr'].append(float(optimizer.init_lr))
            if global_iter % eval_iters == 0:
                v_acc, v_logits, _ = evaluate_acc(model, valid_X, valid_y, EVAL_BATCH)
                v_loss = float(-np.mean(np.log(np.clip(softmax_np(v_logits)[np.arange(valid_y.shape[0]), valid_y], 1e-12, 1.0))))
                history['valid_loss'].append(v_loss)
                history['valid_acc'].append(v_acc)
                history['time'].append(time.time() - t_start)
                if v_acc > best_score:
                    model.save_model(best_ckpt)
                    best_score = v_acc
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                if global_iter % (log_iters * 4) == 0:
                    print(f'[{name}] epoch {epoch + 1}/{epochs} iter {global_iter} '
                          f'lr={optimizer.init_lr:.4f} train_loss={loss_v:.4f} '
                          f'val_acc={v_acc:.4f} best={best_score:.4f}')
            global_iter += 1
        if early_stop_patience is not None:
            evals_per_epoch = max(1, n_iters_per_epoch // eval_iters)
            if epochs_since_improve >= early_stop_patience * evals_per_epoch:
                history['early_stopped_epoch'] = int(epoch + 1)
                print(f'[{name}] early stopping at epoch {epoch + 1}')
                break
    history['best_valid_acc'] = float(best_score)
    history['total_time_sec'] = float(time.time() - t_start)
    _save_history(history, history_path)
    if os.path.exists(best_ckpt):
        model.load_model(best_ckpt)
    return history, best_ckpt


# -----------------------------------------------------------------------------
# Pack 1: Main comparison
# -----------------------------------------------------------------------------

def pack_main(train_set, valid_set, test_set, epochs_mlp: int, epochs_cnn: int):
    summary = {}
    figs = {}

    # MLP baseline
    mlp = make_mlp(weight_decay=WD, hidden=600)
    h_mlp, _ = _train_one('main_mlp', mlp, train_set, valid_set,
                          epochs=epochs_mlp, lr=LR_BASE, optimizer_name='sgd')
    test_acc, logits, preds = evaluate_acc(mlp, test_set[0], test_set[1], EVAL_BATCH)
    cm = confusion_matrix(preds, test_set[1])
    pcs_p, pcs_r, pcs_f = per_class_metrics(cm)
    probs = softmax_np(logits)
    ece, bins = expected_calibration_error(probs, test_set[1])
    plot_curves(h_mlp['train_loss'], h_mlp['valid_loss'], h_mlp['train_acc'], h_mlp['valid_acc'],
                f'MLP baseline (test acc={test_acc:.4f})',
                os.path.join(FIG_DIR, 'main', 'mlp_curves.png'))
    plot_confusion_matrix(cm, os.path.join(FIG_DIR, 'main', 'mlp_cm_counts.png'),
                          'MLP confusion matrix', normalized=False)
    plot_confusion_matrix(cm, os.path.join(FIG_DIR, 'main', 'mlp_cm_normalized.png'),
                          'MLP normalised confusion matrix', normalized=True)
    summary['mlp'] = {
        'epochs': epochs_mlp, 'best_valid_acc': float(h_mlp['best_valid_acc']),
        'test_acc': float(test_acc),
        'per_class_recall': [float(r) for r in pcs_r],
        'per_class_f1': [float(f) for f in pcs_f],
        'ece': float(ece),
        'top_confusions': [(int(a), int(b), int(c)) for a, b, c in top_confusion_pairs(cm, 5)],
        'params': count_params(mlp),
        'total_time_sec': float(h_mlp['total_time_sec']),
    }

    # CNN baseline
    cnn = make_cnn(weight_decay=WD, dropout_rate=0.0)
    h_cnn, _ = _train_one('main_cnn', cnn, train_set, valid_set,
                          epochs=epochs_cnn, lr=LR_BASE, optimizer_name='sgd')
    test_acc_c, logits_c, preds_c = evaluate_acc(cnn, test_set[0], test_set[1], EVAL_BATCH)
    cm_c = confusion_matrix(preds_c, test_set[1])
    pcs_p_c, pcs_r_c, pcs_f_c = per_class_metrics(cm_c)
    probs_c = softmax_np(logits_c)
    ece_c, bins_c = expected_calibration_error(probs_c, test_set[1])
    plot_curves(h_cnn['train_loss'], h_cnn['valid_loss'], h_cnn['train_acc'], h_cnn['valid_acc'],
                f'CNN baseline (test acc={test_acc_c:.4f})',
                os.path.join(FIG_DIR, 'main', 'cnn_curves.png'))
    plot_confusion_matrix(cm_c, os.path.join(FIG_DIR, 'main', 'cnn_cm_counts.png'),
                          'CNN confusion matrix', normalized=False)
    plot_confusion_matrix(cm_c, os.path.join(FIG_DIR, 'main', 'cnn_cm_normalized.png'),
                          'CNN normalised confusion matrix', normalized=True)
    plot_reliability([bins, bins_c], [ece, ece_c], ['MLP', 'CNN'],
                     os.path.join(FIG_DIR, 'main', 'reliability_main.png'),
                     'Reliability diagram (test set)')
    summary['cnn'] = {
        'epochs': epochs_cnn, 'best_valid_acc': float(h_cnn['best_valid_acc']),
        'test_acc': float(test_acc_c),
        'per_class_recall': [float(r) for r in pcs_r_c],
        'per_class_f1': [float(f) for f in pcs_f_c],
        'ece': float(ece_c),
        'top_confusions': [(int(a), int(b), int(c)) for a, b, c in top_confusion_pairs(cm_c, 5)],
        'params': count_params(cnn),
        'total_time_sec': float(h_cnn['total_time_sec']),
    }

    # Combined per-class plot
    fig_path = os.path.join(FIG_DIR, 'main', 'per_class_acc.png')
    plot_line_with_markers(np.arange(10), {'MLP recall': pcs_r, 'CNN recall': pcs_r_c},
                           fig_path, 'Per-class recall on test set', 'Class', 'Recall')
    summary['gradient_check'] = {
        'note': 'See codes/gradient_check.py output for analytical-vs-finite gradient comparisons.'
    }
    dump_json(summary, os.path.join(OUT_DIR, 'main', 'summary.json'))
    return summary


# -----------------------------------------------------------------------------
# Pack 2: Optimisation
# -----------------------------------------------------------------------------

def pack_optim(train_set, valid_set, test_set, epochs: int, milestones):
    settings = [
        ('opt_sgd_005', 'sgd', 0.05, None),
        ('opt_sgd_001', 'sgd', 0.01, None),
        ('opt_sgd_010', 'sgd', 0.10, None),
        ('opt_momentum', 'momentum', 0.05, None),
        ('opt_multistep', 'sgd', 0.05, 'multistep'),
    ]
    rows = []
    for name, opt_name, lr, sched in settings:
        cnn = make_cnn(weight_decay=WD, dropout_rate=0.0)
        h, _ = _train_one(name, cnn, train_set, valid_set,
                          epochs=epochs, lr=lr, optimizer_name=opt_name,
                          scheduler_name=sched, milestones=milestones, gamma=0.5)
        test_acc, _, _ = evaluate_acc(cnn, test_set[0], test_set[1], EVAL_BATCH)
        rows.append({
            'name': name,
            'optimizer': opt_name,
            'lr': lr,
            'scheduler': sched,
            'best_valid_acc': float(h['best_valid_acc']),
            'test_acc': float(test_acc),
            'history': name,
        })
    # plot the bar comparison and curves
    labels = ['SGD-0.05', 'SGD-0.01', 'SGD-0.10', 'Momentum', 'MultiStep']
    test_accs = [r['test_acc'] for r in rows]
    plot_bar_comparison(labels, test_accs, os.path.join(FIG_DIR, 'optim', 'optim_test_acc.png'),
                        'Optimisation: test accuracy', 'Test accuracy',
                        baseline=test_accs[0], color='#4C78A8',
                        ylim=(min(test_accs) - 0.01, max(test_accs) + 0.01))

    # combined valid-acc curves
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2']
    for r, label, color in zip(rows, labels, colors):
        with open(os.path.join(OUT_DIR, 'history', f"{r['name']}.json"), 'r', encoding='utf-8') as f:
            h = json.load(f)
        xs = np.linspace(0, max(len(h['train_loss']) - 1, 1), len(h['valid_acc']))
        ax.plot(xs, h['valid_acc'], '-', lw=1.4, color=color, label=label)
    ax.set_xlabel('iteration')
    ax.set_ylabel('valid accuracy')
    ax.set_title('Validation accuracy under different optimisers')
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'optim', 'optim_valid_curves.png'))
    plt.close(fig)

    summary = {'rows': rows, 'labels': labels}
    dump_json(summary, os.path.join(OUT_DIR, 'optim', 'summary.json'))
    return summary


# -----------------------------------------------------------------------------
# Pack 3: Regularisation
# -----------------------------------------------------------------------------

def pack_reg(train_set, valid_set, test_set, epochs: int):
    rows = []

    def _run(name, weight_decay, dropout_rate, early_stop):
        cnn = make_cnn(weight_decay=weight_decay, dropout_rate=dropout_rate)
        h, _ = _train_one(name, cnn, train_set, valid_set,
                          epochs=epochs, lr=LR_BASE, optimizer_name='sgd',
                          early_stop_patience=early_stop)
        test_acc, _, _ = evaluate_acc(cnn, test_set[0], test_set[1], EVAL_BATCH)
        rows.append({
            'name': name,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
            'early_stop': early_stop,
            'best_valid_acc': float(h['best_valid_acc']),
            'test_acc': float(test_acc),
            'stop_epoch': int(h.get('early_stopped_epoch', 0)),
        })

    _run('reg_no', 0.0, 0.0, None)
    _run('reg_l2_1e4', 1e-4, 0.0, None)
    _run('reg_l2_5e4', 5e-4, 0.0, None)
    _run('reg_dropout_02', 1e-4, 0.2, None)
    _run('reg_dropout_03', 1e-4, 0.3, None)
    _run('reg_earlystop', 1e-4, 0.0, 4)

    labels = ['No-Reg', 'L2-1e-4', 'L2-5e-4', 'Dropout-0.2', 'Dropout-0.3', 'EarlyStop']
    test_accs = [r['test_acc'] for r in rows]
    plot_bar_comparison(labels, test_accs, os.path.join(FIG_DIR, 'reg', 'reg_test_acc.png'),
                        'Regularisation: test accuracy', 'Test accuracy',
                        baseline=test_accs[0], color='#54A24B',
                        ylim=(min(test_accs) - 0.005, max(test_accs) + 0.005))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2', '#B279A2']
    for r, label, color in zip(rows, labels, colors):
        with open(os.path.join(OUT_DIR, 'history', f"{r['name']}.json"), 'r', encoding='utf-8') as f:
            h = json.load(f)
        xs = np.linspace(0, max(len(h['train_loss']) - 1, 1), len(h['valid_acc']))
        ax.plot(xs, h['valid_acc'], '-', lw=1.4, color=color, label=label)
    ax.set_xlabel('iteration')
    ax.set_ylabel('valid accuracy')
    ax.set_title('Validation accuracy under different regularisers')
    ax.set_ylim(0.92, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'reg', 'reg_valid_curves.png'))
    plt.close(fig)

    summary = {'rows': rows, 'labels': labels}
    dump_json(summary, os.path.join(OUT_DIR, 'reg', 'summary.json'))
    return summary


# -----------------------------------------------------------------------------
# Pack 4: Data augmentation
# -----------------------------------------------------------------------------

AUG_SETTINGS = [
    ('aug_clean', False, False, False),
    ('aug_R',     True,  False, False),
    ('aug_T',     False, True,  False),
    ('aug_S',     False, False, True),
    ('aug_RT',    True,  True,  False),
    ('aug_RS',    True,  False, True),
    ('aug_TS',    False, True,  True),
    ('aug_RTS',   True,  True,  True),
]


def pack_aug(train_set, valid_set, test_set, epochs: int):
    rows = []
    for name, use_r, use_t, use_s in AUG_SETTINGS:
        cnn = make_cnn(weight_decay=WD, dropout_rate=0.0)
        aug = make_augmenter(use_r, use_t, use_s, seed=SEED + 7) if (use_r or use_t or use_s) else None
        h, _ = _train_one(name, cnn, train_set, valid_set,
                          epochs=epochs, lr=LR_BASE, optimizer_name='sgd',
                          augment_fn=aug)
        test_acc, _, _ = evaluate_acc(cnn, test_set[0], test_set[1], EVAL_BATCH)
        rows.append({
            'name': name,
            'use_rotation': use_r, 'use_translation': use_t, 'use_scaling': use_s,
            'best_valid_acc': float(h['best_valid_acc']),
            'test_acc': float(test_acc),
        })
    labels = ['Clean', 'R', 'T', 'S', 'RT', 'RS', 'TS', 'RTS']
    test_accs = [r['test_acc'] for r in rows]
    clean_acc = test_accs[0]
    plot_bar_comparison(labels, test_accs, os.path.join(FIG_DIR, 'aug', 'aug_test_acc.png'),
                        'Augmentation ablation: test accuracy', 'Test accuracy',
                        baseline=clean_acc, color='#E45756',
                        ylim=(min(test_accs) - 0.005, max(test_accs) + 0.005))

    summary = {'rows': rows, 'labels': labels}
    dump_json(summary, os.path.join(OUT_DIR, 'aug', 'summary.json'))
    return summary


# -----------------------------------------------------------------------------
# Pack 5: Robustness (no training, evaluate trained models)
# -----------------------------------------------------------------------------

def pack_robust(test_set):
    """Reuse the Clean (aug_clean) and RTS (aug_RTS) checkpoints from pack_aug."""
    rts_path = os.path.join(MODEL_DIR, 'aug_RTS', 'best_model.pickle')
    clean_path = os.path.join(MODEL_DIR, 'aug_clean', 'best_model.pickle')
    if not (os.path.exists(rts_path) and os.path.exists(clean_path)):
        raise RuntimeError('Pack robust requires aug_clean and aug_RTS checkpoints from pack_aug.')

    clean_cnn = make_cnn(weight_decay=WD, dropout_rate=0.0)
    clean_cnn.load_model(clean_path)
    rts_cnn = make_cnn(weight_decay=WD, dropout_rate=0.0)
    rts_cnn.load_model(rts_path)

    scenarios = ['clean', 'rot+10', 'rot-10', 'shift_left2', 'shift_right2',
                 'shift_up2', 'shift_down2', 'scale_up_1.08', 'scale_down_0.92']
    results = []
    test_X, test_y = test_set
    for scen in scenarios:
        perturbed = apply_perturbation_set(test_X, scen)
        clean_acc, _, _ = evaluate_acc(clean_cnn, perturbed, test_y, EVAL_BATCH)
        rts_acc, _, _ = evaluate_acc(rts_cnn, perturbed, test_y, EVAL_BATCH)
        results.append({'scenario': scen, 'clean_cnn_acc': float(clean_acc),
                        'rts_cnn_acc': float(rts_acc),
                        'gain': float(rts_acc - clean_acc)})

    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    noise_results = []
    for s in sigmas:
        if s == 0.0:
            noisy = test_X
        else:
            noisy = gaussian_noise(test_X, s, seed=SEED + 11)
        c_acc, _, _ = evaluate_acc(clean_cnn, noisy, test_y, EVAL_BATCH)
        r_acc, _, _ = evaluate_acc(rts_cnn, noisy, test_y, EVAL_BATCH)
        noise_results.append({'sigma': float(s), 'clean_cnn_acc': float(c_acc),
                              'rts_cnn_acc': float(r_acc)})

    # plots
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xs = np.arange(len(scenarios))
    width = 0.4
    clean_vals = [r['clean_cnn_acc'] for r in results]
    rts_vals = [r['rts_cnn_acc'] for r in results]
    ax.bar(xs - width / 2, clean_vals, width, color='#4C78A8', label='Clean CNN')
    ax.bar(xs + width / 2, rts_vals, width, color='#F58518', label='Affine-All CNN')
    ax.set_xticks(xs)
    ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Test accuracy')
    ax.set_title('Robustness under fixed perturbations')
    ax.set_ylim(0.7, 1.005)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, 'robust', 'robust_perturb.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    s_xs = [r['sigma'] for r in noise_results]
    ax.plot(s_xs, [r['clean_cnn_acc'] for r in noise_results], '-o', color='#4C78A8', label='Clean CNN')
    ax.plot(s_xs, [r['rts_cnn_acc'] for r in noise_results], '-s', color='#F58518', label='Affine-All CNN')
    ax.set_xlabel(r'Gaussian noise $\sigma$')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Accuracy vs additive Gaussian noise')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'robust', 'robust_noise_curve.png'))
    plt.close(fig)

    summary = {'scenarios': results, 'noise_sweep': noise_results}
    dump_json(summary, os.path.join(OUT_DIR, 'robust', 'summary.json'))
    return summary


# -----------------------------------------------------------------------------
# Pack 6: Error analysis on the best model
# -----------------------------------------------------------------------------

def _select_best_model_path():
    """Return (name, path) of the best CNN by test accuracy among aug_* models.

    Falls back to aug_T if results are unavailable."""
    candidate = 'aug_T'
    aug_summary = os.path.join(OUT_DIR, 'aug', 'summary.json')
    if os.path.exists(aug_summary):
        with open(aug_summary, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rows = sorted(data['rows'], key=lambda r: -r['test_acc'])
        candidate = rows[0]['name']
    return candidate, os.path.join(MODEL_DIR, candidate, 'best_model.pickle')


def pack_error(train_set, valid_set, test_set):
    name, path = _select_best_model_path()
    if not os.path.exists(path):
        raise RuntimeError(f'Best model checkpoint not found at {path}')
    cnn = make_cnn(weight_decay=WD, dropout_rate=0.0)
    cnn.load_model(path)
    test_X, test_y = test_set
    test_acc, logits, preds = evaluate_acc(cnn, test_X, test_y, EVAL_BATCH)
    cm = confusion_matrix(preds, test_y)
    pcs_p, pcs_r, pcs_f = per_class_metrics(cm)
    probs = softmax_np(logits)
    ece, bins = expected_calibration_error(probs, test_y)
    top_pairs = top_confusion_pairs(cm, 5)

    plot_confusion_matrix(cm, os.path.join(FIG_DIR, 'error', 'best_cm_counts.png'),
                          f'Best model ({name}) confusion matrix', normalized=False)
    plot_confusion_matrix(cm, os.path.join(FIG_DIR, 'error', 'best_cm_normalized.png'),
                          f'Best model ({name}) normalised confusion', normalized=True)
    plot_misclassified(test_X, test_y, preds, probs,
                       os.path.join(FIG_DIR, 'error', 'best_high_conf_errors.png'),
                       f'High-confidence errors of {name}')
    # filter visualisation
    conv_layers = [l for l in cnn.layers if hasattr(l, 'in_channels')]
    if conv_layers:
        plot_filters(conv_layers[0].params['W'],
                     os.path.join(FIG_DIR, 'error', 'best_conv1_filters.png'),
                     f'First-layer convolution filters ({name})')
    if len(conv_layers) > 1:
        # show the average over input channels for layer 2
        w = conv_layers[1].params['W']  # (16, 8, 3, 3)
        avg = w.mean(axis=1, keepdims=True)
        plot_filters(avg, os.path.join(FIG_DIR, 'error', 'best_conv2_filters.png'),
                     f'Second-layer filters averaged across input channels ({name})')
    plot_reliability([bins], [ece], [name],
                     os.path.join(FIG_DIR, 'error', 'best_reliability.png'),
                     f'Reliability diagram of {name}')

    # MLP weight visualisation (use the main MLP if available)
    mlp_ckpt = os.path.join(MODEL_DIR, 'main_mlp', 'best_model.pickle')
    if os.path.exists(mlp_ckpt):
        mlp = make_mlp(weight_decay=WD, hidden=600)
        mlp.load_model(mlp_ckpt)
        first = next((l for l in mlp.layers if hasattr(l, 'params') and l.params.get('W') is not None
                      and l.params['W'].shape[0] == 784), None)
        if first is not None:
            plot_mlp_weights(first.params['W'], os.path.join(FIG_DIR, 'error', 'mlp_first_layer_weights.png'),
                             'First-layer MLP weights (per neuron)', max_neurons=64)

    # t-SNE / PCA of penultimate features
    feats = feature_extractor_cnn(cnn, test_X[:4000], EVAL_BATCH)
    labels_sub = test_y[:4000]
    feats_centered = feats - feats.mean(axis=0, keepdims=True)
    cov = feats_centered.T @ feats_centered / max(1, feats_centered.shape[0] - 1)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    order = np.argsort(-eig_vals)
    pca_components = eig_vecs[:, order[:2]]
    proj = feats_centered @ pca_components

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = plt.get_cmap('tab10')
    for c in range(10):
        m = labels_sub == c
        ax.scatter(proj[m, 0], proj[m, 1], s=3, color=cmap(c), label=str(c), alpha=0.6)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'PCA of penultimate features ({name}, 4000 test samples)')
    ax.legend(loc='upper right', ncol=2, fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'error', 'best_features_pca.png'))
    plt.close(fig)

    saliency = saliency_cnn(cnn, test_X, test_y, max_per_class=1)
    plot_saliency(test_X, test_y, saliency,
                  os.path.join(FIG_DIR, 'error', 'best_saliency.png'),
                  f'Input saliency |dL/dx| of {name}')

    summary = {
        'best_model': name,
        'test_acc': float(test_acc),
        'ece': float(ece),
        'per_class_recall': [float(r) for r in pcs_r],
        'per_class_precision': [float(p) for p in pcs_p],
        'per_class_f1': [float(f) for f in pcs_f],
        'top_confusions': [(int(a), int(b), int(c)) for a, b, c in top_pairs],
        'easiest_classes': sorted(range(10), key=lambda c: -pcs_r[c])[:3],
        'hardest_classes': sorted(range(10), key=lambda c: pcs_r[c])[:3],
        'reliability_bins': bins,
    }
    dump_json(summary, os.path.join(OUT_DIR, 'error', 'summary.json'))
    return summary


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', choices=['main', 'optim', 'reg', 'aug', 'robust', 'error', 'all'], required=True)
    parser.add_argument('--epochs-mlp', type=int, default=30)
    parser.add_argument('--epochs-cnn', type=int, default=20)
    parser.add_argument('--epochs-direction', type=int, default=10)
    parser.add_argument('--train-limit', type=int, default=None,
                        help='Optional cap on training samples (for tractable direction studies).')
    parser.add_argument('--valid-size', type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    np.random.seed(SEED)
    train_images, train_labels, test_images, test_labels = load_mnist(DATA_DIR)
    train_X, train_y, valid_X, valid_y = split_train_valid(train_images, train_labels, args.valid_size, SEED)
    if args.train_limit is not None:
        train_X = train_X[:args.train_limit]
        train_y = train_y[:args.train_limit]
    train_set = (train_X, train_y)
    valid_set = (valid_X, valid_y)
    test_set = (test_images, test_labels)
    print(f'data loaded: train={train_X.shape}, valid={valid_X.shape}, test={test_images.shape}')

    direction_milestones = [int(args.epochs_direction * (train_X.shape[0] // BATCH) * f)
                            for f in (0.5, 0.75)]

    if args.pack in ('main', 'all'):
        pack_main(train_set, valid_set, test_set,
                  epochs_mlp=args.epochs_mlp, epochs_cnn=args.epochs_cnn)
    if args.pack in ('optim', 'all'):
        pack_optim(train_set, valid_set, test_set,
                   epochs=args.epochs_direction, milestones=direction_milestones)
    if args.pack in ('reg', 'all'):
        pack_reg(train_set, valid_set, test_set, epochs=args.epochs_direction)
    if args.pack in ('aug', 'all'):
        pack_aug(train_set, valid_set, test_set, epochs=args.epochs_direction)
    if args.pack in ('robust', 'all'):
        pack_robust(test_set)
    if args.pack in ('error', 'all'):
        pack_error(train_set, valid_set, test_set)


if __name__ == '__main__':
    main()
