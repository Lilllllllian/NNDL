"""Utilities shared by all experiment packs.

The module contains data loading, train/valid splitting, geometric augmentation
(rotation / translation / scaling) implemented via bilinear sampling, model
factories, evaluation helpers, and a few publication-quality plotting routines.
The augmentation routines deliberately use NumPy only so they remain compatible
with the from-scratch requirement of the project."""

from __future__ import annotations

import gzip
import json
import os
from struct import unpack
from typing import Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


# -----------------------------------------------------------------------------
# Data loading and splitting
# -----------------------------------------------------------------------------

def load_mnist(data_dir: str):
    paths = {
        'train_images': os.path.join(data_dir, 'train-images-idx3-ubyte.gz'),
        'train_labels': os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'),
        'test_images': os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'),
        'test_labels': os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'),
    }
    with gzip.open(paths['train_images'], 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(paths['train_labels'], 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)
    with gzip.open(paths['test_images'], 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(paths['test_labels'], 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)
    return (train_images.astype(np.float32) / 255.0, train_labels,
            test_images.astype(np.float32) / 255.0, test_labels)


def split_train_valid(images, labels, valid_size, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(images.shape[0])
    images = images[idx]
    labels = labels[idx]
    return images[valid_size:], labels[valid_size:], images[:valid_size], labels[:valid_size]


# -----------------------------------------------------------------------------
# Geometric augmentation (rotation / translation / scaling) via bilinear sampling
# -----------------------------------------------------------------------------

def _bilinear_sample(images_2d, src_y, src_x):
    """Bilinear sampling for a batch of 28x28 images.

    Parameters
    ----------
    images_2d : (N, 28, 28) array
    src_y, src_x : (28, 28) arrays of source coordinates (in pixel space, can be
        out of bounds; out-of-bounds samples become 0).
    """
    H = W = 28
    y0 = np.floor(src_y).astype(np.int32)
    x0 = np.floor(src_x).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = src_y - y0
    wx = src_x - x0

    def gather(yy, xx):
        valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        yy_c = np.clip(yy, 0, H - 1)
        xx_c = np.clip(xx, 0, W - 1)
        vals = images_2d[:, yy_c, xx_c]  # broadcasting on first axis
        vals = np.where(valid[None, :, :], vals, 0.0)
        return vals

    Ia = gather(y0, x0)
    Ib = gather(y0, x1)
    Ic = gather(y1, x0)
    Id = gather(y1, x1)
    out = (Ia * (1 - wy) * (1 - wx)
           + Ib * (1 - wy) * wx
           + Ic * wy * (1 - wx)
           + Id * wy * wx)
    return out.astype(images_2d.dtype)


def affine_transform_batch(images_flat, angles_deg, scales, shifts_yx):
    """Apply a batch of (rotation, scaling, translation) transforms.

    Parameters
    ----------
    images_flat : (N, 784) array, normalised to [0,1].
    angles_deg : (N,) array of rotation angles in degrees.
    scales : (N,) array of isotropic scales (e.g. 0.95 to 1.05).
    shifts_yx : (N, 2) array of (shift_y, shift_x) in pixel units.

    Returns
    -------
    out : (N, 784) array of transformed images.
    """
    N = images_flat.shape[0]
    images_2d = images_flat.reshape(N, 28, 28)
    H = W = 28
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)
    out = np.empty_like(images_2d)
    for i in range(N):
        theta = np.deg2rad(angles_deg[i])
        s = scales[i]
        sy, sx = shifts_yx[i]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dy = yy - cy - sy
        dx = xx - cx - sx
        src_y = (cos_t * dy + sin_t * dx) / s + cy
        src_x = (-sin_t * dy + cos_t * dx) / s + cx
        out[i] = _bilinear_sample(images_2d[i:i + 1], src_y, src_x)[0]
    return out.reshape(N, -1)


def make_augmenter(use_rot: bool, use_trans: bool, use_scale: bool, seed: int = 12345,
                   rot_deg: float = 8.0, trans_pix: int = 2, scale_range: Tuple[float, float] = (0.95, 1.05)):
    """Return a callable that augments a batch of (N, 784) flat images on the fly."""
    rng = np.random.default_rng(seed)

    def augment(images_flat):
        N = images_flat.shape[0]
        if use_rot:
            angles = rng.uniform(-rot_deg, rot_deg, size=N)
        else:
            angles = np.zeros(N)
        if use_scale:
            scales = rng.uniform(scale_range[0], scale_range[1], size=N)
        else:
            scales = np.ones(N)
        if use_trans:
            shifts = rng.integers(-trans_pix, trans_pix + 1, size=(N, 2)).astype(np.float32)
        else:
            shifts = np.zeros((N, 2), dtype=np.float32)
        if not (use_rot or use_trans or use_scale):
            return images_flat
        return affine_transform_batch(images_flat, angles, scales, shifts)
    return augment


def apply_perturbation_set(images_flat, mode: str):
    """Apply a deterministic perturbation to all test images.

    mode is one of: ``clean``, ``rot+10``, ``rot-10``, ``shift_left2``, ``shift_right2``,
    ``shift_up2``, ``shift_down2``, ``scale_up_1.08``, ``scale_down_0.92``.
    """
    N = images_flat.shape[0]
    angles = np.zeros(N)
    scales = np.ones(N)
    shifts = np.zeros((N, 2), dtype=np.float32)
    if mode == 'clean':
        return images_flat
    if mode == 'rot+10':
        angles = np.full(N, 10.0)
    elif mode == 'rot-10':
        angles = np.full(N, -10.0)
    elif mode == 'shift_left2':
        shifts[:, 1] = -2
    elif mode == 'shift_right2':
        shifts[:, 1] = 2
    elif mode == 'shift_up2':
        shifts[:, 0] = -2
    elif mode == 'shift_down2':
        shifts[:, 0] = 2
    elif mode == 'scale_up_1.08':
        scales = np.full(N, 1.08)
    elif mode == 'scale_down_0.92':
        scales = np.full(N, 0.92)
    else:
        raise ValueError(f'Unknown perturbation mode {mode}')
    return affine_transform_batch(images_flat, angles, scales, shifts)


def gaussian_noise(images_flat, sigma, seed):
    rng = np.random.default_rng(seed)
    return np.clip(images_flat + rng.normal(0.0, sigma, images_flat.shape).astype(images_flat.dtype), 0.0, 1.0)


# -----------------------------------------------------------------------------
# Model factories
# -----------------------------------------------------------------------------

def make_mlp(weight_decay: float = 0.0, hidden: int = 600):
    lambdas = [weight_decay, weight_decay] if weight_decay > 0 else None
    return nn.models.Model_MLP([784, hidden, 10], 'ReLU', lambdas)


def make_cnn(weight_decay: float = 1e-4, dropout_rate: float = 0.0):
    if weight_decay > 0:
        lambdas = [weight_decay] * 4
    else:
        lambdas = None
    return nn.models.Model_CNN(num_classes=10, lambda_list=lambdas, dropout_rate=dropout_rate)


def count_params(model) -> int:
    total = 0
    for layer in getattr(model, 'layers', []):
        if getattr(layer, 'optimizable', False):
            for p in layer.params.values():
                total += int(np.prod(p.shape))
    return total


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def predict_logits(model, images_flat, batch_size: int = 256) -> np.ndarray:
    if hasattr(model, 'set_training'):
        model.set_training(False)
    out = []
    for start in range(0, images_flat.shape[0], batch_size):
        batch = images_flat[start:start + batch_size]
        out.append(model(batch))
    return np.concatenate(out, axis=0)


def evaluate_acc(model, images_flat, labels, batch_size: int = 256):
    logits = predict_logits(model, images_flat, batch_size)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == labels)), logits, preds


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


# -----------------------------------------------------------------------------
# Statistics: confusion matrix, per-class metrics, ECE
# -----------------------------------------------------------------------------

def confusion_matrix(preds, labels, num_classes: int = 10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def per_class_metrics(cm):
    diag = np.diag(cm).astype(np.float64)
    row = cm.sum(axis=1).astype(np.float64)
    col = cm.sum(axis=0).astype(np.float64)
    recall = np.where(row > 0, diag / np.maximum(row, 1), 0.0)
    precision = np.where(col > 0, diag / np.maximum(col, 1), 0.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return precision, recall, f1


def top_confusion_pairs(cm, k: int = 5):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((i, j, int(cm[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:k]


def expected_calibration_error(probs, labels, n_bins: int = 15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    N = labels.shape[0]
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidences > lo) & (confidences <= hi) if b > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            bins.append({'bin_lo': float(lo), 'bin_hi': float(hi), 'count': 0,
                         'avg_conf': 0.0, 'avg_acc': 0.0})
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc = float(accuracies[mask].mean())
        weight = mask.sum() / N
        ece += weight * abs(avg_acc - avg_conf)
        bins.append({'bin_lo': float(lo), 'bin_hi': float(hi), 'count': int(mask.sum()),
                     'avg_conf': avg_conf, 'avg_acc': avg_acc})
    return float(ece), bins


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

PLOT_STYLE = {
    'figure.dpi': 220,
    'savefig.dpi': 220,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'font.family': 'serif',
}
plt.rcParams.update(PLOT_STYLE)


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_curves(train_loss, dev_loss, train_acc, dev_acc, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].plot(np.arange(len(train_loss)), train_loss, color='#4C78A8', lw=0.8, label='train')
    if len(dev_loss) > 0:
        xs = np.linspace(0, max(len(train_loss) - 1, 0), len(dev_loss))
        axes[0].plot(xs, dev_loss, '--', color='#F58518', lw=1.4, label='valid')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('loss')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[1].plot(np.arange(len(train_acc)), train_acc, color='#4C78A8', lw=0.8, label='train')
    if len(dev_acc) > 0:
        xs = np.linspace(0, max(len(train_acc) - 1, 0), len(dev_acc))
        axes[1].plot(xs, dev_acc, '--', color='#F58518', lw=1.4, label='valid')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('accuracy')
    axes[1].set_ylim(0.6, 1.005)
    axes[1].legend()
    fig.suptitle(title)
    save_fig(fig, save_path)


def plot_confusion_matrix(cm, save_path, title, normalized=False):
    if normalized:
        denom = cm.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        mat = cm / denom
        fmt = '{:.2f}'
        cmap = 'Blues'
    else:
        mat = cm
        fmt = '{:d}'
        cmap = 'Blues'
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.imshow(mat, cmap=cmap)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    for i in range(10):
        for j in range(10):
            val = mat[i, j]
            if normalized:
                if val > 0.05:
                    ax.text(j, i, fmt.format(val), ha='center', va='center', fontsize=6,
                            color='white' if val > 0.5 else 'black')
            else:
                if val > 0:
                    ax.text(j, i, fmt.format(int(val)), ha='center', va='center', fontsize=6,
                            color='white' if val > mat.max() * 0.5 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, save_path)


def plot_filters(weights, save_path, title):
    """Visualise convolution filters of shape (out, in, k, k)."""
    out_c, in_c, kh, kw = weights.shape
    cols = min(out_c, 16)
    fig, axes = plt.subplots(in_c, cols, figsize=(cols * 0.8, in_c * 0.9))
    if in_c == 1:
        axes = np.array([axes])
    for c in range(in_c):
        for o in range(cols):
            ax = axes[c, o] if cols > 1 else axes[c]
            kernel = weights[o, c]
            vmax = max(abs(kernel.min()), abs(kernel.max()))
            ax.imshow(kernel, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(f'k{o}', fontsize=7)
    fig.suptitle(title)
    save_fig(fig, save_path)


def plot_mlp_weights(weight_matrix, save_path, title, max_neurons: int = 64):
    cols = min(weight_matrix.shape[1], max_neurons)
    side = int(np.ceil(np.sqrt(cols)))
    fig, axes = plt.subplots(side, side, figsize=(side * 0.9, side * 0.9))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for i in range(cols):
        kernel = weight_matrix[:, i].reshape(28, 28)
        vmax = max(abs(kernel.min()), abs(kernel.max()))
        axes[i].imshow(kernel, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    fig.suptitle(title)
    save_fig(fig, save_path)


def plot_misclassified(images_flat, labels, preds, probs, save_path, title, max_items=25):
    bad = np.where(labels != preds)[0]
    if bad.size == 0:
        return
    confidences = probs[bad].max(axis=1)
    order = np.argsort(-confidences)
    bad = bad[order][:max_items]
    side = int(np.ceil(np.sqrt(bad.shape[0])))
    fig, axes = plt.subplots(side, side, figsize=(side * 1.6, side * 1.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, idx in zip(axes, bad):
        img = images_flat[idx].reshape(28, 28)
        conf = probs[idx, preds[idx]]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'T:{int(labels[idx])} P:{int(preds[idx])}\n{conf*100:.0f}%', fontsize=7)
    fig.suptitle(title)
    save_fig(fig, save_path)


def plot_reliability(bins_list, ece_list, names, save_path, title):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
    for i, (bins, ece, name) in enumerate(zip(bins_list, ece_list, names)):
        confs = [b['avg_conf'] for b in bins if b['count'] > 0]
        accs = [b['avg_acc'] for b in bins if b['count'] > 0]
        ax.plot(confs, accs, '-o', color=colors[i % len(colors)], ms=4,
                label=f'{name} (ECE={ece*100:.2f}%)')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    save_fig(fig, save_path)


def plot_bar_comparison(labels, values, save_path, title, ylabel,
                        baseline=None, color='#4C78A8', annotate=True, ylim=None):
    fig, ax = plt.subplots(figsize=(max(5.5, len(labels) * 0.6), 3.5))
    bars = ax.bar(labels, values, color=color)
    if baseline is not None:
        ax.axhline(baseline, color='gray', ls='--', lw=1.0, label=f'baseline={baseline:.4f}')
        ax.legend(fontsize=8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if annotate:
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (max(values) - min(values)) * 0.01,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    save_fig(fig, save_path)


def plot_line_with_markers(xs, ys_dict, save_path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2']
    for i, (name, ys) in enumerate(ys_dict.items()):
        ax.plot(xs, ys, '-o', color=colors[i % len(colors)], label=name, ms=4, lw=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, save_path)


# -----------------------------------------------------------------------------
# Reproducibility helper
# -----------------------------------------------------------------------------

def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, np.floating) else o)


def feature_extractor_cnn(model, images_flat, batch_size: int = 256) -> np.ndarray:
    """Return the activations after the FC(64)+ReLU layer of Model_CNN, used as
    penultimate features for t-SNE / PCA visualisation."""
    if hasattr(model, 'set_training'):
        model.set_training(False)
    feats = []
    layers = model.layers
    # find the index of the second Linear (which produces logits); we want output
    # of the layer right before it (post ReLU after Linear(64)).
    last_linear_idx = None
    for i in range(len(layers) - 1, -1, -1):
        from mynn.op import Linear
        if isinstance(layers[i], Linear):
            last_linear_idx = i
            break
    assert last_linear_idx is not None
    feature_layers = layers[:last_linear_idx]
    for start in range(0, images_flat.shape[0], batch_size):
        x = images_flat[start:start + batch_size]
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, 28, 28)
        for layer in feature_layers:
            x = layer(x)
        feats.append(x)
    return np.concatenate(feats, axis=0)


def saliency_cnn(model, images_flat, labels, max_per_class: int = 1) -> np.ndarray:
    """Compute simple input gradient saliency for one example per class.

    Returns an array of shape (10, 28, 28) containing |dL/dx| for one image of
    each digit. Uses the model's forward + backward without modifying the
    optimisable parameters."""
    if hasattr(model, 'set_training'):
        model.set_training(False)
    chosen = []
    for c in range(10):
        idxs = np.where(labels == c)[0][:max_per_class]
        if len(idxs) == 0:
            continue
        chosen.extend(int(i) for i in idxs)
    chosen = np.array(chosen, dtype=np.int64)
    sel_imgs = images_flat[chosen]
    sel_labels = labels[chosen].astype(np.int64)
    # forward
    x = sel_imgs.reshape(-1, 1, 28, 28)
    for layer in model.layers:
        x = layer(x)
    logits = x  # (N, 10)
    probs = softmax_np(logits)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(chosen)), sel_labels] = 1
    grad_out = (probs - one_hot) / len(chosen)
    g = grad_out
    for layer in reversed(model.layers):
        g = layer.backward(g)
    g = g.reshape(-1, 28, 28)
    return np.abs(g)


def plot_saliency(images_flat, labels, saliency_maps, save_path, title):
    fig, axes = plt.subplots(2, 10, figsize=(10, 2.4))
    for c in range(10):
        idx = int(np.where(labels == c)[0][0]) if (labels == c).any() else 0
        axes[0, c].imshow(images_flat[idx].reshape(28, 28), cmap='gray')
        axes[0, c].set_title(f'{c}', fontsize=8)
        axes[0, c].axis('off')
        axes[1, c].imshow(saliency_maps[c], cmap='hot')
        axes[1, c].axis('off')
    axes[0, 0].set_ylabel('input')
    axes[1, 0].set_ylabel('|dL/dx|')
    fig.suptitle(title)
    save_fig(fig, save_path)
