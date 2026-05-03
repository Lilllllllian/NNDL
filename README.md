# MNIST From Scratch 

A from-scratch MNIST classifier built purely with NumPy.


## What's in this repo

| Path | What it is |
|---|---|
| `codes/mynn/op.py` | `Linear`, `conv2D` (im2col + `einsum`), `ReLU`, `Flatten`, `MaxPool2D`, `Dropout`, `MultiCrossEntropyLoss` |
| `codes/mynn/optimizer.py` | `SGD` and `MomentGD` with decoupled $L_2$ weight decay |
| `codes/mynn/lr_scheduler.py` | `StepLR`, `MultiStepLR`, `ExponentialLR` |
| `codes/mynn/models.py` | `Model_MLP` (784–600–10) and `Model_CNN` (two conv blocks + FC bottleneck) |
| `codes/mynn/runner.py` | Mini-batch train / validate / best-checkpoint loop |
| `codes/study_utils.py` | MNIST loader, bilinear affine sampler, plotting helpers |
| `codes/run_full_study.py` | Experiment driver — `--pack {main,optim,reg,aug,robust,error,all}` |
| `codes/gradient_check.py` | Analytical vs.\ central-difference gradient verification |

MNIST itself sits at `codes/dataset/MNIST/*.gz` .

## Headline results (full 50 k training set)

| Model | Params | Test acc. | ECE (15-bin) | Wall-clock (1 CPU) |
|---|---:|---:|---:|---:|
| MLP (600 hidden) | 477 010 | 97.54 % | 1.08 % | ~2.4 min |
| **CNN (8/16/64)** | **52 138** | **98.78 %** | **0.26 %** | ~59 min |

The CNN wins on every axis except wall-clock — the expected price of
the richer spatial inductive bias — and its ECE is about 4× smaller
than the MLP's despite having 9× fewer parameters.

## Findings

A folk intuition says "data augmentation improves generalisation,
period". In our robustness study that is only half right.

| Probe | Clean-trained CNN | Affine-trained CNN (RTS) |
|---|---:|---:|
| Worst-case held-out affine perturbation | −7.3 % drop | −0.95 % drop |
| Accuracy at Gaussian noise σ = 0.30 | **93.43 %** | **62.38 %** |

Affine augmentation shrinks the worst-case accuracy drop under
held-out affine shifts by nearly an order of magnitude, but the same
model is **dramatically worse** under additive pixel-level Gaussian
noise it never saw during training. The report (§Robustness and
§Discussion) argues this is because augmentation sharpens features
along the sampled symmetry group, and sharper features are more
sensitive to off-family perturbations. That is a directed prior, not a
universal regulariser.

## Reproducing the numbers

### Option A — Google Colab (recommended if you don't have a local Python)

Open `Colab_Run.ipynb` in Colab. Point the `REPO_URL` cell at your
fork, then run everything top-to-bottom. About 2–2.5 hours on a free
CPU runtime; the notebook downloads the final PDF and a zipped
`results_full` bundle of every JSON summary, checkpoint, and figure.

### Option B — Local machine

```bash
pip install -r requirements.txt
cd codes

# 1. sanity: analytical gradients match finite differences
python gradient_check.py

# 2. Part A + Part B (MLP 30 epochs, CNN 20 epochs on full 50k)
python -u run_full_study.py --pack main

# 3. the five ablations
python -u run_full_study.py --pack optim
python -u run_full_study.py --pack reg
python -u run_full_study.py --pack aug
python -u run_full_study.py --pack robust
python -u run_full_study.py --pack error
```

Every pack supports resume — a sub-run is skipped if both its
checkpoint and its history JSON already exist, so you can interrupt
and restart without losing progress.

## Design notes

- **Determinism.** Seed `309` is fixed everywhere (`np.random`, data
  split, per-layer dropout RNG). Re-running the pipeline with the
  same hyperparameters reproduces the numbers in the report
  bit-for-bit.
- **Decoupled $L_2$.** Weight decay is applied to the parameters in
  the optimiser step, not folded into the loss gradient — equivalent
  for plain SGD but cleaner under momentum (cf.\ AdamW /
  Loshchilov–Hutter).
- **im2col convolution.** `conv2D` unrolls patches with an
  `einsum`/view composition; `numpy.add.at` handles the
  scatter-add on overlapping receptive fields in the backward.
- **Fused softmax + cross-entropy gradient.** We compute $\partial
  \mathcal{L}/\partial z = (p-y)/N$ in closed form rather than
  composing the softmax Jacobian with the CE gradient. Faster and
  numerically tighter.
- **One sampler, two uses.** The bilinear affine sampler in
  `study_utils.py` is used both for training-time augmentation and
  for the deterministic test-time perturbations in the robustness
  study — so train- and test-time geometric distributions are
  guaranteed to be produced by the same code path.

