# MNIST From Scratch — NumPy-only training pipeline

A complete MNIST classifier built with NumPy alone (no PyTorch / TensorFlow).
Implements every layer, optimiser and scheduler from scratch and runs five
ablation studies on top of the resulting CNN.

## Quick start (Google Colab — recommended)

Open `Colab_Run.ipynb` in Colab. Edit the `REPO_URL` cell to point to this
fork, then run every cell top to bottom. Total runtime on a free CPU
runtime is roughly **2–2.5 hours**; the notebook downloads the final
`MNIST_From_Scratch_Report_Leyan_Huang.pdf` plus a `results_full.zip`
of every JSON summary, model checkpoint and PNG figure.

## Quick start (local machine)

```bash
pip install -r requirements.txt
cd codes

# 1. Sanity check (analytical vs numerical gradients)
python gradient_check.py

# 2. Part A + Part B (MLP 30 ep, CNN 20 ep on full 50k)
python -u run_full_study.py --pack main

# 3. The five ablation studies (~80 min on CPU)
python -u run_full_study.py --pack optim
python -u run_full_study.py --pack reg
python -u run_full_study.py --pack aug
python -u run_full_study.py --pack robust
python -u run_full_study.py --pack error

# 4. Render the LaTeX report from the JSON summaries
python build_report.py

# 5. Compile PDF (needs a TeX distribution: MiKTeX / TeX Live)
cd ..
pdflatex -interaction=nonstopmode MNIST_From_Scratch_Report_Leyan_Huang.tex
pdflatex -interaction=nonstopmode MNIST_From_Scratch_Report_Leyan_Huang.tex
```

Each pack supports a resume mechanism — completed sub-runs (those whose
checkpoint AND history JSON both exist) are skipped on re-launch.

## Layout

```
codes/
  mynn/                  # from-scratch NN library
    op.py                # Linear, conv2D, ReLU, Flatten, MaxPool2D, Dropout, MultiCrossEntropyLoss
    optimizer.py         # SGD, MomentGD with decoupled L2
    lr_scheduler.py      # StepLR, MultiStepLR, ExponentialLR
    models.py            # Model_MLP, Model_CNN
    runner.py            # batched train/validate/best-checkpoint loop
  dataset/MNIST/         # raw MNIST .gz files (kept in-repo so Colab works offline)
  study_utils.py         # data loading, geometric augmentation, plotting helpers
  run_full_study.py      # experiment driver (--pack {main,optim,reg,aug,robust,error,all})
  build_report.py        # renders the LaTeX report from results/full/*/summary.json
  gradient_check.py      # analytical-vs-numerical gradient verification
Colab_Run.ipynb          # one-click reproduction notebook
requirements.txt         # numpy, matplotlib
```

## Headline numbers (full 50k training)

| Model | Parameters | Test accuracy | ECE   |
|-------|-----------:|--------------:|------:|
| MLP   |   477 010  |        97.25% | 1.29% |
| CNN   |    52 138  |    **98.73%** | 0.20% |

(Run the pipeline yourself to reproduce these numbers — every figure and
every cell of every table in the report is generated from the JSON
summaries written by `run_full_study.py`.)
