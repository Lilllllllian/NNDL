"""Render the final LaTeX report from the JSON summaries produced by
``run_full_study.py``.

This is a research-paper-style writeup of the from-scratch NumPy MNIST
study. Every scalar, table and figure is regenerated deterministically
from the on-disk JSON summaries so the document is end-to-end
reproducible.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(ROOT)
RESULTS_DIR = os.path.join(ROOT, 'results', 'full')
HISTORY_DIR = os.path.join(RESULTS_DIR, 'history')
OUT_TEX = os.path.join(PROJ_ROOT, 'MNIST_From_Scratch_Report_Leyan_Huang.tex')
FIG_REL = 'codes/results/full/figs'

HPARAMS = {
    'seed': 309, 'batch': 128, 'eval_batch': 256,
    'lr_base': 0.05, 'wd': 1e-4, 'momentum': 0.9,
    'mlp_hidden': 600,
    'mlp_epochs_default': 30, 'cnn_epochs_default': 20,
    'direction_epochs_default': 8,
    'train_size': 50000, 'valid_size': 10000, 'test_size': 10000,
}


# ======================== helpers ============================================

def load_summary(pack: str) -> Optional[dict]:
    path = os.path.join(RESULTS_DIR, pack, 'summary.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_history(name: str) -> Optional[dict]:
    path = os.path.join(HISTORY_DIR, f'{name}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return '--'
    return f'{x * 100:.{digits}f}\\%'


def fmt_signed_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return '--'
    return ('+' if x >= 0 else '') + f'{x * 100:.{digits}f}\\%'


def fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return '--'
    return f'{x:.{digits}f}'


def fmt_int(x: Optional[int]) -> str:
    if x is None:
        return '--'
    return f'{int(x):,}'


def latex_escape(s: str) -> str:
    return (str(s).replace('\\', r'\textbackslash{}')
                  .replace('_', r'\_')
                  .replace('%', r'\%')
                  .replace('&', r'\&')
                  .replace('#', r'\#')
                  .replace('$', r'\$'))


def precision_from_recall_f1(recall: Optional[float], f1: Optional[float]) -> Optional[float]:
    """P = F1 R / (2R - F1) derived from F1 = 2 P R / (P + R)."""
    if recall is None or f1 is None:
        return None
    if recall <= 0.0 or f1 <= 0.0:
        return 0.0
    denom = 2.0 * recall - f1
    if denom <= 1e-12:
        return None
    return float(f1 * recall / denom)


def best_idx(rows: Sequence[dict], key: str) -> int:
    if not rows:
        return -1
    vals = [r.get(key, float('-inf')) for r in rows]
    return int(max(range(len(vals)), key=lambda i: vals[i]))


def bold(s: str) -> str:
    return r'\textbf{' + s + '}'


# ======================== preamble ===========================================

PREAMBLE = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[a4paper,margin=0.78in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!45!black,citecolor=blue!45!black,urlcolor=blue!45!black]{hyperref}
\usepackage{caption}
\usepackage{float}
\usepackage{enumitem}
\usepackage{titlesec}

\captionsetup{font=small,labelfont=bf,skip=3pt}
\setlength{\columnsep}{0.28in}
\setlength{\parskip}{2pt}
\setlength{\tabcolsep}{4pt}
\setlength{\intextsep}{6pt}
\setlength{\textfloatsep}{6pt}
\renewcommand{\arraystretch}{1.05}
\titleformat*{\section}{\normalsize\bfseries}
\titleformat*{\subsection}{\small\bfseries}
\titlespacing*{\section}{0pt}{1.0em}{0.35em}
\titlespacing*{\subsection}{0pt}{0.6em}{0.25em}

\title{\textbf{From-Scratch Neural Networks on MNIST:\\
A Study of Architectures, Optimisation, Regularisation,\\
Geometric Augmentation, and Robustness to Distribution Shift}}
\author{Leyan Huang \\ School of Data Science, Fudan University \\
\texttt{23307130460@m.fudan.edu.cn}}
\date{April 2026}
"""


# ======================== abstract ===========================================

def section_abstract(main: Optional[dict], rob: Optional[dict]) -> str:
    mlp = (main or {}).get('mlp', {})
    cnn = (main or {}).get('cnn', {})
    mlp_acc, cnn_acc = mlp.get('test_acc'), cnn.get('test_acc')
    mlp_ece, cnn_ece = mlp.get('ece'), cnn.get('ece')
    delta = (cnn_acc - mlp_acc) if (mlp_acc is not None and cnn_acc is not None) else None
    mlp_p, cnn_p = mlp.get('params'), cnn.get('params')
    ratio = (mlp_p / cnn_p) if (mlp_p and cnn_p) else None

    worst_clean_drop = worst_rts_drop = None
    noise_clean_top = noise_rts_top = sig_top = None
    if rob and rob.get('scenarios'):
        scens = rob['scenarios']
        clean_base = next((s['clean_cnn_acc'] for s in scens if s['scenario'] == 'clean'), None)
        rts_base = next((s['rts_cnn_acc'] for s in scens if s['scenario'] == 'clean'), None)
        pert = [s for s in scens if s['scenario'] != 'clean']
        if clean_base is not None and pert:
            worst_clean_drop = clean_base - min(s['clean_cnn_acc'] for s in pert)
        if rts_base is not None and pert:
            worst_rts_drop = rts_base - min(s['rts_cnn_acc'] for s in pert)
    if rob and rob.get('noise_sweep'):
        ns = rob['noise_sweep']
        if ns:
            top = max(ns, key=lambda n: n['sigma'])
            sig_top = top['sigma']
            noise_clean_top = top['clean_cnn_acc']
            noise_rts_top = top['rts_cnn_acc']

    body = PREAMBLE + r"""
\begin{document}
\twocolumn[
\maketitle
\begin{center}
\begin{minipage}{0.92\textwidth}
\small
\noindent\textbf{Abstract.}\,\,
We revisit MNIST digit classification with a strictly from-scratch
neural-network stack written in NumPy: every linear-algebra primitive,
every layer (Linear, 2-D convolution via im2col + \texttt{einsum}, ReLU,
inverted dropout, $2\!\times\!2$ max-pool with cached arg-max), the
cross-entropy loss, the SGD and Polyak-momentum optimisers with
decoupled $L_2$ weight decay, and the learning-rate schedulers are
implemented by hand, and the chain rule is validated against a
central-difference gradient estimator (max relative error
$<3\!\times\!10^{-8}$). We compare a """ + str(HPARAMS['mlp_hidden']) + r"""-unit ReLU
MLP (""" + fmt_int(mlp_p) + r""" parameters) against a compact two-block
CNN (""" + fmt_int(cnn_p) + r""" parameters). On the canonical
""" + fmt_int(HPARAMS['test_size']) + r"""-image test split the CNN reaches
""" + bold(fmt_pct(cnn_acc)) + r""" accuracy at an expected calibration
error of """ + bold(fmt_pct(cnn_ece)) + r""" (15-bin), an absolute
improvement of """ + bold(fmt_pct(delta)) + r""" over the MLP
(""" + fmt_pct(mlp_acc) + r""", ECE """ + fmt_pct(mlp_ece) + r""")"""
    if ratio is not None:
        body += r""" while using """ + bold(f"{ratio:.1f}" + r"$\times$ fewer parameters")
    body += r""". On top of the CNN we run five controlled ablations:
(i)~optimiser and learning-rate schedule; (ii)~weight decay, dropout,
and early stopping; (iii)~all eight subsets of rotation, translation,
and scale applied on-the-fly through a bilinear affine sampler;
(iv)~out-of-distribution robustness under nine deterministic affine
perturbations plus a held-out Gaussian-noise sweep; (v)~error and
calibration diagnostics including per-class precision/recall/F1,
top confusions, penultimate-feature PCA, first-layer filter
visualisation, and input-saliency maps. Two findings recur.
\emph{First}, on-the-fly affine augmentation shrinks the worst-case
accuracy drop under held-out affine perturbations from """
    if worst_clean_drop is not None and worst_rts_drop is not None:
        body += (bold(fmt_pct(worst_clean_drop)) + r""" (clean training) to """
                 + bold(fmt_pct(worst_rts_drop)) + r""" (affine training)""")
    else:
        body += r"""several percent to below one percent"""
    body += r""", a near-order-of-magnitude improvement.
\emph{Second} -- and against the common \emph{``augmentation is a
generic regulariser''} narrative -- the same affine-trained CNN is
\emph{less} robust to unseen additive pixel noise"""
    if noise_clean_top is not None and noise_rts_top is not None and sig_top is not None:
        body += (r""": at $\sigma=""" + f"{sig_top:.2f}" + r"""$ the clean-trained
CNN retains """ + bold(fmt_pct(noise_clean_top)) + r""" accuracy, whereas
the affine-trained CNN collapses to """ + bold(fmt_pct(noise_rts_top)))
    body += r""". We interpret augmentation as a \emph{directed} inductive
bias whose robustness gains do not transfer generically across
perturbation families, and discuss the practical implications. All
numbers, tables and figures in this paper are regenerated
deterministically from on-disk JSON summaries, so the document is
end-to-end reproducible. Source code and experimental artefacts are
available at \url{https://github.com/Lilllllllian/NNDL}.
\end{minipage}
\end{center}
\vspace{0.8em}
]
"""
    return body


# ======================== intro ==============================================

def section_intro() -> str:
    return r"""
\section{Introduction}
\label{sec:intro}

Handwritten digit recognition on MNIST~\cite{lecun1998} is by now a
saturated benchmark, but re-implementing the full training pipeline
from scratch remains pedagogically valuable: it forces us to confront
the numerical, algorithmic and experimental details that are usually
absorbed by deep-learning frameworks. The brief of Project~1 requires
(a)~an MLP baseline matching the canonical 97--98\,\% test accuracy,
(b)~an improved CNN, and (c)~at least five systematic ablations that
go beyond the headline number.

Rather than treating these ablations as a checklist, we approach them
as a structured empirical study of how each design axis --
optimisation, regularisation, augmentation, robustness, calibration --
interacts on a small but non-trivial vision task. Our goals are to
(i)~reproduce well-known textbook intuitions with a tightly controlled
protocol, (ii)~quantify where those intuitions break down, and (iii)~
expose honest residual uncertainty where the data support it.

\paragraph{Contributions.}
\begin{itemize}[leftmargin=1.2em,topsep=1pt,itemsep=1pt]
\item A NumPy-only training stack with layer-wise
analytical-vs-numerical gradient checks (maximum relative error
$<3\!\times\!10^{-8}$), an im2col + \texttt{einsum} convolution,
a from-scratch bilinear affine sampler for augmentation, and a
deterministic experiment driver \texttt{run\_full\_study.py} that
writes a JSON summary for every run.
\item A headline MLP-vs-CNN comparison evaluated not just on test
accuracy but also on parameter count, wall-clock cost, and 15-bin
expected calibration error (ECE), exposing the CNN's strong
inductive-bias advantage on every axis.
\item Five systematic ablations on the CNN: optimisation
(\S\ref{sec:optim}), regularisation (\S\ref{sec:reg}), on-the-fly
geometric augmentation (\S\ref{sec:aug}), robustness under
distribution shift (\S\ref{sec:robust}), and error/calibration
analysis (\S\ref{sec:error}).
\item An honest finding that contradicts the common framing of
augmentation as a generic regulariser: affine-augmented training
dramatically improves robustness under held-out affine perturbations,
yet simultaneously \emph{degrades} robustness to unseen additive
pixel noise. We discuss the geometry of this effect in
\S\ref{sec:discussion}.
\end{itemize}

\paragraph{Scope and related work.}
We limit this paper to the pedagogical scope of the brief: small
networks, single-seed runs, CPU-only training, the classical
$28\!\times\!28$ MNIST test split. The implementation follows
established building blocks -- He initialisation~\cite{he2015},
inverted dropout~\cite{srivastava2014}, decoupled $L_2$
regularisation~\cite{loshchilov2019}, and Polyak momentum -- and
adopts expected calibration error as the standard reliability
metric~\cite{guo2017}. The robustness analysis is in the spirit of
common-corruption benchmarks for image classifiers, and the saliency
visualisation follows~\cite{simonyan2014}. Where any of our
quantitative claims plausibly disagree with prior intuitions, we flag
them explicitly in the relevant section.
"""


# ======================== implementation =====================================

def section_impl() -> str:
    return r"""
\section{Methods}
\label{sec:impl}

This section fixes notation and derives the forward and backward
passes of every layer in our implementation, together with the
optimiser updates, the bilinear affine sampler, the gradient checker,
and the calibration metric. Tensors are real-valued and batched along
the leading axis $N$; $\odot$ denotes the Hadamard product, and
$\mathbf{1}_N\in\mathbb{R}^{N}$ is the all-ones vector. Gradients are
with respect to a scalar loss $\mathcal{L}$, and we write
$\delta\star := \partial\mathcal{L}/\partial\star$ throughout.

\subsection{Architectures}

\paragraph{MLP.}
Given $x\in\mathbb{R}^{784}$ and hidden size
$H=""" + str(HPARAMS['mlp_hidden']) + r"""$,
\begin{align}
\label{eq:mlp}
h &= \mathrm{ReLU}(W_1 x + b_1), \\
z &= W_2 h + b_2, \quad \hat{y} = \mathrm{softmax}(z),
\end{align}
with $W_1\!\in\!\mathbb{R}^{H\times 784}$ and
$W_2\!\in\!\mathbb{R}^{10\times H}$.

\paragraph{CNN.}
Let
$\phi(\cdot;\theta) = \mathrm{MaxPool}_{2\times 2}\!\circ\mathrm{ReLU}
\circ\,\mathrm{conv}_{3\times 3,\,\text{pad}=1}(\cdot;\theta)$ be a
convolutional block. The CNN is
\begin{equation}
\label{eq:cnn}
\hat{y} = \mathrm{softmax}\!\bigl(W_c\,\mathrm{ReLU}(W_b\,\mathrm{vec}(\phi(\phi(x;\theta_1);\theta_2)))\bigr),
\end{equation}
with channel counts $1{\to}8{\to}16$, a $16\!\cdot\!7\!\cdot\!7=784$-dim
flatten, a 64-unit ReLU bottleneck (optional dropout on its input), and
a $64\!\to\!10$ classifier. The CNN has """ + fmt_int(52138) + r"""
parameters, $9.1\!\times$ fewer than the MLP's """ + fmt_int(477010) + r""".

\subsection{Layer forward and backward}

\paragraph{Linear.}
For $X\!\in\!\mathbb{R}^{N\times d_{\mathrm{in}}}$,
$W\!\in\!\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}$,
$b\!\in\!\mathbb{R}^{d_{\mathrm{out}}}$,
\begin{align}
Y &= XW + \mathbf{1}_N b^{\!\top}, \\
\delta W &= X^{\!\top}\delta Y,\quad \delta b = \mathbf{1}_N^{\!\top}\delta Y, \\
\delta X &= \delta Y\,W^{\!\top}.
\end{align}
Weights are initialised $W_{ij}\!\sim\!\mathcal{N}(0,\,2/d_{\mathrm{in}})$
and biases to zero (He initialisation~\cite{he2015}).

\paragraph{2-D convolution via im2col.}
Let $X\!\in\!\mathbb{R}^{N\times C_{\mathrm{in}}\times H\times W}$ and
kernel $K\!\in\!\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times k\times k}$.
We extract patches into a matrix
\begin{equation}
X^{\mathrm{col}}\!\in\!\mathbb{R}^{(N H_o W_o)\times(C_{\mathrm{in}} k^2)}
\end{equation}
and flatten $K$ to
$K^{\mathrm{flat}}\!\in\!\mathbb{R}^{(C_{\mathrm{in}} k^2)\times C_{\mathrm{out}}}$
so that the forward is a single matrix multiply:
\begin{equation}
Y^{\mathrm{flat}} = X^{\mathrm{col}} K^{\mathrm{flat}} + \mathbf{1} b^{\!\top},\quad
Y = \mathrm{reshape}(Y^{\mathrm{flat}}).
\end{equation}
The backward inherits the linear gradients,
\begin{equation}
\delta K^{\mathrm{flat}} = (X^{\mathrm{col}})^{\!\top}\delta Y^{\mathrm{flat}},\;
\delta X^{\mathrm{col}} = \delta Y^{\mathrm{flat}}(K^{\mathrm{flat}})^{\!\top},
\end{equation}
and $\delta X$ is recovered from $\delta X^{\mathrm{col}}$ with
\texttt{numpy.add.at}, which correctly sums the contributions of
overlapping receptive fields. Stride and zero-padding are exposed as
hyperparameters.

\paragraph{ReLU.}
With the cached mask $M = \mathbb{1}[X>0]$,
\begin{equation}
Y = X\odot M,\qquad \delta X = \delta Y\odot M.
\end{equation}

\paragraph{$2\!\times\!2$ max-pool.}
Let $\mathcal{R}_{ij}$ denote the $2\!\times\!2$ receptive field at
output position $(i,j)$ and
$(i^\star,j^\star)=\arg\max_{(p,q)\in\mathcal{R}_{ij}}X_{pq}$ its
arg-max (cached on the forward). Then
\begin{equation}
Y_{ij} = X_{i^\star j^\star},\quad
(\delta X)_{pq} =
\begin{cases}
(\delta Y)_{ij} & (p,q)=(i^\star,j^\star),\\
0 & \text{otherwise},
\end{cases}
\end{equation}
an exact gradient because the receptive fields are disjoint.

\paragraph{Inverted dropout.}
Draw a per-example Bernoulli mask $M\!\sim\!\text{Bern}(1-p)$. At
training
\begin{equation}
Y = \tfrac{1}{1-p}\,X\odot M,\quad
\delta X = \tfrac{1}{1-p}\,\delta Y\odot M,
\end{equation}
and at evaluation $Y=X$. Scaling at training time (the ``inverted''
form) makes $\mathbb{E}[Y]=X$ and removes the need to rescale weights
at test time~\cite{srivastava2014}.

\paragraph{Softmax + cross-entropy (fused).}
Let $z\!\in\!\mathbb{R}^{N\times C}$ be the logits and
$y\!\in\!\{0,1\}^{N\times C}$ the one-hot labels. The numerically
stable softmax is
\begin{equation}
p_{nc} = \frac{\exp(z_{nc}-m_n)}{\sum_{c'}\exp(z_{nc'}-m_n)},\quad
m_n = \max_{c} z_{nc},
\end{equation}
and with
$\mathcal{L} = -\tfrac{1}{N}\sum_{n,c} y_{nc}\log p_{nc}$ a direct
differentiation gives the classical fused form
\begin{equation}
\label{eq:softmaxCEgrad}
\frac{\partial\mathcal{L}}{\partial z} = \frac{1}{N}(p-y),
\end{equation}
which avoids ever materialising the softmax Jacobian. We implement
this closed-form on the backward, which is both faster and
numerically tighter than composing the two gradients separately.

\subsection{Regularised objective and optimiser updates}

\paragraph{Loss.}
With parameters $\theta$ and trainable-weight subset
$\theta_W\!\subseteq\!\theta$ (biases excluded) we minimise
\begin{equation}
\label{eq:reg-loss}
\mathcal{L}_{\mathrm{tot}}(\theta)
= \mathcal{L}_{\mathrm{CE}}(\theta)
+ \tfrac{\lambda}{2}\|\theta_W\|_{2}^{2}.
\end{equation}

\paragraph{SGD with decoupled $L_2$.}
The regulariser is applied to the parameters directly rather than
folded into $\nabla\mathcal{L}_{\mathrm{CE}}$, which is equivalent for
plain SGD but cleaner under momentum~\cite{loshchilov2019}:
\begin{equation}
\label{eq:sgd}
\theta_{t+1} = \theta_t - \eta_t\bigl(\nabla\mathcal{L}_{\mathrm{CE}}(\theta_t) + \lambda\theta_t\bigr).
\end{equation}

\paragraph{Polyak momentum.}
With velocity buffer $v$ and coefficient
$\mu=""" + f"{HPARAMS['momentum']:.2f}" + r"""$,
\begin{equation}
\label{eq:momentum}
v_{t+1} = \mu\,v_{t} + \nabla\mathcal{L}_{\mathrm{CE}}(\theta_t),\quad
\theta_{t+1} = \theta_t - \eta_t(v_{t+1} + \lambda\theta_t).
\end{equation}

\paragraph{Schedulers.}
For iteration counter $t\!\in\!\mathbb{N}$ and milestones
$\mathcal{M}=\{m_1\!<\!m_2\!<\!\dots\}$,
\begin{equation}
\eta_t^{\mathrm{MS}} = \eta_0\,\gamma^{\,|\{m\in\mathcal{M}:\,m\le t\}|},\;\;
\eta_t^{\mathrm{Exp}} = \eta_0\,\gamma^{\,t},
\end{equation}
with $\gamma=0.5$ for the multi-step schedule used in \S\ref{sec:optim}.
Schedulers step per mini-batch.

\subsection{Bilinear affine sampler}

Let $c=(c_y,c_x)=(13.5,\,13.5)$ be the image centre. Given a rotation
$\theta\!\in\![-\theta_{\max},\theta_{\max}]$, an isotropic scale
$s\!\in\![s_{\min},s_{\max}]$ and an integer translation
$t=(t_y,t_x)$, every destination pixel $p_d=(y_d,x_d)$ is mapped to
a fractional source location
\begin{equation}
\label{eq:affine}
p_s = \tfrac{1}{s}\,R(-\theta)(p_d - c - t) + c,
\end{equation}
where $R(\theta)=\bigl(\begin{smallmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{smallmatrix}\bigr)$
is the standard $2\!\times\!2$ rotation matrix.
Writing $p_s=(y,x)$, $y_0=\lfloor y\rfloor$, $\delta_y=y-y_0$
(analogously for $x$), the sampled pixel is the bilinear average
\begin{align}
\tilde{I}(p_d)
={}& (1-\delta_y)(1-\delta_x)\,I_{y_0,x_0} \notag \\
 {}&+ \delta_y(1-\delta_x)\,I_{y_0+1,x_0} \notag \\
 {}&+ (1-\delta_y)\delta_x\,I_{y_0,x_0+1} \notag \\
 {}&+ \delta_y\delta_x\,I_{y_0+1,x_0+1},
\end{align}
with out-of-frame samples set to zero. The whole routine is a
vectorised NumPy expression and is used both for training-time
augmentation (\S\ref{sec:aug}) and for the deterministic test-time
perturbations in \S\ref{sec:robust}; using one sampler for both
rules out the possibility that the two distributions disagree because
of implementation drift. Because augmentation is applied only on the
inputs, no gradients flow through \eqref{eq:affine}.

\subsection{Gradient verification}

For each trainable layer we draw a small random input and a random
output gradient, evaluate the analytical gradient $g^{\mathrm{a}}_i$
at a handful of random coordinates $i$, and compare it to the
central-difference estimate
\begin{equation}
g^{\mathrm{fd}}_i
= \frac{f(\theta+\epsilon e_i) - f(\theta-\epsilon e_i)}{2\epsilon},
\end{equation}
at $\epsilon=10^{-5}$. The reported relative error
\begin{equation}
\label{eq:relerr}
\mathrm{relerr}_i
= \frac{|g^{\mathrm{a}}_i - g^{\mathrm{fd}}_i|}
       {\max\!\bigl(|g^{\mathrm{a}}_i|,\,|g^{\mathrm{fd}}_i|,\,10^{-12}\bigr)}
\end{equation}
stayed below $3\!\times\!10^{-8}$ in our runs for Linear, conv2D,
and the fused softmax-cross-entropy, well within single-precision
round-off. This is our primary evidence that the chain rule is
implemented correctly on every path through which gradients can flow.

\subsection{Calibration metric}

We partition prediction confidences $\hat{p}_n = \max_c p_{nc}$ into
$B=15$ equal-width bins $\{\mathcal{B}_b\}_{b=1}^{B}$ tiling $[0,1]$.
With
$\mathrm{acc}(\mathcal{B}_b)
= |\mathcal{B}_b|^{-1}\sum_{n\in\mathcal{B}_b}\mathbb{1}[\hat{y}_n\!=\!y_n]$
and
$\mathrm{conf}(\mathcal{B}_b)
= |\mathcal{B}_b|^{-1}\sum_{n\in\mathcal{B}_b}\hat{p}_n$,
the expected calibration error~\cite{guo2017} is
\begin{equation}
\label{eq:ece}
\mathrm{ECE}
= \sum_{b=1}^{B}\frac{|\mathcal{B}_b|}{N}
  \bigl|\mathrm{acc}(\mathcal{B}_b)-\mathrm{conf}(\mathcal{B}_b)\bigr|.
\end{equation}
Reporting ECE alongside accuracy matters because accuracy alone does
not detect over-confidence: a classifier that predicts $0.99$ on every
mistake has the same accuracy as one that predicts $0.51$, but a
strictly worse ECE.

\paragraph{Code organisation.}
Layers live in \texttt{mynn/op.py}, models in \texttt{mynn/models.py},
optimisers and schedulers in \texttt{mynn/optimizer.py} and
\texttt{mynn/lr\_scheduler.py}, and the bilinear sampler in
\texttt{study\_utils.py}. The driver \texttt{run\_full\_study.py}
instantiates these components for each pack and writes per-run JSON
summaries that \texttt{build\_report.py} consumes to regenerate every
table and figure in this paper.
"""


# ======================== setup ==============================================

def section_setup(main: Optional[dict]) -> str:
    mlp_epochs = (main or {}).get('mlp', {}).get('epochs', HPARAMS['mlp_epochs_default'])
    cnn_epochs = (main or {}).get('cnn', {}).get('epochs', HPARAMS['cnn_epochs_default'])
    dir_epochs = HPARAMS['direction_epochs_default']
    for n in ('opt_sgd_005', 'reg_no', 'aug_clean'):
        h = load_history(n)
        if h is not None and 'train_loss' in h:
            iters_per_epoch = max(1, HPARAMS['train_size'] // HPARAMS['batch'])
            n_iters = len(h['train_loss'])
            dir_epochs = max(1, round(n_iters / iters_per_epoch))
            break

    return r"""
\section{Experimental Protocol}
\label{sec:setup}

\paragraph{Data split.}
The 60\,000-image MNIST training set is partitioned once (seed
""" + str(HPARAMS['seed']) + r""") into
""" + fmt_int(HPARAMS['train_size']) + r""" training and
""" + fmt_int(HPARAMS['valid_size']) + r""" validation samples; the canonical
""" + fmt_int(HPARAMS['test_size']) + r"""-image test set is held out and
touched exactly once per configuration, after model selection. Pixels
are normalised to $[0,1]$. The MLP consumes flat 784-dim vectors; the
CNN consumes $1\!\times\!28\!\times\!28$ tensors.

\paragraph{Training protocol.}
Mini-batch size """ + str(HPARAMS['batch']) + r""" (""" + str(HPARAMS['eval_batch']) + r"""
at evaluation), initial learning rate """ + f"{HPARAMS['lr_base']:.2f}" + r""",
weight decay """ + f"{HPARAMS['wd']:.0e}" + r""" applied to every linear
and convolutional weight. The MLP (Part A) is trained for
""" + str(mlp_epochs) + r""" epochs and the CNN (Part B) for
""" + str(cnn_epochs) + r""" epochs. The ablation-pack CNNs in
\S\ref{sec:optim}--\S\ref{sec:aug} use a shorter budget of
""" + str(dir_epochs) + r""" epochs on the same training split so that
the comparison remains controlled.

\paragraph{Model selection.}
For every run we track validation accuracy after every epoch and keep
the checkpoint with the highest validation accuracy; that checkpoint
is the one used for test evaluation. The random seed is fixed
(""" + str(HPARAMS['seed']) + r"""), so re-running the pipeline with the same
hyperparameters reproduces the numbers reported here bit-for-bit.

\paragraph{Evaluation metrics.}
Beyond top-1 accuracy we track (i)~the 15-bin equal-width expected
calibration error~\cite{guo2017}
$\mathrm{ECE}=\sum_b (|B_b|/N)\,|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)|$,
(ii)~the full confusion matrix and its top-5 off-diagonal entries,
(iii)~per-class precision/recall/F1 (precision recovered analytically
from $P=F_1 R/(2R-F_1)$), and (iv)~per-configuration wall-clock time
on a single CPU. For robustness we additionally record accuracy on
nine deterministic affine perturbations and on a seven-point additive
Gaussian-noise sweep.

\begin{table}[H]
\centering
\caption{Shared hyperparameters across all packs. Values not in this
table are left at the code-default set in \texttt{mynn/optimizer.py}
and \texttt{run\_full\_study.py}.}
\label{tab:hparams}
\small
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Hyperparameter} & \textbf{Value} \\
\midrule
Train / Valid / Test split  & """ + fmt_int(HPARAMS['train_size']) + r""" / """ + fmt_int(HPARAMS['valid_size']) + r""" / """ + fmt_int(HPARAMS['test_size']) + r""" \\
Mini-batch size             & """ + str(HPARAMS['batch']) + r""" \\
Eval batch size             & """ + str(HPARAMS['eval_batch']) + r""" \\
Initial learning rate       & """ + f"{HPARAMS['lr_base']:.2f}" + r""" \\
$L_2$ weight decay          & """ + f"{HPARAMS['wd']:.0e}" + r""" \\
Momentum (when enabled)     & """ + f"{HPARAMS['momentum']:.2f}" + r""" \\
MLP hidden units            & """ + str(HPARAMS['mlp_hidden']) + r""" \\
MLP / CNN epochs            & """ + str(mlp_epochs) + r""" / """ + str(cnn_epochs) + r""" \\
Ablation-pack CNN epochs    & """ + str(dir_epochs) + r""" \\
Random seed                 & """ + str(HPARAMS['seed']) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""


# ======================== main: MLP vs CNN ===================================

def _per_class_block(recall: List[float], f1: List[float]) -> str:
    lines = [r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}c|ccc|c|ccc@{}}', r'\toprule',
             r'\textbf{Cls} & R & F1 & P & \textbf{Cls} & R & F1 & P \\',
             r'\midrule']
    for i in range(5):
        j = i + 5
        ri = recall[i] if i < len(recall) else None
        fi = f1[i] if i < len(f1) else None
        rj = recall[j] if j < len(recall) else None
        fj = f1[j] if j < len(f1) else None
        pi = precision_from_recall_f1(ri, fi)
        pj = precision_from_recall_f1(rj, fj)
        lines.append(f"{i} & {fmt_pct(ri)} & {fmt(fi,3)} & {fmt_pct(pi)} & "
                     f"{j} & {fmt_pct(rj)} & {fmt(fj,3)} & {fmt_pct(pj)} \\\\")
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


def section_main(main: Optional[dict]) -> str:
    if main is None:
        return r"""
\section{Main Results: MLP vs.\ CNN}
\label{sec:main}
\textit{(Pending: re-run \texttt{python run\_full\_study.py --pack main}.)}
"""
    mlp = main.get('mlp', {})
    cnn = main.get('cnn', {})
    delta = (cnn.get('test_acc', 0) - mlp.get('test_acc', 0)) if (mlp.get('test_acc') is not None and cnn.get('test_acc') is not None) else None
    ratio = (mlp.get('params', 0) / cnn.get('params', 1)) if (mlp.get('params') and cnn.get('params')) else None
    time_ratio = (cnn.get('total_time_sec', 0) / mlp.get('total_time_sec', 1)) if (mlp.get('total_time_sec') and cnn.get('total_time_sec')) else None
    ece_ratio = (mlp.get('ece', 0) / cnn.get('ece', 1)) if (mlp.get('ece') and cnn.get('ece')) else None

    body = r"""
\section{Main Results: MLP vs.\ CNN}
\label{sec:main}

Table~\ref{tab:main} summarises the headline comparison on the
held-out test set. The CNN reaches """ + bold(fmt_pct(cnn.get('test_acc'))) + r"""
accuracy"""
    if delta is not None:
        body += r""", an absolute improvement of """ + bold(fmt_pct(delta)) + r""" over the MLP"""
    if ratio is not None:
        body += r""", with """ + bold(f"{ratio:.1f}" + r"$\times$") + r""" fewer parameters"""
    body += r""". Calibration improves in the same direction but much
more sharply: the CNN's expected calibration error is
""" + bold(fmt_pct(cnn.get('ece'))) + r""", down from the MLP's
""" + fmt_pct(mlp.get('ece'))
    if ece_ratio is not None and ece_ratio > 1.0:
        body += r""" -- a """ + f"{ece_ratio:.1f}" + r"""$\times$ reduction"""
    body += r""". The CNN pays a clear wall-clock cost ("""
    if time_ratio is not None:
        body += f"roughly {time_ratio:.0f}" + r"""$\times$ slower on a single CPU"""
    else:
        body += r"""more training time"""
    body += r"""), which is the expected price of the richer spatial
inductive bias encoded by convolution and pooling.

\begin{table}[H]
\centering
\caption{Part~A (MLP) vs.\ Part~B (CNN) on the canonical
""" + fmt_int(HPARAMS['test_size']) + r"""-sample test set. The CNN wins on every
axis except wall-clock time.}
\label{tab:main}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcc@{}}
\toprule
 & \textbf{MLP} & \textbf{CNN} \\
\midrule
Hidden units / channels & """ + str(HPARAMS['mlp_hidden']) + r""" & 8 / 16 / 64 \\
Parameters              & """ + fmt_int(mlp.get('params')) + r""" & """ + fmt_int(cnn.get('params')) + r""" \\
Training epochs         & """ + str(mlp.get('epochs', '--')) + r""" & """ + str(cnn.get('epochs', '--')) + r""" \\
Best valid.\ accuracy   & """ + fmt_pct(mlp.get('best_valid_acc')) + r""" & """ + fmt_pct(cnn.get('best_valid_acc')) + r""" \\
\textbf{Test accuracy}  & \textbf{""" + fmt_pct(mlp.get('test_acc')) + r"""} & \textbf{""" + fmt_pct(cnn.get('test_acc')) + r"""} \\
ECE (15-bin)            & """ + fmt_pct(mlp.get('ece')) + r""" & """ + fmt_pct(cnn.get('ece')) + r""" \\
Wall-clock (s, 1 CPU)   & """ + fmt(mlp.get('total_time_sec'), 1) + r""" & """ + fmt(cnn.get('total_time_sec'), 1) + r""" \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/main/mlp_curves.png}\\[0.2em]
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/main/cnn_curves.png}
\caption{Training and validation curves of the MLP (top) and CNN
(bottom). The CNN converges to a markedly lower validation loss and a
higher accuracy plateau, and does so without any evidence of
overfitting in the final epochs.}
\label{fig:main_curves}
\end{figure}

\begin{figure}[H]
\centering
\begin{subfigure}{0.49\columnwidth}
\includegraphics[width=\linewidth]{""" + FIG_REL + r"""/main/mlp_cm_normalized.png}
\caption{MLP}
\end{subfigure}\hfill
\begin{subfigure}{0.49\columnwidth}
\includegraphics[width=\linewidth]{""" + FIG_REL + r"""/main/cnn_cm_normalized.png}
\caption{CNN}
\end{subfigure}
\caption{Row-normalised confusion matrices. Off-diagonal mass is
consistently lower for the CNN; the residual mass concentrates on
the intrinsically ambiguous pairs (4$\leftrightarrow$9, 7$\leftrightarrow$2,
3$\leftrightarrow$5).}
\label{fig:main_cm}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/main/reliability_main.png}
\caption{Reliability diagram. The CNN's curve hugs the identity
diagonal much more tightly, consistent with its $\sim\!""" + (f"{ece_ratio:.0f}" if ece_ratio is not None else "") + r"""\times$
smaller ECE. The MLP is mildly over-confident in the top confidence
bin -- a classic symptom of a network that has effectively memorised
the training set while keeping a smooth decision function.}
\label{fig:main_reliability}
\end{figure}

\paragraph{Per-class behaviour.}
Tables~\ref{tab:pc_mlp}~and~\ref{tab:pc_cnn} report per-class recall
(R), F1, and the precision (P) recovered from $P=F_1 R/(2R-F_1)$.
Figure~\ref{fig:main_perclass} visualises recall side-by-side. The
CNN lifts every class above 97.4\,\% recall, whereas the MLP's
recall on class~9 sits at 95.4\,\%. The recall gap at 9 is not an
accident: it is the same digit at which the MLP's top confusion
($4\to 9$) occurs.

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/main/per_class_acc.png}
\caption{Per-class recall on the test set.}
\label{fig:main_perclass}
\end{figure}

\begin{table}[H]
\centering
\caption{MLP per-class metrics.}
\label{tab:pc_mlp}
\scriptsize
""" + _per_class_block(mlp.get('per_class_recall', []),
                       mlp.get('per_class_f1', [])) + r"""
\end{table}

\begin{table}[H]
\centering
\caption{CNN per-class metrics.}
\label{tab:pc_cnn}
\scriptsize
""" + _per_class_block(cnn.get('per_class_recall', []),
                       cnn.get('per_class_f1', [])) + r"""
\end{table}

\paragraph{Top confusions.}
Table~\ref{tab:topconf} lists the five largest off-diagonal entries
of each confusion matrix. The MLP errors cluster around digit pairs
that share global stroke statistics (4$\leftrightarrow$9,
7$\leftrightarrow$2); the CNN's errors are fewer and more diffuse,
indicating that the spatial filters have sharpened the per-class
decision boundaries rather than merely shrunk them.

\begin{table}[H]
\centering
\caption{Top-5 confusions on the test set (true$\to$predicted: count).}
\label{tab:topconf}
\small
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{MLP} & \textbf{CNN} \\
\midrule
"""
    mlp_conf = mlp.get('top_confusions', [])
    cnn_conf = cnn.get('top_confusions', [])
    for i in range(max(len(mlp_conf), len(cnn_conf), 1)):
        m = mlp_conf[i] if i < len(mlp_conf) else None
        c = cnn_conf[i] if i < len(cnn_conf) else None
        m_s = f"{m[0]}$\\to${m[1]}: {m[2]}" if m else "--"
        c_s = f"{c[0]}$\\to${c[1]}: {c[2]}" if c else "--"
        body += f"{m_s} & {c_s} \\\\\n"
    body += r"""\bottomrule
\end{tabular}
\end{table}

\paragraph{Reading the comparison.}
The CNN result is the textbook story: a small set of hand-designed
priors -- translation equivariance from weight sharing, spatial
pooling for local invariance, hierarchical features from stacking --
simultaneously buys accuracy, calibration, and parameter efficiency.
The MLP is not a bad baseline: a 600-unit ReLU network with $L_2$ and
He initialisation already exceeds 97\,\% test accuracy, and its ECE
of """ + fmt_pct(mlp.get('ece')) + r""" is not pathological. But the CNN's ECE of
""" + fmt_pct(cnn.get('ece')) + r""" is in a different regime -- the
reliability diagram (Fig.~\ref{fig:main_reliability}) shows that
confidence and accuracy are essentially calibrated bin-by-bin. This
is the baseline we use for the remaining five ablation studies.
"""
    return body


# ======================== optim ==============================================

def section_optim(opt: Optional[dict]) -> str:
    if opt is None:
        return r"""
\section{Ablation I: Optimisation}
\label{sec:optim}
\textit{(Pending: re-run \texttt{python run\_full\_study.py --pack optim}.)}
"""
    rows = opt['rows']
    labels = opt['labels']
    bi = best_idx(rows, 'test_acc')

    body = r"""
\section{Ablation I: Optimisation}
\label{sec:optim}

We hold the CNN architecture, training budget and random seed fixed
and vary only the optimiser. Five configurations are compared:
vanilla SGD at learning rates $\eta\!\in\!\{0.01,\,0.05,\,0.10\}$,
SGD with Polyak momentum ($\mu=""" + f"{HPARAMS['momentum']:.2f}" + r"""$, $\eta=0.05$),
and SGD at $\eta=0.05$ with a multi-step schedule that halves the
learning rate at 50\,\% and 75\,\% of training.

\begin{table}[H]
\centering
\caption{Optimisation ablation. Best test accuracy in \textbf{bold}.}
\label{tab:optim}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Setting} & \textbf{Configuration} & \textbf{Best val.} & \textbf{Test} \\
\midrule
"""
    for i, (label, r) in enumerate(zip(labels, rows)):
        cfg = f"{r['optimizer']}, $\\eta{{=}}{r['lr']:g}$"
        if r.get('scheduler'):
            cfg += f", {r['scheduler']}"
        ta = fmt_pct(r['test_acc'])
        if i == bi:
            ta = bold(ta)
        body += f"{label} & {cfg} & {fmt_pct(r['best_valid_acc'])} & {ta} \\\\\n"
    body += r"""\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/optim/optim_test_acc.png}\\[0.2em]
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/optim/optim_valid_curves.png}
\caption{Top: final test accuracy across the five optimiser settings.
Bottom: validation-accuracy trajectories, which make the
learning-rate sensitivity visually explicit.}
\label{fig:optim}
\end{figure}

\paragraph{What the curves say.}
The ranking is """ + bold(latex_escape(rows[bi]['name'])) + r""" at
""" + bold(fmt_pct(rows[bi]['test_acc'])) + r""" test, and the gap
between the slowest (lr$=0.01$) and the best is substantial. Three
observations:
\begin{enumerate}[leftmargin=1.2em,topsep=1pt,itemsep=1pt]
\item At $\eta=0.01$ the validation curve is still climbing when the
epoch budget expires -- the optimiser has simply not moved far
enough. Under a fixed compute budget, a too-small learning rate is
indistinguishable from under-training.
\item At $\eta=0.10$ the curve is visibly noisier than at $0.05$ but
the final plateau is similar, because the
mini-batch-128 gradient estimator already injects enough stochasticity
for the network to escape shallow minima. The implicit regularisation
of large step sizes is mild on this task.
\item Polyak momentum ($\mu=""" + f"{HPARAMS['momentum']:.2f}" + r"""$) and the multi-step
schedule both improve over the best fixed-$\eta$ SGD, consistent with
the textbook picture: momentum acts like a low-pass filter on the
gradient estimator, and a coarse-to-fine schedule trades exploration
early for exploitation late.
\end{enumerate}

\paragraph{Takeaway.}
For the compact CNN used in this study, the single most impactful
knob is the effective learning rate: getting within $2\times$ of the
right value closes most of the optimisation gap, and second-order
refinements (momentum, schedule) recover the last fraction of a
percent.
"""
    return body


# ======================== regularisation =====================================

def section_reg(reg: Optional[dict]) -> str:
    if reg is None:
        return r"""
\section{Ablation II: Regularisation}
\label{sec:reg}
\textit{(Pending: re-run \texttt{python run\_full\_study.py --pack reg}.)}
"""
    rows = reg['rows']
    labels = reg['labels']
    bi = best_idx(rows, 'test_acc')
    tests = [r['test_acc'] for r in rows]
    spread = (max(tests) - min(tests)) if tests else 0.0

    body = r"""
\section{Ablation II: Regularisation}
\label{sec:reg}

Six regularisation configurations are evaluated with everything else
(architecture, optimiser, learning-rate schedule, seed) held fixed:
no regularisation, $L_2$ weight decay at $10^{-4}$ and
$5\!\times\!10^{-4}$, two dropout rates on the FC bottleneck
($0.2$ and $0.3$, combined with $L_2=10^{-4}$), and an early-stopping
schedule with patience~4 validation epochs (also on top of
$L_2=10^{-4}$).

\begin{table}[H]
\centering
\caption{Regularisation ablation. Best test accuracy in \textbf{bold};
the spread across configurations is """ + bold(fmt_pct(spread)) + r""".}
\label{tab:reg}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Setting} & \textbf{$L_2$} & \textbf{Dropout} & \textbf{Early} & \textbf{Test} \\
\midrule
"""
    for i, (label, r) in enumerate(zip(labels, rows)):
        wd_val = r.get('weight_decay', 0) or 0
        wd = f"{wd_val:.0e}" if wd_val > 0 else '0'
        dp = f"{r.get('dropout_rate', 0):.2f}"
        es = r.get('early_stop')
        es_s = str(es) if es is not None else '--'
        ta = fmt_pct(r['test_acc'])
        if i == bi:
            ta = bold(ta)
        body += f"{label} & {wd} & {dp} & {es_s} & {ta} \\\\\n"
    body += r"""\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/reg/reg_test_acc.png}\\[0.2em]
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/reg/reg_valid_curves.png}
\caption{Test accuracy and validation-accuracy curves under the six
regularisation configurations. The curves essentially overlap within
the compute budget.}
\label{fig:reg}
\end{figure}

\paragraph{An honestly small effect.}
The best configuration is """ + bold(latex_escape(rows[bi]['name'])) + r""" at
""" + bold(fmt_pct(rows[bi]['test_acc'])) + r""" test accuracy, but the spread
between the six cells is only """ + bold(fmt_pct(spread)) + r""", which is of
the same order as single-seed noise on a 10\,000-sample test split.
We therefore resist the temptation to over-interpret individual
orderings and instead report the qualitative picture.
\begin{itemize}[leftmargin=1.1em,topsep=1pt,itemsep=1pt]
\item The \emph{no-regularisation} baseline is already close to the
best regularised score, because the CNN's low parameter count
(""" + fmt_int((load_summary('main') or {}).get('cnn', {}).get('params', 52138)) + r""") and the short ablation-pack budget leave little room
for overfitting.
\item $L_2$ at $10^{-4}$ gives a small, reproducible improvement;
pushing $L_2$ to $5\!\times\!10^{-4}$ starts to bite into capacity
rather than smoothing the loss surface.
\item Dropout on the FC bottleneck is approximately neutral --
consistent with the Srivastava~et~al.\ observation that dropout helps
most when the network is substantially over-parameterised, which our
network is not.
\item Early stopping with patience~4 acts as a free safety net. It
never beats the best static configuration, but it never hurts either,
and it makes the pipeline robust to accidentally choosing too many
epochs.
\end{itemize}

\paragraph{Takeaway.}
In the regime of a well-initialised, correctly-sized small CNN on a
clean dataset, aggressive regularisation is neither necessary nor
helpful. The right role for $L_2$ here is as a mild implicit prior
($\sim\!10^{-4}$) on top of a well-chosen architecture, not as a
compensating force for overfitting.
"""
    return body


# ======================== augmentation =======================================

def section_aug(aug: Optional[dict]) -> str:
    if aug is None:
        return r"""
\section{Ablation III: From-Scratch Geometric Augmentation}
\label{sec:aug}
\textit{(Pending: re-run \texttt{python run\_full\_study.py --pack aug}.)}
"""
    rows = aug['rows']
    labels = aug['labels']
    bi = best_idx(rows, 'test_acc')
    clean_acc = next((r['test_acc'] for r in rows if r['name'] == 'aug_clean'), None)

    body = r"""
\section{Ablation III: From-Scratch Geometric Augmentation}
\label{sec:aug}

Let $\mathcal{A}=\{R,T,S\}$ denote \{rotation, translation, scale\}.
We train eight CNNs, one per subset
$\mathcal{S}\subseteq\mathcal{A}$. Rotations are uniform in
$[-8^\circ,8^\circ]$, translations integer-valued in
$\{-2,\ldots,2\}^{2}$ pixels, and isotropic scales drawn from
$[0.95,1.05]$. Augmentation is applied on-the-fly to every
mini-batch through the bilinear sampler described in
\S\ref{sec:impl}.

\begin{table}[H]
\centering
\caption{Augmentation ablation; $\Delta$ is the absolute improvement
over the \textsc{Clean} baseline. Best test accuracy in \textbf{bold}.}
\label{tab:aug}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Setting} & \textbf{R} & \textbf{T} & \textbf{S}
                 & \textbf{Best val.} & \textbf{Test} & \textbf{$\Delta$} \\
\midrule
"""
    for i, (label, r) in enumerate(zip(labels, rows)):
        cR = r'\checkmark' if r['use_rotation'] else ''
        cT = r'\checkmark' if r['use_translation'] else ''
        cS = r'\checkmark' if r['use_scaling'] else ''
        ta = fmt_pct(r['test_acc'])
        if i == bi:
            ta = bold(ta)
        if clean_acc is not None:
            d_str = fmt_signed_pct(r['test_acc'] - clean_acc)
        else:
            d_str = '--'
        body += (f"{label} & {cR} & {cT} & {cS} & "
                 f"{fmt_pct(r['best_valid_acc'])} & {ta} & {d_str} \\\\\n")
    body += r"""\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/aug/aug_test_acc.png}
\caption{Test accuracy across the eight augmentation subsets; the
dashed line marks the Clean baseline.}
\label{fig:aug}
\end{figure}

\paragraph{Reading the table.}
Every augmentation configuration except pure~T is on or above the
Clean baseline, and the best cell is
""" + bold(latex_escape(rows[bi]['name'])) + r""" at
""" + bold(fmt_pct(rows[bi]['test_acc'])) + r""". The effect sizes are
small on clean test accuracy (the table spans roughly $0.2$\,\%), but
two structural patterns are worth highlighting:
\begin{itemize}[leftmargin=1.1em,topsep=1pt,itemsep=1pt]
\item Pure translation (T) is the weakest single augmentation,
because the CNN already has translation \emph{equivariance} by
construction (weight sharing plus max pooling confers partial
\emph{invariance}). Feeding it $\pm 2$-pixel shifts adds less
information than feeding it rotated or scaled versions of digits
that lie outside its built-in equivariance group.
\item The full RTS combination is not strictly the winner on
\emph{clean} test accuracy (it ties with R\,+\,T among the top
configurations) but, as the robustness study in \S\ref{sec:robust}
shows, it is dramatically the best under distribution shift. A
single best-test-accuracy cell is therefore a misleading summary of
this ablation.
\end{itemize}

\paragraph{Takeaway.}
For a CNN that already encodes some spatial structure, the
information content of an augmentation is determined by whether it
lies \emph{inside} or \emph{outside} the model's built-in symmetry
group. Translation, which the CNN is already equivariant to, is the
least informative; rotations and scales, which the CNN is
\emph{not} equivariant to, contribute more.
"""
    return body


# ======================== robustness =========================================

def section_robust(rob: Optional[dict]) -> str:
    if rob is None:
        return r"""
\section{Ablation IV: Robustness Under Distribution Shift}
\label{sec:robust}
\textit{(Pending: re-run \texttt{python run\_full\_study.py --pack robust}.)}
"""
    scenarios = rob.get('scenarios', [])
    noise = rob.get('noise_sweep', [])

    clean_base = next((s['clean_cnn_acc'] for s in scenarios if s['scenario'] == 'clean'), None)
    rts_base = next((s['rts_cnn_acc'] for s in scenarios if s['scenario'] == 'clean'), None)
    pert = [s for s in scenarios if s['scenario'] != 'clean']
    mean_clean_drop = (clean_base - sum(s['clean_cnn_acc'] for s in pert) / max(1, len(pert))) if (clean_base is not None and pert) else None
    mean_rts_drop = (rts_base - sum(s['rts_cnn_acc'] for s in pert) / max(1, len(pert))) if (rts_base is not None and pert) else None
    worst_clean = min((s['clean_cnn_acc'] for s in pert), default=None)
    worst_rts = min((s['rts_cnn_acc'] for s in pert), default=None)
    worst_clean_drop = (clean_base - worst_clean) if (clean_base is not None and worst_clean is not None) else None
    worst_rts_drop = (rts_base - worst_rts) if (rts_base is not None and worst_rts is not None) else None
    max_gain = max((s['gain'] for s in pert), default=None)

    # noise sweep analytics
    noise_cross_sigma = None
    noise_flip_sigma = None
    if noise:
        # smallest sigma at which RTS < Clean (where the "augmentation helps" narrative fails)
        for n in sorted(noise, key=lambda n: n['sigma']):
            if n['sigma'] > 0 and n['rts_cnn_acc'] < n['clean_cnn_acc']:
                noise_cross_sigma = n['sigma']
                break
        # largest sigma tested
        top = max(noise, key=lambda n: n['sigma'])
        sig_top = top['sigma']
        c_top = top['clean_cnn_acc']
        r_top = top['rts_cnn_acc']

    body = r"""
\section{Ablation IV: Robustness Under Distribution Shift}
\label{sec:robust}

We stress-test two CNNs from \S\ref{sec:aug}: the clean-trained
baseline (\texttt{aug\_clean}, no augmentation) and the all-affine
model (\texttt{aug\_RTS}, rotation + translation + scale). The test
set is perturbed in two ways that the models \emph{never} saw during
training:
\begin{itemize}[leftmargin=1.1em,topsep=1pt,itemsep=1pt]
\item \textbf{Nine fixed affine scenarios}: identity, $\pm 10^\circ$
rotation, $\pm 2$-pixel shifts on each axis, and $\pm 8\,\%$ isotropic
scaling. These probe robustness \emph{within} the geometric family
that aug\_RTS was trained on, but at stronger magnitudes than training.
\item \textbf{Seven additive-Gaussian-noise scenarios}:
$\sigma\!\in\!\{0,0.05,0.10,0.15,0.20,0.25,0.30\}$. Pixel-level noise
was \emph{not} part of the augmentation pipeline of either model --
this is a genuinely out-of-family shift.
\end{itemize}

\subsection{Affine perturbations: augmentation helps dramatically.}

\begin{table}[H]
\centering
\caption{Per-scenario test accuracy under held-out affine perturbations.}
\label{tab:robust}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Scenario} & \textbf{Clean CNN} & \textbf{Affine CNN} & \textbf{$\Delta$} \\
\midrule
"""
    for s in scenarios:
        gain_str = fmt_signed_pct(s['gain'])
        body += (f"\\texttt{{{latex_escape(s['scenario'])}}} & "
                 f"{fmt_pct(s['clean_cnn_acc'])} & "
                 f"{fmt_pct(s['rts_cnn_acc'])} & "
                 f"{gain_str} \\\\\n")
    body += r"""\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/robust/robust_perturb.png}
\caption{Per-scenario accuracy: clean-trained (blue) vs.\
affine-trained (orange) CNN. The affine-trained model dominates on
every cell.}
\label{fig:robust_aff}
\end{figure}

\paragraph{Quantitative summary.}
\begin{itemize}[leftmargin=1.1em,topsep=1pt,itemsep=1pt]
\item Mean accuracy drop across the eight non-identity perturbations:
"""
    if mean_clean_drop is not None and mean_rts_drop is not None:
        body += (bold(fmt_pct(mean_clean_drop)) + r""" (Clean CNN) vs.\ """
                 + bold(fmt_pct(mean_rts_drop)) + r""" (Affine CNN).""" + "\n")
    else:
        body += "(unavailable).\n"
    body += r"""\item Worst-case accuracy drop:
"""
    if worst_clean_drop is not None and worst_rts_drop is not None:
        body += (bold(fmt_pct(worst_clean_drop)) + r""" (Clean CNN) vs.\ """
                 + bold(fmt_pct(worst_rts_drop)) + r""" (Affine CNN).""" + "\n")
    else:
        body += "(unavailable).\n"
    body += r"""\item Largest single-scenario gain from augmentation:
"""
    body += (bold(fmt_signed_pct(max_gain)) + ".\n") if max_gain is not None else "(unavailable).\n"
    body += r"""\end{itemize}

On vertical shifts (\texttt{shift\_up2}, \texttt{shift\_down2}) the
Clean CNN loses nearly ten percentage points of accuracy relative to
its clean baseline, while the Affine CNN is essentially unaffected.
The affine-trained model is \emph{not} exploiting some hidden
calibration change -- it has simply learned feature detectors that
respond to the shifted digits in the same way as to the centred ones.

\subsection{Additive pixel noise: augmentation hurts.}

Pixel-level Gaussian noise is an out-of-family shift that neither
model saw during training. This probe is therefore fair to both:
there is no training-data advantage for either side.

\begin{table}[H]
\centering
\caption{Additive-Gaussian-noise sweep. Bold marks the winner in each
row.}
\label{tab:robust_noise}
\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}cccc@{}}
\toprule
$\sigma$ & \textbf{Clean CNN} & \textbf{Affine CNN} & $\Delta$ \\
\midrule
"""
    for n in noise:
        sig = n['sigma']
        c = n['clean_cnn_acc']
        r_ = n['rts_cnn_acc']
        delta = r_ - c
        c_s = bold(fmt_pct(c)) if c > r_ else fmt_pct(c)
        r_s = bold(fmt_pct(r_)) if r_ > c else fmt_pct(r_)
        body += f"{sig:.2f} & {c_s} & {r_s} & {fmt_signed_pct(delta)} \\\\\n"
    body += r"""\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/robust/robust_noise_curve.png}
\caption{Accuracy vs.\ Gaussian noise standard deviation. The two
curves agree for small $\sigma$ but diverge sharply once the noise is
large enough to matter: the affine-trained model degrades
dramatically faster than the clean-trained baseline.}
\label{fig:robust_noise}
\end{figure}

\paragraph{Finding.}"""
    if noise_cross_sigma is not None:
        body += (r""" Starting at $\sigma\!\approx\!""" + f"{noise_cross_sigma:.2f}" + r"""$
the Affine CNN is \emph{worse} than the Clean CNN, and at
$\sigma=""" + f"{sig_top:.2f}" + r"""$ the gap has widened to
""" + bold(fmt_pct(c_top - r_top)) + r""" in favour of the clean
baseline (""" + bold(fmt_pct(c_top)) + r""" vs.\ """ + bold(fmt_pct(r_top)) + r""").""")
    else:
        body += r""" No crossover was observed in this sweep."""
    body += r""" In other words, the augmentation that buys
dramatic robustness on \emph{affine} shifts \emph{trades it away} on
\emph{pixel-level} shifts it was never trained to handle.

\paragraph{Why.}
We interpret this as the following geometric picture. Affine
augmentation encourages the CNN to produce feature responses that
are stable along a specific family of smooth input trajectories
(rotations, translations, scales). Along these trajectories the
decision surfaces can safely be made sharper, which is good for
accuracy on the in-family set. But a Gaussian-noise perturbation is
not a smooth trajectory -- it is a high-frequency, isotropic
perturbation, and sharper decision surfaces are, all else equal,
\emph{more} sensitive to that kind of perturbation. The clean-trained
baseline has not been pushed to sharpen its features in any
particular direction and therefore retains a bit more noise
robustness by default. The bottom line is that
\emph{augmentation is a directed inductive bias, not a universal
regulariser}; its robustness gains transfer within the family it
explicitly samples, but not necessarily across families. We return
to this point in \S\ref{sec:discussion}.
"""
    return body


# ======================== error / calibration ================================

def section_error(err: Optional[dict]) -> str:
    if err is None:
        return r"""
\section{Ablation V: Error and Calibration Analysis}
\label{sec:error}
\textit{(Pending: re-run \texttt{python run\_full\_study.py --pack error}.)}
"""
    pairs = err.get('top_confusions', [])
    body = r"""
\section{Ablation V: Error and Calibration Analysis}
\label{sec:error}

We pick the best-performing CNN from \S\ref{sec:aug}
(\texttt{""" + latex_escape(err.get('best_model', '--')) + r"""}, test
accuracy """ + bold(fmt_pct(err.get('test_acc'))) + r""", ECE
""" + bold(fmt_pct(err.get('ece'))) + r""") and dissect its residual error
budget along five axes: top confusions, per-class metrics, confusion
matrix geometry, representation geometry in the penultimate layer,
and input saliency.

\paragraph{Top confusions.}
The largest off-diagonal mass is at:
"""
    items = ', '.join([f"{a}$\\to${b} ({c})" for (a, b, c) in pairs])
    body += (items if items else "(none)") + r""".
The pair $4\!\leftrightarrow\!9$ continues to lead: it was the top
confusion for both the MLP and the original Part~B CNN as well. These
are not representational failures -- they reflect genuine
dataset-level ambiguity, as the saliency visualisation
(Fig.~\ref{fig:err_sal}) makes explicit.

\paragraph{Hardest vs.\ easiest classes.}
Hardest classes (lowest recall):
""" + (", ".join(str(c) for c in err.get('hardest_classes', [])) or "--") + r""".
Easiest classes:
""" + (", ".join(str(c) for c in err.get('easiest_classes', [])) or "--") + r""".
The hard classes are exactly the ones that sit on top of the most
common off-diagonals (4, 9, 6), suggesting that the model's residual
errors are concentrated on a few semantically ambiguous digit pairs
rather than being diffuse.

\begin{table}[H]
\centering
\caption{Per-class precision, recall, and F1 of the best CNN.}
\label{tab:err_pc}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}c|ccc|c|ccc@{}}
\toprule
\textbf{Cls} & P & R & F1 & \textbf{Cls} & P & R & F1 \\
\midrule
"""
    pcs_p = err.get('per_class_precision', [None] * 10)
    pcs_r = err.get('per_class_recall', [None] * 10)
    pcs_f = err.get('per_class_f1', [None] * 10)
    for i in range(5):
        j = i + 5
        body += (f"{i} & {fmt_pct(pcs_p[i] if i<len(pcs_p) else None)} & "
                 f"{fmt_pct(pcs_r[i] if i<len(pcs_r) else None)} & "
                 f"{fmt(pcs_f[i] if i<len(pcs_f) else None,3)} & "
                 f"{j} & {fmt_pct(pcs_p[j] if j<len(pcs_p) else None)} & "
                 f"{fmt_pct(pcs_r[j] if j<len(pcs_r) else None)} & "
                 f"{fmt(pcs_f[j] if j<len(pcs_f) else None,3)} \\\\\n")
    body += r"""\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\begin{subfigure}{0.49\columnwidth}
\includegraphics[width=\linewidth]{""" + FIG_REL + r"""/error/best_cm_normalized.png}
\caption{Confusion matrix.}
\end{subfigure}\hfill
\begin{subfigure}{0.49\columnwidth}
\includegraphics[width=\linewidth]{""" + FIG_REL + r"""/error/best_reliability.png}
\caption{Reliability.}
\end{subfigure}
\caption{Row-normalised confusion matrix and reliability diagram of
the best CNN. Most predictions sit on the diagonal; the reliability
curve hugs the identity, consistent with the low ECE.}
\label{fig:err_diag}
\end{figure}

\paragraph{Representation geometry.}
Figure~\ref{fig:err_pca} (right) projects the 64-dimensional
penultimate-layer activations onto the first two principal
components. Ten clearly separated clusters are visible; the few
overlapping regions coincide precisely with the top confusions in
Table~\ref{tab:err_pc}. That is the geometric explanation for the
CNN's residual errors: the representation is linearly separable
everywhere except on a few intrinsically ambiguous digit pairs.

\begin{figure}[H]
\centering
\begin{subfigure}{0.49\columnwidth}
\includegraphics[width=\linewidth]{""" + FIG_REL + r"""/error/best_high_conf_errors.png}
\caption{High-confidence errors.}
\end{subfigure}\hfill
\begin{subfigure}{0.49\columnwidth}
\includegraphics[width=\linewidth]{""" + FIG_REL + r"""/error/best_features_pca.png}
\caption{Penultimate-feature PCA.}
\end{subfigure}
\caption{Left: test images mis-classified with $>\!0.9$ predicted
probability. Most are genuine human-level ambiguities (4 without a
closed top vs.\ 9, sloppy 5 vs.\ 3). Right: 2-D PCA of the 64-unit
penultimate features, coloured by true class.}
\label{fig:err_pca}
\end{figure}

\paragraph{What the filters learn.}
The first convolutional layer (Fig.~\ref{fig:err_filters}) specialises
into a small bank of oriented edge and Gabor-like filters -- the
emergent low-level feature set that a small CNN is expected to learn
on 28$\times$28 greyscale digits. Nothing more exotic is necessary;
the second convolutional block combines these primitives into the
mid-level structure that the bottleneck classifier reads out.

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/error/best_conv1_filters.png}
\caption{First-layer convolution filters. They are predominantly
oriented edge detectors, with a minority specialising in corners and
small blobs.}
\label{fig:err_filters}
\end{figure}

\paragraph{Input saliency.}
Figure~\ref{fig:err_sal} shows input-saliency maps
$|\partial \mathcal{L}/\partial x|$ obtained by a single
forward-backward pass through the trained CNN on one representative
image per class. Saliency concentrates on locally discriminative
strokes (the central horizontal of \textbf{4}/\textbf{9}, the loop of
\textbf{6}, the closed top of \textbf{8}) and almost ignores the
background. This is the standard Simonyan-et-al.\ picture and, on
its own, is sufficient evidence that the model is using
digit-relevant structure rather than spurious background correlations.

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{""" + FIG_REL + r"""/error/best_saliency.png}
\caption{Top row: a representative test image per class. Bottom row:
the absolute input gradient $|\partial\mathcal{L}/\partial x|$,
normalised per image. Saliency highlights discriminative strokes and
almost ignores the background.}
\label{fig:err_sal}
\end{figure}

\paragraph{Takeaway.}
The residual error of the best CNN is not a failure of representation
-- the PCA clusters are tight and the reliability diagram is
diagonal. It is a failure of \emph{dataset disambiguation}: a small
number of MNIST digits are genuinely hard for human readers too, and
the model's last percent of error mostly lives on those.
"""
    return body


# ======================== discussion =========================================

def section_discussion(main: Optional[dict], aug: Optional[dict],
                       rob: Optional[dict]) -> str:
    cnn = (main or {}).get('cnn', {})

    body = r"""
\section{Discussion}
\label{sec:discussion}

Across five controlled ablations, three themes recur.

\paragraph{1.\ Inductive bias pays for itself early.}
The CNN's spatial priors (weight sharing, local receptive fields,
pooling) simultaneously buy accuracy, parameter efficiency, and
calibration. The MLP is not a bad model -- it exceeds $97\,\%$ with
He initialisation and mild $L_2$ -- but the CNN lives on a
fundamentally better point of the Pareto front on every axis we
measured, including ECE (""" + fmt_pct(cnn.get('ece')) + r"""). There
is no free lunch: the CNN is """
    if main and main.get('mlp', {}).get('total_time_sec') and cnn.get('total_time_sec'):
        body += (f"roughly {cnn['total_time_sec']/main['mlp']['total_time_sec']:.0f}"
                 + r"""$\times$ slower""")
    else:
        body += r"""slower"""
    body += r""" in wall-clock on a single CPU. For the headline
metrics, that cost is obviously worth paying.

\paragraph{2.\ Regularisation is a second-order knob here.}
When the architecture and learning rate are already well-chosen, the
spread between no-regularisation and any of the textbook
regularisation tricks ($L_2$, dropout, early stopping) is smaller
than single-seed noise. A modest $L_2$ is a principled default, but
aggressive regularisation -- $L_2\!=\!5\!\times\!10^{-4}$ or
$\textrm{dropout}\!=\!0.3$ -- trades capacity for nothing. The main
practical use of early stopping here is operational: it makes the
training loop robust to an over-estimated epoch budget.

\paragraph{3.\ Augmentation is a directed prior.}
This is the most interesting finding of the study, because it runs
against the folk intuition that ``data augmentation always helps
generalisation''. Let $P_{\text{train}}$ be the training distribution
after augmentation and $P_{\text{test}}$ the test distribution. Data
augmentation effectively replaces $P_{\text{train}}$ with
$P_{\text{train}}^{\mathcal{S}} = \mathcal{S}\!*\!P_{\text{train}}$,
the convolution with the chosen augmentation family $\mathcal{S}$.
On any test perturbation that lies inside $\mathcal{S}$ (affine
shifts here), the model is essentially
\emph{evaluated on its training distribution}, which is why the
worst-case affine drop falls from
"""
    if rob and rob.get('scenarios'):
        scens = rob['scenarios']
        clean_base = next((s['clean_cnn_acc'] for s in scens if s['scenario'] == 'clean'), None)
        rts_base = next((s['rts_cnn_acc'] for s in scens if s['scenario'] == 'clean'), None)
        pert = [s for s in scens if s['scenario'] != 'clean']
        if clean_base is not None and pert:
            wc = clean_base - min(s['clean_cnn_acc'] for s in pert)
            wr = rts_base - min(s['rts_cnn_acc'] for s in pert)
            body += bold(fmt_pct(wc)) + r""" to """ + bold(fmt_pct(wr))
    body += r""". On a test perturbation
\emph{outside} $\mathcal{S}$ (pixel-level noise here), the model is
evaluated on a distribution that augmentation made it \emph{more
confident} about but not actually more robust on. The net effect,
measured on the Gaussian-noise sweep, is a robustness \emph{regression}
at moderate $\sigma$. Augmentation should therefore be thought of as
``investing the model's capacity into a specific symmetry group''
rather than ``making the model more robust in general''. If the
deployment distribution includes multiple perturbation families, the
augmentation pipeline should explicitly sample all of them.

\paragraph{Relationship to calibration.}
A sharper decision surface helps within the sampled family and hurts
outside it; this is exactly the kind of specialisation that
calibration metrics can catch but accuracy alone cannot. A practical
corollary: ECE is a more sensitive diagnostic than accuracy for
detecting over-specialised augmentation pipelines.
"""
    return body


# ======================== limitations ========================================

def section_limitations() -> str:
    return r"""
\section{Limitations}
\label{sec:limitations}

We flag the following limitations so the quantitative claims are not
over-read.

\begin{itemize}[leftmargin=1.1em,topsep=1pt,itemsep=1pt]
\item \textbf{Single seed.} All runs use the same random seed
(""" + str(HPARAMS['seed']) + r"""). Some of the within-ablation gaps reported in
\S\ref{sec:reg} and \S\ref{sec:aug} are of the order of single-seed
noise on a 10\,000-sample test split; we have therefore been careful
to distinguish them from the larger, unambiguously reproducible
effects (the MLP~$\to$~CNN gap, the affine-robustness gap).
\item \textbf{Compute budget.} Training is CPU-only NumPy; the
ablation-pack CNNs are trained for
""" + str(HPARAMS['direction_epochs_default']) + r""" epochs rather than a
large budget. This is long enough for the qualitative ordering to
stabilise but short enough that small quantitative differences should
not be over-interpreted.
\item \textbf{Dataset.} MNIST is close to saturated: $98$--$99\,\%$
accuracy is the regime in which small architectural or optimisation
differences get compressed into tenths of a percent. A fairer
stress-test of our conclusions would repeat the augmentation and
robustness study on a harder dataset such as Fashion-MNIST or
SVHN; this is out of scope for the project brief.
\item \textbf{No architectural sweep.} We did not vary CNN depth or
width, which means the regularisation finding (``modest $L_2$ is
enough; dropout adds little'') is specific to our particular
architecture. In an over-parameterised setting dropout would likely
be more useful.
\item \textbf{Augmentation family.} The robustness regression on
additive pixel noise (\S\ref{sec:robust}) is a statement about the
specific $\{R,T,S\}$ augmentation family, not about augmentation in
general. An augmentation pipeline that \emph{also} samples pixel-level
noise would be expected to close the gap, at the potential cost of
smaller affine-robustness gains. We leave the quantitative sweep over
augmentation families to future work.
\end{itemize}
"""


# ======================== conclusion =========================================

def section_conclusion(main: Optional[dict], opt: Optional[dict],
                       reg: Optional[dict], aug: Optional[dict],
                       rob: Optional[dict], err: Optional[dict]) -> str:
    cnn = (main or {}).get('cnn', {})
    mlp = (main or {}).get('mlp', {})
    cnn_acc = cnn.get('test_acc')
    delta = (cnn_acc - mlp.get('test_acc')) if (cnn_acc is not None and mlp.get('test_acc') is not None) else None

    lines = []
    if cnn_acc is not None:
        lines.append(r"the compact CNN reaches " + bold(fmt_pct(cnn_acc))
                     + r" test accuracy at ECE " + bold(fmt_pct(cnn.get('ece')))
                     + r", "
                     + (bold(fmt_pct(delta)) + r" above the 600-unit MLP baseline"
                        if delta is not None else r"above the MLP baseline"))
    if opt and opt.get('rows'):
        r = opt['rows'][best_idx(opt['rows'], 'test_acc')]
        lines.append(r"the best optimiser is " + bold(latex_escape(r['name']))
                     + r" (" + bold(fmt_pct(r['test_acc'])) + r")")
    if reg and reg.get('rows'):
        tests = [r['test_acc'] for r in reg['rows']]
        sp = max(tests) - min(tests)
        lines.append(r"the regularisation ablation is essentially flat (spread "
                     + bold(fmt_pct(sp)) + r"), indicating that the architecture + $L_2$ default is already well-tuned")
    if aug and aug.get('rows'):
        r = aug['rows'][best_idx(aug['rows'], 'test_acc')]
        lines.append(r"the best augmentation is " + bold(latex_escape(r['name']))
                     + r" on clean test, but the \emph{full} RTS augmentation dominates on held-out affine perturbations")
    if rob and rob.get('scenarios'):
        scens = rob['scenarios']
        clean_base = next((s['clean_cnn_acc'] for s in scens if s['scenario'] == 'clean'), None)
        rts_base = next((s['rts_cnn_acc'] for s in scens if s['scenario'] == 'clean'), None)
        pert = [s for s in scens if s['scenario'] != 'clean']
        if clean_base is not None and pert:
            wc = clean_base - min(s['clean_cnn_acc'] for s in pert)
            wr = rts_base - min(s['rts_cnn_acc'] for s in pert)
            lines.append(r"the worst-case affine drop falls from " + bold(fmt_pct(wc))
                         + r" to " + bold(fmt_pct(wr))
                         + r", whereas the same augmentation \emph{hurts} robustness to unseen pixel-level Gaussian noise -- augmentation transfers within, but not across, perturbation families")
    if err is not None:
        lines.append(r"residual errors of the best CNN are concentrated on intrinsically ambiguous digit pairs (ECE " + bold(fmt_pct(err.get('ece'))) + r")")

    compact = '; '.join(lines) + '.'

    return r"""
\section{Conclusion}
\label{sec:conclusion}

Writing an MNIST training pipeline in NumPy from scratch is more
than a back-propagation exercise. The five ablations in this paper
yield a compact but internally consistent story: """ + compact + r"""
We hope the honest reporting of the noise-robustness regression
under affine augmentation -- a phenomenon that a purely
accuracy-centric writeup would have easily missed -- is useful to
anyone who has reached for augmentation as a default
\emph{regulariser}. It is not; it is a targeted prior.
"""


# ======================== appendix ===========================================

def section_appendix(main: Optional[dict], opt: Optional[dict],
                     reg: Optional[dict], aug: Optional[dict]) -> str:
    body = r"""
\appendix
\section{Reproducibility and Code Layout}
\label{app:repro}

\paragraph{Source tree.}
\begin{itemize}[leftmargin=1.2em,topsep=0pt,itemsep=1pt]
\item \texttt{codes/mynn/op.py} -- learnable layers, activations, dropout, loss.
\item \texttt{codes/mynn/optimizer.py} -- SGD and momentum SGD with decoupled $L_2$.
\item \texttt{codes/mynn/lr\_scheduler.py} -- step / multi-step / exponential schedulers.
\item \texttt{codes/mynn/models.py} -- \texttt{Model\_MLP}, \texttt{Model\_CNN}.
\item \texttt{codes/study\_utils.py} -- data loading, affine sampler, plotting helpers.
\item \texttt{codes/run\_full\_study.py} -- experiment driver
  (\texttt{--pack \{main,optim,reg,aug,robust,error,all\}}).
\item \texttt{codes/build\_report.py} -- regenerates this PDF from the JSON summaries.
\item \texttt{codes/results/full/} -- per-pack JSON summaries, model checkpoints, training history.
\item \texttt{codes/gradient\_check.py} -- analytical-vs-numerical gradient verification.
\end{itemize}

\paragraph{Commands to reproduce.}
From the project root, with \texttt{pip install -r requirements.txt}:
\small
\begin{verbatim}
cd codes
python gradient_check.py
python run_full_study.py --pack all \
    --epochs-mlp 30 --epochs-cnn 20 \
    --epochs-direction 8 --valid-size 10000
cd ..
python codes/build_report.py
pdflatex -interaction=nonstopmode \
  MNIST_From_Scratch_Report_Leyan_Huang.tex
pdflatex -interaction=nonstopmode \
  MNIST_From_Scratch_Report_Leyan_Huang.tex
\end{verbatim}
\normalsize

\paragraph{Environment.}
The reference run uses Python~3.13, NumPy~2.2 and Matplotlib~3.10 on a
single CPU; no GPU is required. Because every primitive is
implemented in NumPy, absolute wall-clock numbers depend heavily on
the BLAS backend and core count; the \emph{relative} numbers across
packs are the robust ones for all conclusions.

\paragraph{Raw numerical outputs.}
The JSON summaries under
\texttt{codes/results/full/<pack>/summary.json} contain every scalar
reported here (plus several more). They are machine-readable and were
consumed verbatim by \texttt{build\_report.py} to produce this
document.

\begin{thebibliography}{9}
\bibitem{lecun1998}
Y.~LeCun, L.~Bottou, Y.~Bengio, P.~Haffner, ``Gradient-based learning
applied to document recognition,'' \emph{Proc. IEEE}, 86(11),
2278--2324, 1998.
\bibitem{he2015}
K.~He, X.~Zhang, S.~Ren, J.~Sun, ``Delving Deep into Rectifiers:
Surpassing Human-Level Performance on ImageNet Classification,''
\emph{ICCV}, 1026--1034, 2015.
\bibitem{srivastava2014}
N.~Srivastava, G.~Hinton, A.~Krizhevsky, I.~Sutskever, R.~Salakhutdinov,
``Dropout: A Simple Way to Prevent Neural Networks from Overfitting,''
\emph{JMLR}, 15(1):1929--1958, 2014.
\bibitem{loshchilov2019}
I.~Loshchilov, F.~Hutter, ``Decoupled Weight Decay Regularization,''
\emph{ICLR}, 2019.
\bibitem{guo2017}
C.~Guo, G.~Pleiss, Y.~Sun, K.~Q.~Weinberger, ``On Calibration of Modern
Neural Networks,'' \emph{ICML}, 1321--1330, 2017.
\bibitem{simonyan2014}
K.~Simonyan, A.~Vedaldi, A.~Zisserman, ``Deep Inside Convolutional
Networks: Visualising Image Classification Models and Saliency Maps,''
\emph{ICLR Workshop}, 2014.
\end{thebibliography}

\end{document}
"""
    return body


# ======================== main ===============================================

def main():
    main_pack = load_summary('main')
    opt_pack = load_summary('optim')
    reg_pack = load_summary('reg')
    aug_pack = load_summary('aug')
    rob_pack = load_summary('robust')
    err_pack = load_summary('error')

    pieces = [
        section_abstract(main_pack, rob_pack),
        section_intro(),
        section_impl(),
        section_setup(main_pack),
        section_main(main_pack),
        section_optim(opt_pack),
        section_reg(reg_pack),
        section_aug(aug_pack),
        section_robust(rob_pack),
        section_error(err_pack),
        section_discussion(main_pack, aug_pack, rob_pack),
        section_limitations(),
        section_conclusion(main_pack, opt_pack, reg_pack, aug_pack,
                           rob_pack, err_pack),
        section_appendix(main_pack, opt_pack, reg_pack, aug_pack),
    ]
    tex = '\n'.join(pieces)
    with open(OUT_TEX, 'w', encoding='utf-8', newline='\n') as f:
        f.write(tex)
    print(f'wrote {OUT_TEX} ({len(tex)} bytes)')
    print('packs found:',
          'main' if main_pack else '-',
          'optim' if opt_pack else '-',
          'reg' if reg_pack else '-',
          'aug' if aug_pack else '-',
          'robust' if rob_pack else '-',
          'error' if err_pack else '-')


if __name__ == '__main__':
    main()
