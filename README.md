# fastbook-fixed

A **fixed starter set** of the [fast.ai](https://course.fast.ai/) course notebooks
(["Practical Deep Learning for Coders"](https://github.com/fastai/fastbook)),
debugged to run cleanly on current **fastai** + **Python 3.13** on Apple Silicon / modern setups.

The upstream [`fastai/fastbook`](https://github.com/fastai/fastbook) notebooks are excellent
but a few early chapters break out-of-the-box on newer library versions. This repo is those
same notebooks with the breakages fixed, so you can start the course without fighting your
environment first.

## What's here

| Chapters | Status |
|----------|--------|
| **01–07** | ✅ Debugged and verified to run end-to-end |
| **08–20, app_\*** | Pristine upstream (not yet run/verified here — fix as you go) |

Generated artifacts (`bears/`, `models/`, `export.pkl`, dataset downloads) are **not** included —
you'll create those as you work through the lessons. That's the point of the course. 🙂

## Getting started

```bash
git clone https://github.com/jasondostal/fastbook-fixed.git
cd fastbook-fixed
pip install -r requirements.txt   # or: conda env create -f environment.yml
jupyter lab                        # or jupyter notebook
```

Then open `01_intro.ipynb` and go. Work top-to-bottom, running each cell.

## Companion reading

**[`PRIMER_algorithms_and_deep_learning_math.md`](PRIMER_algorithms_and_deep_learning_math.md)** — a plain-English primer
spanning three courses at once: **algorithms** ([CS577](https://pages.cs.wisc.edu/~shuchi/courses/577-F14/): divide &
conquer, greedy, dynamic programming, network flow, P-vs-NP), **classical AI & machine learning**
([CS540](https://pages.cs.wisc.edu/~gkotse/cs540s26/index.html): search/A\*, minimax, regression, kNN, naive Bayes,
decision trees, k-means, NLP, reinforcement learning), and the **math behind deep learning** (this course: linear algebra,
gradients, gradient descent, loss/activation functions, backprop, CNNs). Every concept explained by analogy first,
formula second, with a linked table of contents, per-part further-reading pointers, a curated resource list, and a
section mapping each topic back to all three courses. No prior math love required.

## Attribution & license

This is a derivative of [`fastai/fastbook`](https://github.com/fastai/fastbook) by Jeremy Howard
and Sylvain Gugger. The original `LICENSE` is preserved. The course notebook prose is under
[CC BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/3.0/) and the code is Apache-2.0 —
this repo exists for non-commercial educational use. All credit for the course material goes to
fast.ai; the only changes here are bug fixes to make the notebooks run on current tooling.
