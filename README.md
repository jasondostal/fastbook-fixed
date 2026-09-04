# fastbook-fixed

The [fast.ai](https://course.fast.ai/) course notebooks
([*Practical Deep Learning for Coders*](https://github.com/fastai/fastbook)), debugged to
actually run on **current fastai / PyTorch / pandas on Apple Silicon**.

The upstream [`fastai/fastbook`](https://github.com/fastai/fastbook) notebooks are excellent,
but they were written years ago against different library versions and NVIDIA hardware. On a
current setup a fair number of cells break — some loudly, some by **hard-crashing the Python
kernel several minutes into training**, which is a miserable way to learn. This repo is those
same notebooks with those breakages fixed, and with an honest record of what was actually run.

## What "verified" means here

Every chapter marked ✅ was **executed top to bottom** in the environment below, and every cell
either produced output or failed *on purpose* (see [Intentional errors](#intentional-errors)).
Nothing is marked verified because it looked fine.

Verified on: **fastai 2.8.8 · torch 2.14.0 · pandas 3.0.5 · Python 3.12 · macOS 26.4 ·
Apple M5 Pro (MPS)**, on 2026-09-04.

| Chapter | Status | Runtime | Notes |
|---|---|---|---|
| 01_intro | 🔧 fixed, run pending | — | Metal LSTM crash fixed; full run is multi-hour (CPU text cell) |
| 02_production | ✅ verified | 0.8 min | `learn.export()` was broken; upload widget now degrades gracefully |
| 03_ethics | ✅ verified | 0.1 min | prose + a couple of cells |
| 04_mnist_basics | ✅ verified | 0.2 min | needed the Graphviz **system** binary |
| 05_pet_breeds | ✅ verified | 37.6 min | MPS presizing crash fixed; 1 intentional error |
| 06_multicat | ✅ verified | 7.6 min | `BCEWithLogitsLoss` subclass dispatch fixed |
| 07_sizing_and_tta | ✅ verified | 41.4 min | MPS presizing crash fixed |
| 08_collab | ✅ verified | 2.0 min | GroupLens TLS cert expired — mirror fallback added |
| 09_tabular | ⛔ blocked | — | needs Kaggle competition rules accepted (see below) |
| 10_nlp | ⚠️ reduced-scale | 5.7 min | code path verified on a subsampled corpus; see below |
| 11_midlevel_data | ✅ verified | 0.4 min | no changes needed |
| 12_nlp_dive | ✅ verified | 0.3 min | no changes needed |
| 13_convolutions | ✅ verified | 1.7 min | no changes needed |
| 14_resnet | ✅ verified | 37.4 min | no changes needed — 40 epochs of Imagenette |
| 15_arch_details | ✅ verified | 9.2 min | `create_body()` signature fixed |
| 16_accel_sgd | ✅ verified | 5.5 min | no changes needed |
| 17_foundations | ✅ verified | 0.1 min | fastai `Module` import added; 2 intentional errors |
| 18_CAM | ✅ verified | 1.3 min | hardcoded `.cuda()` fixed |
| 19_learner | ✅ verified | 0.5 min | `Self.parent.name()` + `.cuda()` fixed |
| 20_conclusion | ✅ verified | 0.6 min | no changes needed |

## The interesting breakages

**Metal (MPS) hard-crashes on LSTM backward.** The moment an AWD-LSTM body is unfrozen
(`learn.freeze_to(-2)` / `fine_tune`), PyTorch's Metal backend kills the entire process — not an
exception you can catch, the kernel just dies. Confirmed it is *not* memory: it dies at
1.8 GB resident on a 64 GB machine, and the identical workload completes on CPU at every
unfreeze stage. Affects **ch01** and **ch10**, which now drop to CPU for just those cells.

**Metal cannot do non-divisible adaptive pooling.** fastai's *presizing* idiom
(`item_tfms=Resize(460)` + `batch_tfms=aug_transforms(size=224)`) calls
`F.interpolate(..., mode='area')`, which dispatches to `adaptive_avg_pool2d`. MPS implements
that only when the input size is divisible by the output size, and random presizing crops
routinely violate it ([pytorch#96056](https://github.com/pytorch/pytorch/issues/96056)). It
fails **mid-epoch**, several batches in, so it reads like a random flake. Affects **ch05** and
**ch07**. Fixed by running that single op on CPU — everything else stays on the GPU, so
ch05 still trains at ~41 s/epoch.

**`learn.export()` is broken in fastai 2.8.8.** Transforms dispatch through
[plum](https://github.com/beartype/plum), whose `Function` object holds a threading `RLock`.
That makes the entire `Learner` unpicklable, so `export()` dies with
`cannot pickle '_thread.RLock' object` — taking the whole *deploy your model* lesson in ch02
with it. Fixed by teaching plum's `Function` to drop the lock when pickling and rebuild it on
load; `export()` → `load_learner()` → `predict()` then round-trips cleanly. (Note that
`export()` defaults to `cloudpickle` while `load_learner` defaults to stdlib `pickle`, so load
with the default — cloudpickle has no `Unpickler`.)

**GroupLens' TLS certificate expired on 2026-08-28**, which breaks
`untar_data(URLs.ML_100k)` for everyone, not just this repo. ch08 now tries the official
source first and falls back to a Kaggle mirror of the same archive, so it self-heals once
GroupLens renews.

**Library drift.** `create_body()` now needs an instantiated model; fastcore's
`Self.parent.name()` broke because `.name` is a plain `str`; pandas 3 removed `inplace=` from
`set_categories`; `nn.BCEWithLogitsLoss` has no `__torch_function__` implementation for
fastai's tensor subclasses; ch09 had a literal `parents=true` typo; ch17 used fastai's `Module`
without importing it.

## Intentional errors

Some cells are **supposed** to raise — the traceback *is* the lesson. These are left alone:

- `17_foundations` cells 33 & 61 — elementwise ops on mismatched shapes. The markdown above
  one of them literally reads *"This won't work:"*.
- `05_pet_breeds` cell 31 — `pets1.summary()` deliberately fails to collate a batch, and the
  text below it walks through the different-shapes error.

If you "fix" these, you delete the teaching.

## Getting started

```bash
git clone https://github.com/jasondostal/fastbook-fixed.git
cd fastbook-fixed
pip install -r requirements.txt
brew install graphviz      # macOS; Ubuntu: sudo apt install graphviz
jupyter lab
```

The Graphviz **system** binary is a real requirement, not just the Python package — without it
ch04 and ch09 fail with `ExecutableNotFound: failed to execute Path('dot')`.

## Known limits

- **ch09 needs you to accept the Kaggle competition rules** for
  [Bluebook for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/rules).
  Until you do, the download returns HTTP 403. The code fixes are applied, but it has not
  been run here.
- **ch10 is verified on a reduced corpus.** Because the Metal bug forces CPU, a full IMDB
  language-model fine-tune is roughly a 60-hour job on this machine. The whole notebook
  completes cleanly on a subsampled corpus, so the *code* is verified; the published accuracy
  numbers are not reproduced.
- Notebook **outputs are upstream's**, not regenerated. Only source cells were changed, which
  keeps the diff reviewable.

## Attribution & license

A derivative of [`fastai/fastbook`](https://github.com/fastai/fastbook) by Jeremy Howard and
Sylvain Gugger; the original `LICENSE` is preserved. Prose is
[CC BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/3.0/), code is GPL-3.0 per the
bundled LICENSE. Non-commercial educational use. All credit for the course material goes to
fast.ai — the only changes here are bug fixes to make the notebooks run on current tooling.

## Companion reading

**[`PRIMER_algorithms_and_deep_learning_math.md`](PRIMER_algorithms_and_deep_learning_math.md)** —
a plain-English primer spanning three courses: **algorithms**
([CS577](https://pages.cs.wisc.edu/~shuchi/courses/577-F14/)), **classical AI & ML**
([CS540](https://pages.cs.wisc.edu/~gkotse/cs540s26/index.html)), and the **math behind deep
learning**. Every concept by analogy first, formula second.
