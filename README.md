# ai-foundation

Learning AI from the fundamentals, one hand-written piece at a time.

I'm a sophomore software engineering student. This repo is my learning record: notes, exercises and from-scratch implementations, from NumPy up to (eventually) [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/). I care more about understanding how the models work than about using them, so the rule here is simple: **I write the key logic myself, and I must be able to explain every step and every tensor shape.**

## Roadmap

| Stage | When | Content | Milestone | Status |
| --- | --- | --- | --- | --- |
| 0 · Foundations | now – end of semester | NumPy, vectorization, gradients, probability & MLE | Linear & logistic regression from scratch in NumPy, with gradient checks | 🟡 in progress |
| 1 · Neural nets by hand | winter break | Karpathy's *Neural Networks: Zero to Hero*, d2l.ai | A micrograd-level autograd engine training a small MLP | ⚪ |
| 2 · Transformer & PyTorch | next semester | *Attention Is All You Need*, PyTorch basics | A decoder-only Transformer trained on a small corpus | ⚪ |
| 3 · CS336 | summer | Tokenization → architecture → systems → scaling → data → post-training | CS336 assignments | ⚪ |
| 4 · Research | long term | Weekly paper reading, small reproductions | – | ⚪ |

Full plan (Chinese): [`path.md`](path.md) · Stage 0 detailed route: [`stage0/阶段 0 学习路线：从地基到手写回归.md`](stage0/阶段%200%20学习路线：从地基到手写回归.md)

## Stage 0 progress

| Week | Topic | Dates | Status |
| --- | --- | --- | --- |
| [1](stage0/week1/) | NumPy basics: shape, indexing, `axis` / `keepdims`, broadcasting | 11-13 – 11-28 | ✅ |
| [2](stage0/week2/) | Vectorized programming: softmax, cross-entropy, distance matrix | 11-28 – | 🟡 |
| 3 | Gradients and vectorized derivatives, numerical gradient check | | ⚪ |
| 4 | Probability and maximum likelihood | | ⚪ |
| 5 | Linear & logistic regression from scratch | | ⚪ |
| 6 | Review and buffer | | ⚪ |

Each week folder has a `README.md` with the schedule (including time spent per session), plus dated study notes `YYYY-MM-DD-<topic>.md`, code and tests.

## Layout

```
ai-foundation/
  README.md          this file
  path.md            overall roadmap (Chinese)
  stage0/
    AGENTS.md        collaboration rules for AI coding assistants
    week1/           notes, small_funcs.py + tests
    week2/
    ...
```

## How I study

Every week follows the same loop: **read → predict → write → verify → explain**.

- **Predict before running.** For anything about shapes or outputs, write the prediction first; if it's wrong, write down which rule I misused.
- **Derive by hand before coding.** Gradients and losses are worked out on paper first.
- **Key logic is mine.** Losses, gradients, update rules, softmax: no copy-paste, no AI-written solutions. AI assistants only write scaffolding (signatures, docstrings, tests, TODOs) and review my code with graded hints. See [`stage0/AGENTS.md`](stage0/AGENTS.md).
- **Test everything.** Every function has pytest tests, checked against a naive loop version or a numerical gradient.
- **Explain it out loud.** Each week ends with an oral self-check; whatever I can't explain, I haven't learned yet.

Notes are written in English on purpose, to build the habit.

## Setup

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). Dependencies: `numpy`, `pytest` (later `matplotlib`).

```powershell
cd stage0/week1
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe numpy pytest
.\.venv\Scripts\python.exe -m pytest -q
```

Week 2 onwards reuses the week 1 environment: `..\week1\.venv\Scripts\python.exe -m pytest -q`.
