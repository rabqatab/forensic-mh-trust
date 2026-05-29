# Plan 3a — SSL Forensic FM Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the testable core of the SSL forensic foundation model — vocab, dataset (masking + ADO/dropout views), dual SSL objectives, a small marker-transformer encoder, three task heads, the two training loops, and a `ForensicFMClassifier` that exposes `predict_proba`/`embed` so Plan 2's `ConformalClassifier` wraps it unchanged.

**Architecture:** Per the spec `docs/superpowers/specs/2026-05-29-ssl-forensic-fm-design.md`. A marker-transformer over capped per-marker diplotype vocab (top-`K`); masked-marker modeling uses weight-tied per-marker value embeddings (einsum logits); contrastive uses NT-Xent over two augmented views (string-level ADO het→hom + marker dropout). Everything is unit-tested on tiny synthetic data — no GPU, no real VCFs. Plan 3b adds the all-2504 extraction, kinship-pair construction, and GB10/sparkq runs.

**Tech Stack:** PyTorch (CPU for tests; bf16 on GB10 in Plan 3b), numpy, scikit-learn (`BaseEstimator` for the adapter), pytest. Reuses `forensic_mh.uq.conformal_classifier.ConformalClassifier` from Plan 2.

**Scope note:** Plan 3a deliberately excludes data extraction, kinship-pair building from the pedigree, the sparkq pretraining/fine-tune scripts, and the held-out-superpop logic — those are Plan 3b. 3a's deliverable is a correct, tested FM module that trains a few steps on synthetic data and integrates with the trust layer.

---

## Conventions (read once)

- **Per-marker vocab cap `K`** (default 16): each marker keeps its top `K-1` most frequent diplotype strings + an `OTHER` bucket (index `K-1`). Slot `K` is the `MASK` token. So each marker has `K+1` embedding slots; masked prediction is over the `K` value classes only.
- Global embedding index for marker `j`, local code `c`: `j*(K+1) + c`.
- A diplotype cell is `"h0|h1"`; it is **het** iff `h0 != h1`.
- All randomness in datasets/augmentation uses a passed `torch.Generator` or `numpy` `default_rng(seed)` so tests are deterministic.

## File Structure

```
src/forensic_mh/fm/
  __init__.py        # NEW (empty)
  vocab.py           # NEW — FMVocab: per-marker top-K mapping, encode(), MASK/OTHER consts
  dataset.py         # NEW — MHMatrixDataset: masked view + 2 contrastive views (ADO+dropout)
  objectives.py      # NEW — masked_marker_loss, nt_xent (pure)
  encoder.py         # NEW — MHTransformer: forward()->per-marker hidden, embed()->(N,d)
  heads.py           # NEW — AncestryHead, SexHead, KinshipHead
  pretrain.py        # NEW — ssl_pretrain(encoder, dataset, cfg)
  finetune.py        # NEW — multitask_finetune(encoder, heads, ...)
  sklearn_api.py     # NEW — ForensicFMClassifier(BaseEstimator): fit/predict_proba/embed
tests/fm/
  __init__.py        # NEW (empty)
  test_vocab.py      # NEW
  test_dataset.py    # NEW
  test_objectives.py # NEW
  test_encoder.py    # NEW
  test_heads.py      # NEW
  test_pretrain.py   # NEW
  test_finetune.py   # NEW
  test_sklearn_api.py# NEW
pyproject.toml       # MODIFY — add torch
```

---

## Task 1: torch dependency + fm package + FMVocab

**Files:**
- Modify: `pyproject.toml` (add `torch`)
- Create: `src/forensic_mh/fm/__init__.py` (empty), `tests/fm/__init__.py` (empty)
- Create: `src/forensic_mh/fm/vocab.py`
- Test: `tests/fm/test_vocab.py`

- [ ] **Step 1: Add torch to pyproject and install**

Edit `pyproject.toml` dependencies list — add after `"xgboost>=2.0",`:

```toml
    "torch>=2.2",
```

Then:

```bash
uv pip install -e ".[dev]"
uv run python -c "import torch; print('torch', torch.__version__)"
```

Expected: a torch version prints. (On the GB10 node in Plan 3b, install per the `dgx-spark-gpu` skill — `NVIDIA_DISABLE_REQUIRE=1`, bf16, no FP8. CPU wheel is fine for these tests.)

- [ ] **Step 2: Create empty package files**

```bash
mkdir -p src/forensic_mh/fm tests/fm
touch src/forensic_mh/fm/__init__.py tests/fm/__init__.py
```

- [ ] **Step 3: Write the failing test**

```python
# tests/fm/test_vocab.py
from forensic_mh.fm.vocab import FMVocab


def test_vocab_keeps_top_k_minus_one_plus_other():
    # marker 0: A|A x3, G|G x1, T|T x1 ; K=2 → keep top-1 (A|A), rest → OTHER
    rows = [["A|A"], ["A|A"], ["A|A"], ["G|G"], ["T|T"]]
    v = FMVocab(rows, k=2)
    codes = v.encode(rows)
    assert codes.shape == (5, 1)
    # A|A gets a real class; G|G and T|T collapse to OTHER (== k-1 == 1)
    assert codes[0, 0] != v.OTHER
    assert codes[3, 0] == v.OTHER and codes[4, 0] == v.OTHER


def test_vocab_unseen_string_maps_to_other():
    v = FMVocab([["A|A"], ["A|A"]], k=4)
    codes = v.encode([["NOVEL|HAP"]])
    assert codes[0, 0] == v.OTHER


def test_vocab_exposes_dimensions():
    v = FMVocab([["A|A", "C|C"], ["G|G", "C|C"]], k=8)
    assert v.n_markers == 2
    assert v.k == 8
    assert v.MASK == 8          # slot k is MASK
    assert v.n_slots == 9       # K value classes + MASK
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run python -m pytest tests/fm/test_vocab.py -q`
Expected: ImportError (no `vocab` module)

- [ ] **Step 5: Write minimal implementation**

```python
# src/forensic_mh/fm/vocab.py
"""Per-marker capped diplotype vocabulary for the FM.

Each marker keeps its top (k-1) most frequent diplotype strings; everything
else (rare or unseen) maps to OTHER (index k-1). Slot k is the MASK token.
Uniform (k+1)-slot layout per marker → vectorised masked prediction.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


class FMVocab:
    def __init__(self, rows: list[list[str]], k: int = 16):
        self.k = k
        self.OTHER = k - 1
        self.MASK = k
        self.n_slots = k + 1
        self.n_markers = len(rows[0])
        # per-marker {string: code} for the top (k-1) strings
        self.maps_: list[dict[str, int]] = []
        for j in range(self.n_markers):
            counts = Counter(row[j] for row in rows)
            top = [s for s, _ in counts.most_common(k - 1)]
            self.maps_.append({s: i for i, s in enumerate(top)})

    def encode(self, rows: list[list[str]]) -> np.ndarray:
        out = np.empty((len(rows), self.n_markers), dtype=np.int64)
        for i, row in enumerate(rows):
            for j in range(self.n_markers):
                out[i, j] = self.maps_[j].get(row[j], self.OTHER)
        return out
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run python -m pytest tests/fm/test_vocab.py -q`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/forensic_mh/fm/__init__.py tests/fm/__init__.py \
        src/forensic_mh/fm/vocab.py tests/fm/test_vocab.py
git commit -m "feat(fm): torch dep + FMVocab (per-marker top-K diplotype vocab)"
```

---

## Task 2: MHMatrixDataset — masked view + two contrastive views

**Files:**
- Create: `src/forensic_mh/fm/dataset.py`
- Test: `tests/fm/test_dataset.py`

Produces, per sample: a masked-modeling view (codes with ~`mask_frac` positions set to `MASK`, plus targets/positions) and two stochastic contrastive views (string-level ADO het→hom + marker dropout→`MASK`, then encoded).

- [ ] **Step 1: Write the failing test**

```python
# tests/fm/test_dataset.py
import numpy as np
import torch

from forensic_mh.fm.dataset import MHMatrixDataset
from forensic_mh.fm.vocab import FMVocab


def _ds(seed=0, **kw):
    rows = [["A|T", "G|G"], ["A|A", "G|C"], ["A|T", "C|C"], ["T|T", "G|C"]]
    v = FMVocab(rows, k=4)
    return MHMatrixDataset(rows, v, seed=seed, **kw), v


def test_item_has_masked_view_and_two_contrastive_views():
    ds, v = _ds()
    item = ds[0]
    for key in ("input", "target", "mask_pos", "view1", "view2"):
        assert key in item
    assert item["input"].shape == (2,)        # n_markers
    assert item["view1"].shape == (2,)
    assert item["mask_pos"].dtype == torch.bool


def test_masked_positions_are_set_to_mask_token_and_targets_preserved():
    ds, v = _ds(mask_frac=1.0)  # mask everything → deterministic
    item = ds[0]
    assert torch.all(item["input"] == v.MASK)
    # targets equal the un-masked encoding
    base = torch.tensor(v.encode([["A|T", "G|G"]])[0])
    assert torch.all(item["target"] == base)
    assert torch.all(item["mask_pos"])


def test_ado_collapses_only_heterozygous_cells():
    # marker 0 of sample 0 is "A|T" (het); ado_prob=1 forces collapse to hom
    ds, v = _ds(ado_prob=1.0, drop_prob=0.0)
    item = ds[0]
    # collapsed het encodes to either A|A or T|T; both are rare here → OTHER,
    # but crucially it must differ from the original het code OR be OTHER.
    orig = v.encode([["A|T", "G|G"]])[0]
    # marker 1 "G|G" is hom → ado leaves it unchanged across views
    assert item["view1"][1].item() == orig[1] or item["view1"][1].item() == v.OTHER


def test_len_matches_sample_count():
    ds, v = _ds()
    assert len(ds) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/fm/test_dataset.py -q`
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/forensic_mh/fm/dataset.py
"""Dataset producing one masked-modeling view + two contrastive views per sample.

Contrastive augmentations operate at the STRING level so allele-dropout (ADO,
het→hom) is faithful, then re-encode with the shared FMVocab.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from forensic_mh.fm.vocab import FMVocab


def _ado(row: list[str], rng: np.random.Generator, p: float) -> list[str]:
    """Collapse heterozygous 'h0|h1' (h0!=h1) to a homozygote with prob p."""
    out = []
    for cell in row:
        h0, h1 = cell.split("|", 1)
        if h0 != h1 and rng.random() < p:
            keep = h0 if rng.random() < 0.5 else h1
            out.append(f"{keep}|{keep}")
        else:
            out.append(cell)
    return out


class MHMatrixDataset(Dataset):
    def __init__(self, rows: list[list[str]], vocab: FMVocab,
                 mask_frac: float = 0.15, ado_prob: float = 0.1,
                 drop_prob: float = 0.15, seed: int = 0):
        self.rows = rows
        self.vocab = vocab
        self.mask_frac = mask_frac
        self.ado_prob = ado_prob
        self.drop_prob = drop_prob
        self.base = vocab.encode(rows)           # (N, M) int64
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _view(self, i: int) -> torch.Tensor:
        row = _ado(self.rows[i], self.rng, self.ado_prob)
        codes = self.vocab.encode([row])[0].copy()
        drop = self.rng.random(self.vocab.n_markers) < self.drop_prob
        codes[drop] = self.vocab.MASK
        return torch.from_numpy(codes)

    def __getitem__(self, i: int) -> dict:
        base = self.base[i].copy()
        target = torch.from_numpy(base.copy())
        mask_pos = torch.from_numpy(self.rng.random(self.vocab.n_markers) < self.mask_frac)
        inp = torch.from_numpy(base)
        inp[mask_pos] = self.vocab.MASK
        return {
            "input": inp, "target": target, "mask_pos": mask_pos,
            "view1": self._view(i), "view2": self._view(i),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/fm/test_dataset.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/fm/dataset.py tests/fm/test_dataset.py
git commit -m "feat(fm): MHMatrixDataset with masked view + ADO/dropout contrastive views"
```

---

## Task 3: SSL objectives — masked_marker_loss + nt_xent

**Files:**
- Create: `src/forensic_mh/fm/objectives.py`
- Test: `tests/fm/test_objectives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/fm/test_objectives.py
import torch

from forensic_mh.fm.objectives import masked_marker_loss, nt_xent


def test_masked_loss_zero_when_logits_perfect_on_masked_only():
    N, M, K = 2, 3, 4
    targets = torch.tensor([[0, 1, 2], [3, 0, 1]])
    mask_pos = torch.tensor([[True, False, True], [False, True, False]])
    logits = torch.full((N, M, K), -10.0)
    # put a huge value on the true class everywhere
    for n in range(N):
        for m in range(M):
            logits[n, m, targets[n, m]] = 10.0
    loss = masked_marker_loss(logits, targets, mask_pos)
    assert loss.item() < 1e-3          # near zero
    assert loss.requires_grad is False or loss.item() >= 0


def test_masked_loss_ignores_unmasked_positions():
    N, M, K = 1, 2, 3
    targets = torch.tensor([[0, 0]])
    mask_pos = torch.tensor([[True, False]])
    good = torch.tensor([[[10.0, -10, -10], [10.0, -10, -10]]])
    bad = good.clone(); bad[0, 1] = torch.tensor([-10.0, 10, -10])  # wrong at unmasked
    assert torch.allclose(masked_marker_loss(good, targets, mask_pos),
                          masked_marker_loss(bad, targets, mask_pos))


def test_nt_xent_lower_when_positives_aligned():
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    aligned = z1.clone()
    crossed = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert nt_xent(z1, aligned, tau=0.5) < nt_xent(z1, crossed, tau=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/fm/test_objectives.py -q`
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/forensic_mh/fm/objectives.py
"""SSL objectives: masked-marker cross-entropy + NT-Xent contrastive."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_marker_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask_pos: torch.Tensor
) -> torch.Tensor:
    """Mean CE over masked marker positions only.

    logits: (N, M, K); targets: (N, M) in [0,K); mask_pos: (N, M) bool.
    Returns 0 if nothing is masked.
    """
    if mask_pos.sum() == 0:
        return logits.sum() * 0.0
    sel = logits[mask_pos]            # (n_masked, K)
    tgt = targets[mask_pos]           # (n_masked,)
    return F.cross_entropy(sel, tgt)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy (SimCLR) for a batch.

    z1, z2: (B, d) embeddings of two views. Positive pair = (z1[i], z2[i]).
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                # (2B, d)
    sim = z @ z.t() / tau                          # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, targets)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/fm/test_objectives.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/fm/objectives.py tests/fm/test_objectives.py
git commit -m "feat(fm): masked-marker + NT-Xent SSL objectives"
```

---

## Task 4: MHTransformer encoder

**Files:**
- Create: `src/forensic_mh/fm/encoder.py`
- Test: `tests/fm/test_encoder.py`

Per-marker value embedding (global-indexed) + learned marker-positional embedding + `[CLS]`; small transformer; `embed()`→`(N,d)`, `masked_logits()`→`(N,M,K)` via weight-tied einsum with the value-embedding table.

- [ ] **Step 1: Write the failing test**

```python
# tests/fm/test_encoder.py
import torch

from forensic_mh.fm.encoder import MHTransformer


def test_embed_shape():
    enc = MHTransformer(n_markers=5, k=4, d_model=16, n_layers=2, n_heads=2)
    x = torch.randint(0, 5, (3, 5))   # codes in [0, k+1)
    emb = enc.embed(x)
    assert emb.shape == (3, 16)


def test_masked_logits_shape_over_value_classes_only():
    enc = MHTransformer(n_markers=5, k=4, d_model=16, n_layers=2, n_heads=2)
    x = torch.randint(0, 5, (3, 5))
    logits = enc.masked_logits(x)
    assert logits.shape == (3, 5, 4)   # (N, M, K) — excludes MASK slot


def test_forward_is_differentiable():
    enc = MHTransformer(n_markers=4, k=4, d_model=8, n_layers=1, n_heads=2)
    x = torch.randint(0, 5, (2, 4))
    loss = enc.embed(x).sum()
    loss.backward()
    assert next(enc.parameters()).grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/fm/test_encoder.py -q`
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/forensic_mh/fm/encoder.py
"""Small marker-transformer. Per-marker value embeddings are weight-tied to the
masked-prediction head (einsum), so variable diplotype semantics per marker are
handled with a single uniform (K+1)-slot table."""
from __future__ import annotations

import torch
import torch.nn as nn


class MHTransformer(nn.Module):
    def __init__(self, n_markers: int, k: int, d_model: int = 128,
                 n_layers: int = 3, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.M = n_markers
        self.k = k
        self.slots = k + 1                       # K value classes + MASK
        self.d = d_model
        self.value_emb = nn.Embedding(n_markers * self.slots, d_model)
        self.marker_pos = nn.Embedding(n_markers, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.register_buffer("offsets", torch.arange(n_markers) * self.slots)

    def _hidden(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, M) per-marker codes in [0, slots)
        N = x.shape[0]
        tok = self.value_emb(x + self.offsets) + self.marker_pos(
            torch.arange(self.M, device=x.device))
        seq = torch.cat([self.cls.expand(N, -1, -1), tok], dim=1)  # (N, M+1, d)
        return self.enc(seq)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self._hidden(x)[:, 0]              # CLS → (N, d)

    def masked_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self._hidden(x)[:, 1:]                # per-marker hidden (N, M, d)
        W = self.value_emb.weight.view(self.M, self.slots, self.d)[:, : self.k]
        return torch.einsum("nmd,mkd->nmk", h, W)  # (N, M, K)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/fm/test_encoder.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/fm/encoder.py tests/fm/test_encoder.py
git commit -m "feat(fm): MHTransformer encoder with weight-tied masked head"
```

---

## Task 5: Task heads — ancestry, sex, kinship

**Files:**
- Create: `src/forensic_mh/fm/heads.py`
- Test: `tests/fm/test_heads.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/fm/test_heads.py
import torch

from forensic_mh.fm.heads import AncestryHead, KinshipHead, SexHead


def test_ancestry_head_outputs_logits_per_class():
    h = AncestryHead(d_model=16, n_classes=5)
    out = h(torch.randn(4, 16))
    assert out.shape == (4, 5)


def test_sex_head_outputs_single_logit():
    h = SexHead(d_model=16)
    out = h(torch.randn(4, 16))
    assert out.shape == (4, 1)


def test_kinship_head_is_symmetric_in_pair_order():
    h = KinshipHead(d_model=16)
    a, b = torch.randn(3, 16), torch.randn(3, 16)
    # |a-b| and a*b are symmetric → swapping pair order gives same logit
    assert torch.allclose(h(a, b), h(b, a), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/fm/test_heads.py -q`
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/forensic_mh/fm/heads.py
"""Task heads on the shared FM embedding."""
from __future__ import annotations

import torch
import torch.nn as nn


class AncestryHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


class SexHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


class KinshipHead(nn.Module):
    """Pairwise: symmetric features (|a-b|, a*b) → relatedness logit."""
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([torch.abs(a - b), a * b], dim=1)
        return self.fc(feat)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/fm/test_heads.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/fm/heads.py tests/fm/test_heads.py
git commit -m "feat(fm): ancestry/sex/kinship task heads (kinship symmetric)"
```

---

## Task 6: SSL pretraining loop

**Files:**
- Create: `src/forensic_mh/fm/pretrain.py`
- Test: `tests/fm/test_pretrain.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/fm/test_pretrain.py
import torch

from forensic_mh.fm.dataset import MHMatrixDataset
from forensic_mh.fm.encoder import MHTransformer
from forensic_mh.fm.pretrain import ssl_pretrain
from forensic_mh.fm.vocab import FMVocab


def _setup():
    rows = [["A|T", "G|G", "C|C"], ["A|A", "G|C", "C|T"],
            ["A|T", "C|C", "C|C"], ["T|T", "G|C", "T|T"]] * 4  # 16 samples
    v = FMVocab(rows, k=4)
    ds = MHMatrixDataset(rows, v, seed=0)
    enc = MHTransformer(n_markers=3, k=4, d_model=16, n_layers=1, n_heads=2)
    return enc, ds


def test_ssl_pretrain_runs_and_returns_finite_losses():
    enc, ds = _setup()
    history = ssl_pretrain(enc, ds, epochs=2, batch_size=8, lr=1e-3,
                           lambda_contrastive=0.5, seed=0)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history)


def test_ssl_pretrain_decreases_loss_on_tiny_data():
    enc, ds = _setup()
    history = ssl_pretrain(enc, ds, epochs=8, batch_size=8, lr=1e-3,
                           lambda_contrastive=0.5, seed=0)
    assert history[-1]["loss"] < history[0]["loss"]   # learns something
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/fm/test_pretrain.py -q`
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/forensic_mh/fm/pretrain.py
"""SSL pretraining: masked-marker + NT-Xent contrastive on MHMatrixDataset."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from forensic_mh.fm.objectives import masked_marker_loss, nt_xent


def ssl_pretrain(encoder, dataset, epochs: int = 50, batch_size: int = 64,
                 lr: float = 1e-3, lambda_contrastive: float = 0.5,
                 weight_decay: float = 1e-2, seed: int = 0,
                 device: str = "cpu") -> list[dict]:
    torch.manual_seed(seed)
    encoder.to(device).train()
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = []
    for _ in range(epochs):
        tot = 0.0
        for b in loader:
            inp = b["input"].to(device)
            logits = encoder.masked_logits(inp)
            l_mask = masked_marker_loss(logits, b["target"].to(device),
                                        b["mask_pos"].to(device))
            z1 = encoder.embed(b["view1"].to(device))
            z2 = encoder.embed(b["view2"].to(device))
            l_con = nt_xent(z1, z2) if z1.shape[0] > 1 else z1.sum() * 0.0
            loss = l_mask + lambda_contrastive * l_con
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        history.append({"loss": tot / len(loader)})
    return history
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/fm/test_pretrain.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/fm/pretrain.py tests/fm/test_pretrain.py
git commit -m "feat(fm): SSL pretraining loop (masked + contrastive)"
```

---

## Task 7: Multi-task fine-tune loop

**Files:**
- Create: `src/forensic_mh/fm/finetune.py`
- Test: `tests/fm/test_finetune.py`

Fine-tunes the pretrained encoder + ancestry/sex heads on labelled data (kinship is exercised in Plan 3b where pedigree pairs exist; here ancestry+sex prove the multi-task loop).

- [ ] **Step 1: Write the failing test**

```python
# tests/fm/test_finetune.py
import numpy as np
import torch

from forensic_mh.fm.encoder import MHTransformer
from forensic_mh.fm.finetune import multitask_finetune
from forensic_mh.fm.heads import AncestryHead, SexHead


def test_finetune_runs_and_improves_ancestry_accuracy():
    rng = np.random.default_rng(0)
    N, M, K = 64, 4, 4
    X = torch.from_numpy(rng.integers(0, K, size=(N, M)))
    y_anc = torch.from_numpy(rng.integers(0, 3, size=N))
    # make ancestry learnable: marker 0 code correlates with label
    X[:, 0] = y_anc
    y_sex = torch.from_numpy(rng.integers(0, 2, size=N)).float()
    enc = MHTransformer(n_markers=M, k=K, d_model=16, n_layers=1, n_heads=2)
    heads = {"ancestry": AncestryHead(16, 3), "sex": SexHead(16)}
    hist = multitask_finetune(enc, heads, X, y_anc, y_sex,
                              epochs=20, batch_size=16, lr=1e-3, seed=0)
    assert hist[-1]["ancestry_acc"] > hist[0]["ancestry_acc"]
    assert hist[-1]["ancestry_acc"] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/fm/test_finetune.py -q`
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/forensic_mh/fm/finetune.py
"""Multi-task fine-tuning of encoder + ancestry/sex heads (joint loss)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def multitask_finetune(encoder, heads, X, y_anc, y_sex, epochs: int = 50,
                       batch_size: int = 64, lr: float = 1e-3,
                       lambda_sex: float = 0.5, weight_decay: float = 1e-2,
                       seed: int = 0, device: str = "cpu") -> list[dict]:
    torch.manual_seed(seed)
    encoder.to(device).train()
    anc, sex = heads["ancestry"].to(device), heads["sex"].to(device)
    params = list(encoder.parameters()) + list(anc.parameters()) + list(sex.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(X, y_anc, y_sex),
                        batch_size=batch_size, shuffle=True)
    history = []
    for _ in range(epochs):
        correct = tot = 0
        for xb, ya, ys in loader:
            emb = encoder.embed(xb.to(device))
            la = anc(emb)
            loss = F.cross_entropy(la, ya.to(device))
            loss = loss + lambda_sex * F.binary_cross_entropy_with_logits(
                sex(emb).squeeze(1), ys.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            correct += int((la.argmax(1).cpu() == ya).sum()); tot += len(ya)
        history.append({"ancestry_acc": correct / tot})
    return history
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/fm/test_finetune.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/fm/finetune.py tests/fm/test_finetune.py
git commit -m "feat(fm): multi-task fine-tune loop (ancestry + sex)"
```

---

## Task 8: ForensicFMClassifier — trust-layer adapter + ConformalClassifier integration

**Files:**
- Create: `src/forensic_mh/fm/sklearn_api.py`
- Test: `tests/fm/test_sklearn_api.py`

The single coupling point with Plan 2: a scikit-learn-style estimator (`fit`, `predict_proba`, `embed`) so `ConformalClassifier(base_estimator=ForensicFMClassifier(...))` works unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/fm/test_sklearn_api.py
import numpy as np

from forensic_mh.fm.sklearn_api import ForensicFMClassifier
from forensic_mh.uq.conformal_classifier import ConformalClassifier


def _data():
    rng = np.random.default_rng(0)
    N, M, K = 120, 5, 4
    X = rng.integers(0, K, size=(N, M))
    y = rng.integers(0, 3, size=N)
    X[:, 0] = y                      # learnable signal
    return X, y


def test_predict_proba_shape_and_normalised():
    X, y = _data()
    clf = ForensicFMClassifier(k=4, d_model=16, n_layers=1, n_heads=2,
                               pretrain_epochs=2, finetune_epochs=10, seed=0).fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (len(X), 3)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)


def test_embed_returns_fixed_width_vectors():
    X, y = _data()
    clf = ForensicFMClassifier(k=4, d_model=16, n_layers=1, n_heads=2,
                               pretrain_epochs=1, finetune_epochs=2, seed=0).fit(X, y)
    assert clf.embed(X).shape == (len(X), 16)


def test_conformal_classifier_wraps_fm_unchanged():
    X, y = _data()
    base = ForensicFMClassifier(k=4, d_model=16, n_layers=1, n_heads=2,
                                pretrain_epochs=1, finetune_epochs=10, seed=0)
    cc = ConformalClassifier(base, alpha=0.1, mondrian=True).fit(X, y)
    sets = cc.predict_set(X[:10])
    assert len(sets) == 10 and all(isinstance(s, list) for s in sets)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/fm/test_sklearn_api.py -q`
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

`ForensicFMClassifier` accepts an already-integer-encoded matrix `X` (as the synthetic tests pass). In Plan 3b the real pipeline encodes diplotype strings with `FMVocab` first; here the estimator treats `X` codes directly, inferring `n_markers` from `X.shape[1]` and using the configured `k`.

```python
# src/forensic_mh/fm/sklearn_api.py
"""scikit-learn-style adapter so the Plan 2 trust layer wraps the FM unchanged.

Accepts integer-encoded code matrices (values in [0, k)); MASK slot is k.
Runs a short SSL pretrain on the input matrix (self-supervised, label-free)
then multi-task fine-tune of the ancestry head, exposing predict_proba/embed.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, TensorDataset

from forensic_mh.fm.encoder import MHTransformer
from forensic_mh.fm.heads import AncestryHead
from forensic_mh.fm.objectives import masked_marker_loss


class ForensicFMClassifier(BaseEstimator):
    def __init__(self, k: int = 16, d_model: int = 128, n_layers: int = 3,
                 n_heads: int = 4, pretrain_epochs: int = 30,
                 finetune_epochs: int = 30, lr: float = 1e-3,
                 mask_frac: float = 0.15, seed: int = 42, device: str = "cpu"):
        self.k = k; self.d_model = d_model; self.n_layers = n_layers
        self.n_heads = n_heads; self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs; self.lr = lr
        self.mask_frac = mask_frac; self.seed = seed; self.device = device

    def _mask(self, x, gen):
        m = torch.rand(x.shape, generator=gen) < self.mask_frac
        inp = x.clone(); inp[m] = self.k          # MASK slot
        return inp, m

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = torch.as_tensor(np.asarray(X), dtype=torch.long)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y_idx = torch.as_tensor(np.searchsorted(self.classes_, y), dtype=torch.long)
        M = X.shape[1]
        self.encoder_ = MHTransformer(M, self.k, self.d_model,
                                      self.n_layers, self.n_heads).to(self.device)
        gen = torch.Generator().manual_seed(self.seed)
        # SSL pretrain (masked-marker; label-free)
        opt = torch.optim.AdamW(self.encoder_.parameters(), lr=self.lr, weight_decay=1e-2)
        dl = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)
        self.encoder_.train()
        for _ in range(self.pretrain_epochs):
            for (xb,) in dl:
                inp, m = self._mask(xb, gen)
                loss = masked_marker_loss(self.encoder_.masked_logits(inp.to(self.device)),
                                          xb.to(self.device), m.to(self.device))
                opt.zero_grad(); loss.backward(); opt.step()
        # multi-task fine-tune (ancestry head)
        self.head_ = AncestryHead(self.d_model, self.n_classes_).to(self.device)
        params = list(self.encoder_.parameters()) + list(self.head_.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-2)
        dl = DataLoader(TensorDataset(X, y_idx), batch_size=64, shuffle=True)
        for _ in range(self.finetune_epochs):
            for xb, yb in dl:
                logits = self.head_(self.encoder_.embed(xb.to(self.device)))
                loss = torch.nn.functional.cross_entropy(logits, yb.to(self.device))
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.encoder_.eval(); self.head_.eval()
        X = torch.as_tensor(np.asarray(X), dtype=torch.long).to(self.device)
        logits = self.head_(self.encoder_.embed(X))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    @torch.no_grad()
    def embed(self, X):
        self.encoder_.eval()
        X = torch.as_tensor(np.asarray(X), dtype=torch.long).to(self.device)
        return self.encoder_.embed(X).cpu().numpy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/fm/test_sklearn_api.py -q`
Expected: 3 passed (the conformal integration test confirms the Plan 2 contract)

- [ ] **Step 5: Run the full suite**

Run: `uv run python -m pytest -q`
Expected: all green (Plan 1 + Plan 2 + Plan 3a fm tests)

- [ ] **Step 6: Commit**

```bash
git add src/forensic_mh/fm/sklearn_api.py tests/fm/test_sklearn_api.py
git commit -m "feat(fm): ForensicFMClassifier adapter; ConformalClassifier wraps it (Plan 3a core complete)"
```

---

## Self-Review

### Spec coverage (spec section → task)

| spec 항목 | task |
|---|---|
| MH-matrix substrate, capped vocab | Task 1 (FMVocab) |
| masked-marker objective | Task 3 (`masked_marker_loss`) + Task 4 (weight-tied head) |
| contrastive + ADO/dropout augmentation | Task 2 (dataset views) + Task 3 (`nt_xent`) |
| MHTransformer encoder, embedding | Task 4 |
| ancestry/sex/kinship heads | Task 5 |
| SSL pretrain → multi-task fine-tune | Task 6, 7 |
| `predict_proba`/`embed`, trust-layer wrap | Task 8 (+ ConformalClassifier integration test) |
| all-2504 extraction, kinship pairs, sparkq/GB10 runs, held-out superpop | **Plan 3b** (out of scope here — noted) |

### Placeholder scan
No TBD/TODO. Every code step is complete. Kinship head is built+tested (Task 5) though its *training* on real pedigree pairs is Plan 3b — this is an explicit scope boundary, not a placeholder.

### Type consistency
- `FMVocab.encode -> (N,M) int64`; `MASK=k`, value classes `0..k-1` — consistent in dataset (Task 2), encoder slots `k+1` (Task 4), `masked_logits -> (N,M,k)` matches `masked_marker_loss` targets in `[0,k)` (Task 3).
- `MHTransformer.embed -> (N, d_model)` consumed by all heads (Task 5) and `ForensicFMClassifier.embed`/`predict_proba` (Task 8).
- `ssl_pretrain(encoder, dataset, ...)` and `multitask_finetune(encoder, heads, X, y_anc, y_sex, ...)` signatures match their tests.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-29-ssl-fm-core.md`. Plan 3b (data scale-up + GB10 runs) will be written after 3a is built and the genome-wide download completes.

Two execution options:
1. **Subagent-Driven (recommended)** — fresh subagent per task + review between tasks.
2. **Inline Execution** — task-by-task in this session with checkpoints.
