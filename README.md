# mh-eas-panel — Trustworthy Forensic-FM

동아시아 집단 분류를 위한 마이크로하플로타입(MH) 패널 연구.
SSL pretraining + Conformal/Open-set UQ 통합 — Trustworthy Forensic-FM.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run baseline

```bash
bash scripts/01_download_1000g.sh
bash scripts/02_extract_eas_samples.sh
python scripts/03_run_baseline.py
```

## Documents

- `docs/01_proposal_review.md` — 비판적 리뷰
- `docs/02_literature_landscape.md` — 문헌 매핑
- `docs/03_novelty_options.md` — novelty 옵션 (v2)
- `docs/proposal_extension_v1.md` — 팀 보강 제안서 (Option 1)
- `docs/superpowers/plans/` — 2주 스프린트 plan
