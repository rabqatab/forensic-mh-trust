"""Embedding approach B — Nucleotide-Transformer transfer (Dalla-Torre 2024), GPU.

External large-scale pretrained DNA embeddings, to sidestep small-n: for each MH
marker we fetch the hg19 reference window (UCSC API), substitute each haplotype's
alleles at the SNP offsets to form the amplicon sequence, and embed it with the
pretrained Nucleotide Transformer (frozen). Per sample = concat of its markers'
NT embeddings (mean of the two haplotypes); a LogReg classifies. Tests whether a
foundation-model SEQUENCE embedding captures the (frequency-based) ancestry signal.
Marker subset = N_MARK richest (NumVars) markers. Same eval: 5-fold CV LogReg,
acc + far-OOD AUROC. Compare LogReg(one-hot) 79.6 / DietNet 73.6 / EmbMLP 29.7.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import AutoModelForMaskedLM, AutoTokenizer

from forensic_mh.data.markers import load_mh_markers, parse_positions
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs, load_eas_labels

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
NT = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
N_MARK, PAD = 150, 25
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ucsc(chrom, start, end):
    import time
    url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg19;chrom={chrom};start={start};end={end}"
    for attempt in range(4):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=30).read())["dna"].upper()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    y, _ = load_eas_labels(PANEL, sids)
    classes = np.unique(y); y = np.searchsorted(classes, y)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    osids = sorted(orows)
    mk = load_mh_markers().set_index("Name")

    # pick markers: richest (NumVars), allele-count consistent, with hg19 coords
    cand = []
    for m in names:
        if m not in mk.index:
            continue
        pos = parse_positions(mk.loc[m], build="hg19")
        d = rows[sids[0]].get(m, "N|N")
        if pos and "N" not in d and len(d.split("|")[0].split("-")) == len(pos):
            cand.append((m, int(mk.loc[m].get("NumVars", len(pos))), pos, str(mk.loc[m]["Chrom"])))
    cand.sort(key=lambda t: -t[1])
    cand = cand[:N_MARK]
    print(f"NT markers={len(cand)} (consistent, richest) device={DEV}", flush=True)

    # NT model
    tok = AutoTokenizer.from_pretrained(NT)
    model = AutoModelForMaskedLM.from_pretrained(NT, output_hidden_states=True).to(DEV).eval()

    def embed_seqs(seqs):
        out = []
        for i in range(0, len(seqs), 16):
            ids = tok(seqs[i:i + 16], return_tensors="pt", padding=True)["input_ids"].to(DEV)
            with torch.no_grad():
                out.append(model(ids).hidden_states[-1].mean(1).cpu().numpy())
        return np.vstack(out)

    def build_seq(refwin, wstart, pos, hap):
        s = list(refwin)
        for p, a in zip(pos, hap.split("-")):
            off = p - wstart
            if 0 <= off < len(s) and a in "ACGT":
                s[off] = a
        return "".join(s)

    # per marker: fetch ref, embed distinct haplotype sequences -> lookup; build sample features
    Xe = np.zeros((len(sids), len(cand) * 1280), np.float32)
    Xo = np.zeros((len(osids), len(cand) * 1280), np.float32)
    for j, (m, nv, pos, chrom) in enumerate(cand):
        wstart = min(pos) - PAD
        ref = ucsc(chrom, wstart, max(pos) + PAD)
        haps = set()
        for s in sids:
            for h in rows[s].get(m, "N|N").split("|"):
                if "N" not in h:
                    haps.add(h)
        for s in osids:
            for h in orows[s].get(m, "N|N").split("|"):
                if "N" not in h:
                    haps.add(h)
        haps = sorted(haps)
        emb = embed_seqs([build_seq(ref, wstart, pos, h) for h in haps]) if haps else np.zeros((1, 1280))
        lut = {h: emb[i] for i, h in enumerate(haps)}
        zero = np.zeros(1280, np.float32)

        def feat(s, src):
            hs = [h for h in src[s].get(m, "N|N").split("|") if "N" not in h]
            return np.mean([lut.get(h, zero) for h in hs], 0) if hs else zero
        for i, s in enumerate(sids):
            Xe[i, j * 1280:(j + 1) * 1280] = feat(s, rows)
        for i, s in enumerate(osids):
            Xo[i, j * 1280:(j + 1) * 1280] = feat(s, orows)
        if (j + 1) % 25 == 0:
            print(f"  embedded {j+1}/{len(cand)} markers", flush=True)

    from forensic_mh.uq.openset import msp_score, ood_auroc
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    accs, aurocs = [], []
    for tr, te in cv.split(Xe, y):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xe[tr], y[tr])
        accs.append(float((clf.predict(Xe[te]) == y[te]).mean()))
        aurocs.append(ood_auroc(msp_score(clf.predict_proba(Xe[te])), msp_score(clf.predict_proba(Xo))))

    out = {"model": "NT-transfer (LogReg on NT embeddings)", "nt_model": NT,
           "n_markers": len(cand), "feat_dim": len(cand) * 1280,
           "acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs)), 4),
           "auroc_mean": round(float(np.mean(aurocs)), 4),
           "reference": {"LogReg_onehot": 0.796, "DietNet": 0.736, "EmbMLP": 0.297}}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/nt_transfer.json").write_text(json.dumps(out, indent=2))
    print(f"\nNT-transfer acc={out['acc_mean']*100:.1f}±{out['acc_std']*100:.1f}  "
          f"far-OOD AUROC={out['auroc_mean']:.3f}  | ref LogReg 79.6, DietNet 73.6", flush=True)
    print("saved results/baseline/nt_transfer.json", flush=True)
    print("NT_TRANSFER_DONE", flush=True)


if __name__ == "__main__":
    main()
