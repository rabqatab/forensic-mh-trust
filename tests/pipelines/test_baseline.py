import forensic_mh.pipelines.baseline as b
from forensic_mh.pipelines.baseline import (
    _encode,
    build_genome_wide_matrix,
    collect_genome_wide_strings,
    discover_chrom_vcfs,
)


def test_collect_genome_wide_strings_merges_and_orders(monkeypatch):
    fake = {
        "1": ({"s1": {"m1": "A|A"}}, ["m1"]),
        "2": ({"s1": {"m2": "C|C"}, "s2": {"m2": "T|T"}}, ["m2"]),
    }
    monkeypatch.setattr(
        b, "_collect_chrom",
        lambda vcf_path, chrom, sample_ids, build: fake[chrom.replace("chr", "")],
    )
    rows, names = collect_genome_wide_strings({"2": "v2", "1": "v1"})
    assert names == ["m1", "m2"]               # sorted chromosome order
    assert rows["s1"] == {"m1": "A|A", "m2": "C|C"}  # merged across chroms
    assert rows["s2"] == {"m2": "T|T"}


def test_discover_chrom_vcfs_maps_chrom_to_path(tmp_path):
    for name in ["EAS_chr1.vcf.gz", "EAS_chr22.vcf.gz", "EAS_chr1.vcf.gz.csi"]:
        (tmp_path / name).write_text("")
    found = discover_chrom_vcfs(str(tmp_path))
    assert set(found) == {"1", "22"}  # .csi index not matched
    assert found["1"].endswith("EAS_chr1.vcf.gz")


def test_discover_chrom_vcfs_custom_prefix_isolates_ood(tmp_path):
    # EAS and OOD subsets coexist; prefix must select the right family
    for name in ["EAS_chr1.vcf.gz", "OOD_chr1.vcf.gz", "OOD_chr2.vcf.gz"]:
        (tmp_path / name).write_text("")
    assert set(discover_chrom_vcfs(str(tmp_path), prefix="OOD_chr")) == {"1", "2"}
    assert set(discover_chrom_vcfs(str(tmp_path), prefix="EAS_chr")) == {"1"}


def test_encode_assigns_per_column_codes_and_fills_missing():
    rows = {
        "s1": {"m1": "A|A", "m2": "G|G"},
        "s2": {"m1": "A|A"},  # missing m2 → filled "N|N"
    }
    X, ids = _encode(rows, ["m1", "m2"])
    assert ids == ["s1", "s2"]
    assert X.shape == (2, 2)
    # m1 identical for both samples → same code
    assert X[0, 0] == X[1, 0]
    # m2: s1 has G|G, s2 filled N|N → different codes
    assert X[0, 1] != X[1, 1]


def test_genome_wide_concatenates_markers_across_chromosomes(monkeypatch):
    # stub the per-chromosome I/O collector with synthetic data for two chroms
    fake = {
        "1": ({"s1": {"m1a": "A|A"}, "s2": {"m1a": "T|T"}}, ["m1a"]),
        "2": ({"s1": {"m2a": "G|G"}, "s2": {"m2a": "G|G"}}, ["m2a"]),
    }
    monkeypatch.setattr(
        b, "_collect_chrom",
        lambda vcf_path, chrom, sample_ids, build: fake[chrom.replace("chr", "")],
    )
    # pass out of order to confirm sorted-by-chromosome column order
    X, ids, names = build_genome_wide_matrix({"2": "v2.vcf.gz", "1": "v1.vcf.gz"})
    assert names == ["m1a", "m2a"]   # chr1 markers before chr2 markers
    assert ids == ["s1", "s2"]
    assert X.shape == (2, 2)


def test_genome_wide_merges_per_sample_rows_across_chromosomes(monkeypatch):
    fake = {
        "1": ({"s1": {"m1": "A|A"}}, ["m1"]),
        "2": ({"s1": {"m2": "C|C"}}, ["m2"]),
    }
    monkeypatch.setattr(
        b, "_collect_chrom",
        lambda vcf_path, chrom, sample_ids, build: fake[chrom.replace("chr", "")],
    )
    X, ids, names = build_genome_wide_matrix({"1": "v1", "2": "v2"})
    assert names == ["m1", "m2"]
    assert ids == ["s1"]
    assert X.shape == (1, 2)  # one sample, both chromosomes' markers
