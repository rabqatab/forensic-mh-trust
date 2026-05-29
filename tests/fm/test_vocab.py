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


def test_encode_multi_marker_shape_and_values():
    # Two markers, two training rows
    v = FMVocab([["A|A", "C|C"], ["A|A", "G|G"]], k=4)
    codes = v.encode([["A|A", "C|C"], ["T|T", "G|G"]])
    assert codes.shape == (2, 2)
    # marker 0: "A|A" seen → non-OTHER code; "T|T" unseen → OTHER
    assert codes[0, 0] != v.OTHER, "A|A should get a real (non-OTHER) code"
    assert codes[1, 0] == v.OTHER, "T|T is unseen and must map to OTHER"
    # marker 1: "C|C" and "G|G" are both seen but must get distinct codes
    assert codes[0, 1] != codes[1, 1], "C|C and G|G must have distinct codes"
