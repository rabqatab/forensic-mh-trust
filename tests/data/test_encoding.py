import numpy as np

from forensic_mh.data.encoding import DiplotypeEncoder


def test_encoder_assigns_stable_codes_to_seen_strings():
    enc = DiplotypeEncoder()
    train = [["A-T|G-C", "A-A|A-A"], ["A-T|G-C", "T-T|T-T"]]  # 2 samples, 2 markers
    X = enc.fit_transform(train)
    assert X.shape == (2, 2)
    # same string in column 0 → same code
    assert X[0, 0] == X[1, 0]


def test_encoder_maps_unseen_to_reserved_code():
    enc = DiplotypeEncoder(unseen_code=-1)
    enc.fit([["A-T|G-C"], ["A-T|G-C"]])
    X = enc.transform([["NOVEL|HAP"]])  # unseen in training
    assert X[0, 0] == -1


def test_encoder_unseen_fraction_reported():
    enc = DiplotypeEncoder()
    enc.fit([["A|A"], ["A|A"]])
    enc.transform([["A|A"], ["B|B"]])
    assert enc.last_unseen_fraction == 0.5  # one of two cells unseen
