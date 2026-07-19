from scripts.bootstrap_stats import N_PERMUTATIONS, permutation_test


def test_permutation_p_value_never_zero_on_extreme_delta():
    # An observed delta far outside any per-item edit-distance range can
    # never be matched by a resample, so raw n_ge/N would read exactly 0.
    per_item = [(0, 10, 0, 10)] * 20
    p_value, _ = permutation_test(per_item, observed_delta=1.0, seed=42)
    assert p_value > 0.0
    assert p_value == 1 / (N_PERMUTATIONS + 1)


def test_permutation_p_value_is_one_when_observed_delta_is_zero():
    per_item = [(1, 10, 2, 10)] * 20
    p_value, _ = permutation_test(per_item, observed_delta=0.0, seed=42)
    assert p_value == 1.0
