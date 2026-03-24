from core.rng import make_rng_pack, stable_seed_from_key


def test_stable_seed_from_key_is_deterministic():
    s1 = stable_seed_from_key("alpha", base_seed=123)
    s2 = stable_seed_from_key("alpha", base_seed=123)
    assert s1 == s2


def test_rng_pack_deterministic_sequences():
    a = make_rng_pack("sequence:test", base_seed=777)
    b = make_rng_pack("sequence:test", base_seed=777)
    seq_a = [float(a.np_rng.random()) for _ in range(5)]
    seq_b = [float(b.np_rng.random()) for _ in range(5)]
    assert seq_a == seq_b

