from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    CalibrationError,
    Sample,
    fit_profile,
    load_profile,
    load_samples,
    predict_model,
    save_profile,
    save_samples,
    split_holdout,
)


def linear_samples(a=0.35, b=0.8, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for d_true in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        r = (d_true - b) / a
        r += rng.normal(0, noise)
        out.append(Sample(r=float(r), d_true=float(d_true), scene="test"))
    return out


def reciprocal_samples(a=2.5, b=0.15, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for d_true in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        r = a / d_true - b
        r += rng.normal(0, noise)
        out.append(Sample(r=float(r), d_true=float(d_true), scene="test"))
    return out


def test_linear_fit_recovers_parameters():
    profile = fit_profile(linear_samples(), val_fraction=0.0)
    assert profile.model in ("linear", "reciprocal")
    preds = [profile.predict(s.r)[0] for s in linear_samples()]
    errs = np.abs(np.array(preds) - [s.d_true for s in linear_samples()])
    assert errs.max() < 0.05


def test_reciprocal_data_fit_is_accurate():
    samples = reciprocal_samples()
    profile = fit_profile(samples, val_fraction=0.0)
    errs = [abs(profile.predict(s.r)[0] - s.d_true) for s in samples]
    assert max(errs) < 0.05


def test_noisy_fit_reports_metrics():
    samples = linear_samples(noise=0.03)
    profile = fit_profile(samples)
    assert profile.train_metrics["n"] > 0
    assert profile.val_metrics["n"] > 0
    assert profile.val_metrics["rmse"] < 0.3


def test_too_few_samples_rejected():
    with pytest.raises(CalibrationError, match="at least 4"):
        fit_profile(linear_samples()[:3])


def test_too_few_distances_rejected():
    samples = [Sample(r=1.0, d_true=1.0), Sample(r=2.0, d_true=1.0),
               Sample(r=3.0, d_true=1.0), Sample(r=4.0, d_true=1.0)]
    with pytest.raises(CalibrationError, match="distinct"):
        fit_profile(samples)


def test_negative_distance_rejected():
    samples = linear_samples()
    samples[0] = Sample(r=1.0, d_true=-1.0)
    with pytest.raises(CalibrationError):
        fit_profile(samples)


def test_non_finite_sample_rejected():
    samples = linear_samples()
    samples[1] = Sample(r=float("nan"), d_true=1.0)
    with pytest.raises(CalibrationError, match="finite"):
        fit_profile(samples)


def test_extrapolation_flagged():
    profile = fit_profile(linear_samples(), val_fraction=0.0)
    inside = (profile.r_min + profile.r_max) / 2
    _, extrapolated_in = profile.predict(inside)
    assert extrapolated_in is False
    _, extrapolated_out = profile.predict(profile.r_max + 10.0)
    assert extrapolated_out is True


def test_profile_persistence_roundtrip(tmp_path):
    profile = fit_profile(linear_samples(), note="unit test")
    path = save_profile(profile, tmp_path / "profile.json")
    loaded = load_profile(path)
    assert loaded.model == profile.model
    assert loaded.params == pytest.approx(profile.params)
    assert len(loaded.samples) == len(profile.samples)
    assert loaded.note == "unit test"


def test_missing_profile_raises(tmp_path):
    with pytest.raises(CalibrationError, match="not found"):
        load_profile(tmp_path / "none.json")


def test_malformed_profile_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{\"model\": \"linear\"}")
    with pytest.raises(CalibrationError):
        load_profile(p)


def test_holdout_keeps_distances_disjoint():
    samples = linear_samples()
    train, val = split_holdout(samples, val_fraction=0.3)
    train_d = {s.d_true for s in train}
    val_d = {s.d_true for s in val}
    assert train_d and val_d
    assert train_d.isdisjoint(val_d)


def test_samples_persistence(tmp_path):
    samples = linear_samples()
    path = save_samples(samples, tmp_path / "samples.json")
    loaded = load_samples(path)
    assert [(s.r, s.d_true) for s in loaded] == [(s.r, s.d_true) for s in samples]


def test_predict_model_rejects_unknown():
    with pytest.raises(CalibrationError):
        predict_model("cubic", (1.0, 2.0), 1.0)


def test_predict_model_singularity():
    with pytest.raises(CalibrationError, match="Singular"):
        predict_model("reciprocal", (1.0, -1.0), 1.0)
