"""Synthetic, dependency-light QA for paired GeoC versus peer LOOCV contracts."""
import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.peer_champions_loo import (
    BASELINE, ID, MODELS, Y,
    align_peer_targets, outer_record, summarize_complete, validate_fold,
)


def _targets():
    ids = [f"P{i:03d}" for i in range(138)]
    y = pd.Series(np.linspace(2, 6, 138), index=ids)
    peer = pd.DataFrame({ID: ids[::-1], Y: y.iloc[::-1].to_numpy()})
    return peer, y


def test_alignment_reorders_without_changing_targets():
    peer, expected = _targets()
    assert align_peer_targets(peer, expected) == peer[ID].tolist()
    wrong = peer.copy()
    wrong.loc[0, Y] += .2
    with pytest.raises(AssertionError):
        align_peer_targets(wrong, expected)
    with pytest.raises(ValueError, match="IDs differ"):
        align_peer_targets(peer.assign(ID=lambda d: [*d[ID].iloc[:-1], "FOREIGN"]), expected)


def test_each_outer_and_inner_split_excludes_current_target():
    for held in range(138):
        record = outer_record(held)
        tr = np.array(record["train_positions"])
        assert len(tr) == 137 and held not in tr
        assert record["validation_positions"] == [held]
        seen = []
        for split in record["inner"]:
            a = np.array(split["train_relative_positions"])
            b = np.array(split["validation_relative_positions"])
            assert set(a).isdisjoint(b)
            assert len(a) > 0 and len(b) > 0
            assert held not in tr[a] and held not in tr[b]
            seen.extend(b.tolist())
        assert sorted(seen) == list(range(137))


def _predictions():
    peer, y = _targets()
    result = pd.DataFrame({ID: y.index, "observed": y.to_numpy()})
    result[BASELINE] = y.to_numpy() + .2
    for j, name in enumerate(MODELS[1:]):
        result[name] = y.to_numpy() + .1 + j * .01
    return result


def test_paired_summary_reports_improvement_with_same_138_parcels():
    predictions = _predictions()
    scores, paired = summarize_complete(predictions, bootstraps=100)
    assert len(scores) == 5 and len(paired) == 4
    assert scores.iloc[0]["model"] == "CT3"
    assert paired["delta_rmse"].lt(0).all()
    assert paired["fraction_parcels_peer_better"].eq(1).all()
    assert scores["training_labels_per_fit"].eq(137).all()


def test_cached_fold_must_match_id_label_and_all_predictions():
    full = _predictions()
    fold = full.iloc[[0]]
    validate_fold(fold, expected_id="P000", expected_y=float(fold.iloc[0].observed))
    with pytest.raises(ValueError, match="wrong ID"):
        validate_fold(fold, expected_id="P001", expected_y=float(fold.iloc[0].observed))
    with pytest.raises(ValueError, match="schema"):
        validate_fold(
            fold.drop(columns=["CT3"]),
            expected_id="P000", expected_y=float(fold.iloc[0].observed),
        )
    bad = fold.copy()
    bad.loc[0, "Reference26"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_fold(bad, expected_id="P000", expected_y=float(fold.iloc[0].observed))
    with pytest.raises(ValueError, match="138"):
        summarize_complete(full.iloc[:-1], bootstraps=100)
