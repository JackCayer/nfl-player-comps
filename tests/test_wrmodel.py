import numpy as np
import pandas as pd

from src.wrmodel import add_recent_features, assign_flags, feature_names, walk_forward


def toy_games(n=14, pid="a"):
    return pd.DataFrame({
        "player_id": pid, "season": 2025, "week": range(1, n + 1),
        "fp": np.arange(1.0, n + 1), "targets": 6.0, "wopr": 0.3,
    })


def test_recent_average_excludes_current_game():
    df = toy_games()
    out = add_recent_features(df, window=8, min_periods=4)
    # game 9 (index 8) should be the mean of games 1-8 = 4.5, not including itself
    assert out.loc[8, "fp_r"] == np.mean(range(1, 9))
    # too little history -> no feature
    assert out["fp_r"].iloc[:4].isna().all()


def test_changing_a_games_result_never_changes_its_own_features():
    df = toy_games()
    base = add_recent_features(df)["fp_r"]
    df2 = df.copy()
    df2.loc[8, "fp"] = 1000.0                        # rewrite game 9's result
    changed = add_recent_features(df2)["fp_r"]
    assert changed.loc[8] == base.loc[8]             # game 9's own features unchanged
    assert changed.loc[9] != base.loc[9]             # but game 10 sees it, as it should


def test_walk_forward_ignores_the_future():
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(30):
        for week in range(1, 19):
            fp_r = rng.uniform(2, 15)
            rows.append({"player_id": pid, "season": 2025, "week": week, "t": 202500 + week,
                         "fp_r": fp_r, "targets_r": rng.uniform(4, 10), "wopr_r": rng.uniform(0.1, 0.6),
                         "fp": fp_r + rng.normal(0, 4)})
    bt = pd.DataFrame(rows)
    feats = feature_names()
    p1 = walk_forward(bt, feats, min_train=50)

    bt2 = bt.copy()
    later = bt2["t"] > 202510
    bt2.loc[later, "fp"] = bt2.loc[later, "fp"] + 500  # corrupt everything after week 10
    p2 = walk_forward(bt2, feats, min_train=50)

    early = bt["t"] <= 202510
    pd.testing.assert_series_equal(p1[early], p2[early])


def test_flags_are_relative_tails():
    df = pd.DataFrame({"usage_effect": np.linspace(-2, 2, 100)})
    flags = assign_flags(df, q=0.10)
    assert (flags == "sell-high").sum() == 10
    assert (flags == "buy-low").sum() == 10
    assert flags[df["usage_effect"].idxmin()] == "sell-high"
    assert flags[df["usage_effect"].idxmax()] == "buy-low"
