"""Shared data, features, model and backtest code for the WR project.

Everything the notebooks and the Streamlit app need lives here, so the
numbers can't quietly drift apart between them.
"""
import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FIRST_SEASON = 2021
BASE_COLS = ["fp", "targets", "wopr"]          # columns we build "recent" versions of
COMP_METRICS = {
    "targets_per_game": "Targets / game",
    "catch_rate": "Catch rate",
    "yards_per_target": "Yards / target",
    "air_yards_per_target": "Air yards / target",
    "yac_per_reception": "YAC / reception",
    "epa_per_target": "EPA / target",
}


# ------------------------------------------------------------------ data
def load_wr(first_season=FIRST_SEASON):
    """Regular-season WR game logs (standard scoring in `fp`) and the current season."""
    try:
        season = int(nfl.get_current_season())
    except Exception:
        season = 2026
    try:
        raw = nfl.load_player_stats(list(range(first_season, season + 1))).to_pandas()
    except Exception:
        season -= 1                              # offseason: new season has no data yet
        raw = nfl.load_player_stats(list(range(first_season, season + 1))).to_pandas()

    wr = raw[(raw["position"] == "WR") & (raw["season_type"] == "REG")].copy()
    wr["fp"] = wr["fantasy_points"]
    wr["t"] = wr["season"] * 100 + wr["week"]    # orders games in time
    wr = wr.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    return wr, season


# -------------------------------------------------------------- features
def add_recent_features(wr, window=8, min_periods=4, cols=BASE_COLS, halflife=None):
    """Add `<col>_r`: the player's average over his PREVIOUS games only.

    shift(1) removes the current game, so a row never sees its own result.
    With `halflife` set, recent games are weighted more (exponential average).
    """
    out = wr.copy()
    g = out.groupby("player_id")
    for c in cols:
        if halflife is None:
            out[c + "_r"] = g[c].transform(
                lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
            )
        else:
            out[c + "_r"] = g[c].transform(
                lambda s: s.shift(1).ewm(halflife=halflife, min_periods=min_periods).mean()
            )
    return out


def make_backtest_frame(wr, window=8, min_periods=4, min_targets=4, cols=BASE_COLS, halflife=None):
    """Rows that can be predicted: enough history and real target volume."""
    bt = add_recent_features(wr, window, min_periods, cols, halflife)
    feats = [c + "_r" for c in cols]
    bt = bt.dropna(subset=["fp"] + feats)
    if min_targets:
        bt = bt[bt["targets_r"] >= min_targets]
    return bt


def feature_names(cols=BASE_COLS):
    return [c + "_r" for c in cols]


# -------------------------------------------------------------- backtest
def walk_forward(bt, features, target="fp", min_train=500):
    """Predict each week using a model fit ONLY on earlier weeks."""
    pred = pd.Series(np.nan, index=bt.index)
    for t in sorted(bt["t"].unique()):
        train = bt[bt["t"] < t]
        mask = bt["t"] == t
        if len(train) < min_train or not mask.any():
            continue
        model = LinearRegression().fit(train[features], train[target])
        pred[mask] = model.predict(bt.loc[mask, features])
    return pred


def evaluate(bt, feature_sets, min_train=500):
    """Walk-forward predictions for each feature set, plus the plain-average baseline.

    feature_sets: {"name": [feature columns]}. Returns one row per tested game.
    """
    res = pd.DataFrame({"actual": bt["fp"], "season": bt["season"], "t": bt["t"]})
    res["naive"] = bt["fp_r"]
    for name, feats in feature_sets.items():
        res[name] = walk_forward(bt, feats, min_train=min_train)
    return res.dropna()


def summarize(res):
    """MAE for every predictor and % change versus the plain average."""
    preds = [c for c in res.columns if c not in ("actual", "season", "t")]
    mae = pd.Series({c: (res["actual"] - res[c]).abs().mean() for c in preds})
    table = pd.DataFrame({"MAE": mae})
    table["vs naive %"] = (table["MAE"] / table.loc["naive", "MAE"] - 1) * 100
    return table.round(3)


def by_season(res):
    preds = [c for c in res.columns if c not in ("actual", "season", "t")]
    return pd.DataFrame(
        {c: (res["actual"] - res[c]).abs().groupby(res["season"]).mean() for c in preds}
    ).round(3)


# ----------------------------------------------------------- projections
def assign_flags(df, col="usage_effect", q=0.10):
    """Bottom q of `col` -> sell-high, top q -> buy-low (relative, so always some flags)."""
    lo, hi = df[col].quantile([q, 1 - q])
    flag = pd.Series("hold", index=df.index)
    flag[df[col] >= hi] = "buy-low"
    flag[df[col] <= lo] = "sell-high"
    return flag


def project_current(wr, bt, season, window=8, min_games=4, min_targets=4, features=None, halflife=None):
    """Fit on all history, then project each player from his latest `window` games."""
    features = features or feature_names()
    final = LinearRegression().fit(bt[features], bt["fp"])
    tier = LinearRegression().fit(bt[["fp_r"]], bt["fp"])

    latest = wr.groupby("player_id").tail(window)
    cur = (
        latest.groupby(["player_id", "player_display_name"])
        .agg(
            team=("team", "last"),
            games=("week", "count"),
            games_current=("season", lambda s: int((s == season).sum())),
            fp_r=("fp", "mean"),
            targets_r=("targets", "mean"),
            wopr_r=("wopr", "mean"),
        )
        .reset_index()
    )
    if halflife is not None:
        # match how the model was trained: exponentially weighted over ALL prior games
        g = wr.groupby("player_id")
        ew = pd.DataFrame({c + "_r": g[c].transform(
            lambda x: x.ewm(halflife=halflife, min_periods=min_games).mean()) for c in BASE_COLS})
        last = wr[["player_id"]].join(ew).groupby("player_id").tail(1).set_index("player_id")
        for c in BASE_COLS:
            cur[c + "_r"] = cur["player_id"].map(last[c + "_r"])

    cur = cur[
        (cur["games"] >= min_games) & (cur["games_current"] > 0) & (cur["targets_r"] >= min_targets)
    ].dropna()

    cur["projection"] = final.predict(cur[features])
    cur["usage_effect"] = cur["projection"] - tier.predict(cur[["fp_r"]])
    cur["flag"] = assign_flags(cur)
    cur["note"] = cur["games_current"].apply(lambda n: "mostly prior-season data" if n <= 2 else "")
    coefs = dict(zip(features, final.coef_))
    return cur.reset_index(drop=True), coefs


# ------------------------------------------------------------ comparison
def season_table(wr, season, min_targets):
    """One row per WR for one season, with the 6 comparison metrics."""
    s = wr[wr["season"] == season]
    p = (
        s.groupby(["player_id", "player_display_name"])
        .agg(
            team=("team", "last"),
            games=("week", "nunique"),
            targets=("targets", "sum"),
            receptions=("receptions", "sum"),
            yards=("receiving_yards", "sum"),
            air=("receiving_air_yards", "sum"),
            yac=("receiving_yards_after_catch", "sum"),
            epa=("receiving_epa", "sum"),
        )
        .reset_index()
    )
    p = p[(p["targets"] >= min_targets) & (p["receptions"] > 0)].copy()
    p["targets_per_game"] = p["targets"] / p["games"]
    p["catch_rate"] = p["receptions"] / p["targets"]
    p["yards_per_target"] = p["yards"] / p["targets"]
    p["air_yards_per_target"] = p["air"] / p["targets"]
    p["yac_per_reception"] = p["yac"] / p["receptions"]
    p["epa_per_target"] = p["epa"] / p["targets"]
    return p.reset_index(drop=True)


def closest_comps(table, name, n=8):
    cols = list(COMP_METRICS)
    z = (table[cols] - table[cols].mean()) / table[cols].std()
    z.index = table["player_display_name"].values
    if name not in z.index:
        return None
    dist = ((z - z.loc[name]) ** 2).sum(axis=1) ** 0.5
    return dist.drop(name).sort_values().head(n)


# ------------------------------------------------------------ flag check
def flag_deciles(bt, features, min_train=500):
    """Do the flags predict anything? Walk-forward, bucketed into weekly deciles.

    beat_tier = points scored above what recent scoring ALONE predicts.
    Decile 0 = usage says sell, decile 9 = usage says buy.
    """
    parts = []
    for t in sorted(bt["t"].unique()):
        train = bt[bt["t"] < t]
        test = bt[bt["t"] == t].copy()
        if len(train) < min_train or test.empty:
            continue
        full = LinearRegression().fit(train[features], train["fp"])
        tier = LinearRegression().fit(train[["fp_r"]], train["fp"])
        test["usage_effect"] = full.predict(test[features]) - tier.predict(test[["fp_r"]])
        test["beat_tier"] = test["fp"] - tier.predict(test[["fp_r"]])
        parts.append(test)
    v = pd.concat(parts)
    v["decile"] = v.groupby("t")["usage_effect"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    d = v.groupby("decile")["beat_tier"].agg(["mean", "count", "std"])
    d["sem"] = d["std"] / d["count"] ** 0.5
    return d.round(2)
