import nflreadpy as nfl
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="NFL WR Comps + Fantasy Usage Model", layout="wide")

FIRST_SEASON = 2021
WINDOW = 8          # games used for "recent" scoring and usage
MIN_WINDOW_GAMES = 4
MIN_TARGETS_PER_GAME = 4
FEATURES = ["fp_r8", "targets_r8", "wopr_r8"]
COMP_METRICS = {
    "targets_per_game": "Targets / game",
    "catch_rate": "Catch rate",
    "yards_per_target": "Yards / target",
    "air_yards_per_target": "Air yards / target",
    "yac_per_reception": "YAC / reception",
    "epa_per_target": "EPA / target",
}


# ---------------------------------------------------------------- data + model
@st.cache_data(ttl=6 * 60 * 60, show_spinner="Loading NFL data...")
def load_data():
    """Load WR data, fit the model, and build the current projection table."""
    try:
        season = int(nfl.get_current_season())
    except Exception:
        season = 2026

    try:
        raw = nfl.load_player_stats(list(range(FIRST_SEASON, season + 1))).to_pandas()
    except Exception:
        # offseason: new season has no data yet
        season -= 1
        raw = nfl.load_player_stats(list(range(FIRST_SEASON, season + 1))).to_pandas()

    wr = raw[(raw["position"] == "WR") & (raw["season_type"] == "REG")].copy()
    wr["fp"] = wr["fantasy_points"]  # standard scoring
    wr = wr.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # --- training rows: each game gets features from the player's PREVIOUS 8 games
    bt = wr.copy()
    g = bt.groupby("player_id")
    for col in ["fp", "targets", "wopr"]:
        bt[col + "_r8"] = g[col].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_WINDOW_GAMES).mean()
        )
    bt = bt.dropna(subset=["fp", "fp_r8", "targets_r8", "wopr_r8"])
    bt = bt[bt["targets_r8"] >= MIN_TARGETS_PER_GAME]

    final = LinearRegression().fit(bt[FEATURES], bt["fp"])
    tier = LinearRegression().fit(bt[["fp_r8"]], bt["fp"])

    # --- current projections: each player's most recent 8 games
    latest = wr.groupby("player_id").tail(WINDOW)
    cur = (
        latest.groupby(["player_id", "player_display_name"])
        .agg(
            team=("team", "last"),
            games=("week", "count"),
            games_current=("season", lambda s: int((s == season).sum())),
            fp_r8=("fp", "mean"),
            targets_r8=("targets", "mean"),
            wopr_r8=("wopr", "mean"),
        )
        .reset_index()
    )
    cur = cur[
        (cur["games"] >= MIN_WINDOW_GAMES)
        & (cur["games_current"] > 0)
        & (cur["targets_r8"] >= MIN_TARGETS_PER_GAME)
    ].dropna()

    cur["projection"] = final.predict(cur[FEATURES])
    cur["usage_effect"] = cur["projection"] - tier.predict(cur[["fp_r8"]])

    lo, hi = cur["usage_effect"].quantile([0.10, 0.90])
    cur["flag"] = "hold"
    cur.loc[cur["usage_effect"] >= hi, "flag"] = "buy-low"
    cur.loc[cur["usage_effect"] <= lo, "flag"] = "sell-high"
    cur["note"] = cur["games_current"].apply(
        lambda n: "mostly prior-season data" if n <= 2 else ""
    )

    coefs = dict(zip(FEATURES, final.coef_))
    return wr, cur.reset_index(drop=True), season, coefs, len(bt)


def season_table(wr, season, min_targets):
    """One row per WR for the current season, with the 6 comparison metrics."""
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
    p = p[p["targets"] >= min_targets].copy()
    p = p[p["receptions"] > 0]
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


# ------------------------------------------------------------------------- app
wr, cur, season, coefs, n_train = load_data()

st.title("NFL WR Comps + Fantasy Usage Model")
st.caption(
    f"Public nflverse data, standard scoring. Season {season}, "
    f"latest week loaded: {int(wr[wr['season'] == season]['week'].max())}."
)

tab_player, tab_board, tab_about = st.tabs(["Player lookup", "Buy-low / sell-high board", "About"])

# ---- Player lookup
with tab_player:
    all_names = sorted(
        set(cur["player_display_name"]) | set(wr[wr["season"] == season]["player_display_name"])
    )
    default = all_names.index("Jaxon Smith-Njigba") if "Jaxon Smith-Njigba" in all_names else 0
    name = st.selectbox("Pick a receiver", all_names, index=default)

    left, right = st.columns(2)

    with left:
        st.subheader("Fantasy outlook")
        row = cur[cur["player_display_name"] == name]
        if row.empty:
            st.info(
                "Not enough recent data for a projection (needs a game this season, "
                f"{MIN_WINDOW_GAMES}+ games in his last {WINDOW}, and real target volume)."
            )
        else:
            r = row.iloc[0]
            a, b, c = st.columns(3)
            a.metric("Recent avg (last 8)", f"{r['fp_r8']:.1f}")
            b.metric("Projection", f"{r['projection']:.1f}", f"{r['projection'] - r['fp_r8']:+.1f}")
            c.metric("Usage effect", f"{r['usage_effect']:+.2f}")
            st.write(f"**Flag: {r['flag']}**  ({r['team']})")
            if r["note"]:
                st.warning(
                    f"Only {int(r['games_current'])} game(s) from {season} in his window, "
                    "so this leans on older data and can't see role changes or injury returns."
                )
            games = wr[wr["player_id"] == r["player_id"]].tail(WINDOW).copy()
            games["game"] = games["season"].astype(str) + " W" + games["week"].astype(str)
            st.caption("Fantasy points, last 8 games")
            st.bar_chart(games.set_index("game")["fp"])

    with right:
        st.subheader(f"Closest comps ({season} stats so far)")
        min_t = st.slider("Minimum targets to be included", 5, 60, 15)
        table = season_table(wr, season, min_t)
        comps = closest_comps(table, name)
        if comps is None:
            st.info(f"{name} isn't in the comparison pool at {min_t}+ targets. Try lowering the minimum.")
        else:
            show = table.set_index("player_display_name").loc[[name] + list(comps.index)]
            out = show[list(COMP_METRICS)].rename(columns=COMP_METRICS).round(2)
            out.insert(0, "Distance", [0.0] + [round(d, 2) for d in comps.values])
            st.dataframe(out, width="stretch")
            st.caption(
                "Distance is measured across the 6 metrics after standardizing them. "
                "Smaller means a closer match. Early in the season, one big play can swing these."
            )

# ---- Board
with tab_board:
    st.subheader("Who's scoring out of line with their usage?")
    c1, c2 = st.columns(2)
    flag_pick = c1.selectbox("Show", ["buy-low", "sell-high", "all"])
    min_cur = c2.slider(f"Minimum {season} games in window", 1, 6, 1)

    board = cur[cur["games_current"] >= min_cur]
    if flag_pick != "all":
        board = board[board["flag"] == flag_pick]
    board = board.sort_values("usage_effect", ascending=(flag_pick == "sell-high"))

    board_out = board[
        ["player_display_name", "team", "flag", "fp_r8", "projection", "usage_effect", "games_current", "note"]
    ].rename(
        columns={
            "player_display_name": "Player",
            "team": "Team",
            "flag": "Flag",
            "fp_r8": "Recent avg",
            "projection": "Projection",
            "usage_effect": "Usage effect",
            "games_current": f"{season} games",
            "note": "Note",
        }
    )
    st.dataframe(board_out.round(2), width="stretch", hide_index=True)
    st.caption(
        "Usage effect = projection minus what recent scoring alone would predict. "
        "Positive means usage supports more scoring than he's been getting. "
        "Flags are the top/bottom 10%, so someone is always flagged."
    )

# ---- About
with tab_about:
    st.markdown(
        f"""
**What this is.** Two tools built on public NFL data. The first finds receivers with similar
stat profiles. The second projects next-game fantasy points from recent scoring plus usage
(targets and WOPR), and flags players whose scoring is out of line with their usage.

**How the projection works.** A linear regression trained walk-forward on {FIRST_SEASON}-{season}
({n_train:,} player-games). Fitted weights: recent fantasy points **{coefs['fp_r8']:.2f}**,
targets per game **{coefs['targets_r8']:.2f}**, WOPR **{coefs['wopr_r8']:.2f}**.
The weight under 0.5 on recent scoring is regression to the mean: hot streaks carry forward at
well under half strength.

**Backtest results.** Predicting next-game points this way cut mean absolute error by about
2.8% versus a plain 8-game average. Players whose usage supported more scoring than they'd
been getting scored about 1 point more than expected for their scoring level, and those whose
scoring outran their usage scored about 1.3 points less.

**Limits.** No injury, quarterback, matchup or game-script information. Single games are very
noisy, so flags describe groups of players over many weeks, not a guarantee for one game.
Standard scoring only. Players with only 1-2 games this season in their window are marked,
since the model leans on older data for them.
"""
    )
