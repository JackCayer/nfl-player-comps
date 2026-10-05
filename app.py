import pandas as pd
import streamlit as st

from src.wrmodel import (
    COMP_METRICS, FIRST_SEASON, closest_comps, load_wr, make_backtest_frame,
    project_current, season_table,
)

st.set_page_config(page_title="NFL WR Comps + Fantasy Usage Model", layout="wide")

WINDOW = 8              # games used for "recent" scoring and usage
MIN_WINDOW_GAMES = 4


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Loading NFL data...")
def load_data():
    wr, season = load_wr()
    bt = make_backtest_frame(wr, window=WINDOW)
    cur, coefs = project_current(wr, bt, season, window=WINDOW)
    return wr, cur, season, coefs, len(bt)


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
            a.metric("Recent avg (last 8)", f"{r['fp_r']:.1f}")
            b.metric("Projection", f"{r['projection']:.1f}", f"{r['projection'] - r['fp_r']:+.1f}")
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
        ["player_display_name", "team", "flag", "fp_r", "projection", "usage_effect", "games_current", "note"]
    ].rename(
        columns={
            "player_display_name": "Player",
            "team": "Team",
            "flag": "Flag",
            "fp_r": "Recent avg",
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
({n_train:,} player-games). Fitted weights: recent fantasy points **{coefs['fp_r']:.2f}**,
targets per game **{coefs['targets_r']:.2f}**, WOPR **{coefs['wopr_r']:.2f}**.
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
