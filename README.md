# NFL WR Comps + Fantasy Usage Model

Two things in one project, both built on public NFL data (nflverse, via `nflreadpy`):

1. **WR comparison** -- input a player's name, get the closest matches right now
2. **Fantasy buy-low / sell-high flags** -- finds WRs whose fantasy scoring is out of line with their usage, then tests whether that actually predicts what happens next

Built as a learning/portfolio project to practice Python, stats and ML on something I actually follow.

---

## Part 1: WR Comparison

Basic version. Input a player's name and get the closest matches based on 2026 stats so far.

**Stats gathered (2026 regular season, WRs only)**

# of games
# of targets
# of receptions
- total receiving yards
- total receiving TDs
- total receiving air yards
- total YAC
- total receiving EPA

**Comparison metrics**

| Metric | Calculation |
|---|---|
| Targets per game | targets / games |
| Catch rate | receptions / targets |
| Yards per target | total yards / targets |
| Air yards per target | total air yards / targets |
| YAC per reception | total YAC / receptions |
| EPA per target | total EPA / targets |

- Only WRs with 15+ targets are included
- Metrics are standardized with z-scores so no single stat dominates
- Similarity = straight-line (Euclidean) distance across all 6 metrics. Smaller = closer match

**Example: Jaxon Smith-Njigba (as of Oct 4, 2026)**

| Player | Distance |
|---|---|
| Chris Olave | 1.37 |
| Garrett Wilson | 2.36 |
| CeeDee Lamb | 2.40 |
| Amon-Ra St. Brown | 2.50 |
| Christian Watson | 2.64 |
| Jalen Coker | 2.80 |
| Ja'Marr Chase | 2.85 |
| Tee Higgins | 2.91 |
| DeVonta Smith | 3.00 |
| Josh Downs | 3.09 |

Olave is a clear nearest match, the rest are looser comps in the same neighborhood.

**Known limitation:** this is only a few games into the season, so one big play can swing a player's profile. (Early example: Davante Adams matching with Tee Higgins and Matthew Golden because of a big Week 2 with a few deep catches.)

---

## Part 2: Fantasy Usage Model

**Question:** are some WRs scoring more (or less) fantasy points than their usage says they should, and does that predict what happens next game?

Why usage: targets and share of the offense's looks are steadier week to week than touchdowns and long plays, so scoring that's out of line with usage is more likely to correct itself.

### Data and setup

- 2021-2026 regular season WR data, **standard scoring**
- For each player-game, features are built from that player's **previous 8 games only** (no future data leaks in)
- Recent scoring (`fp_r8`), recent targets per game (`targets_r8`), recent WOPR (`wopr_r8`)
- Players need 4+ games in their window and 4+ targets per game to be included
- Tested **walk-forward**: for each week, the model is fit only on earlier weeks, then predicts that week

### The model

Predict next-game fantasy points from recent scoring plus usage (targets and WOPR) with a simple linear regression.

Fitted weights (all seasons):

| Input | Weight |
|---|---|
| Recent fantasy points (8-game avg) | 0.418 |
| Targets per game | 0.217 |
| WOPR | 4.574 |

The 0.42 on recent scoring means a hot streak carries forward at well under half strength. That's regression to the mean.

### Results

**1. Prediction error (mean absolute error, points per game)**

| Predictor | MAE | vs. plain average |
|---|---|---|
| Plain 8-game average | 4.688 | -- |
| Blend toward usage expectation | 4.641 | -1.0% |
| Direct regression (recent points + usage) | 4.559 | -2.8% |

**2. Do the flags mean anything?**

`usage_effect` = the model's projection minus what recent scoring alone would predict. Positive means usage supports more scoring than he's been getting (buy-low), negative means scoring is running ahead of usage (sell-high). Players were bucketed into deciles each week; the table shows how many points they scored above or below what their scoring level alone predicted.

| Decile (0 = sell-high, 9 = buy-low) | Avg pts vs. expected | Games |
|---|---|---|
| 0 | -1.34 | 551 |
| 1 | -0.57 | 514 |
| 2 | -0.40 | 505 |
| 3 | -0.54 | 508 |
| 4 | -0.07 | 521 |
| 5 | +0.63 | 494 |
| 6 | +0.25 | 498 |
| 7 | +0.66 | 515 |
| 8 | +0.76 | 504 |
| 9 | +1.06 | 539 |

- Players whose usage supported higher scoring than their recent results scored about **1 point more** than expected for their scoring level
- Players whose scoring outran their usage scored about **1.3 points less**
- Spread of ~2.4 points between the two ends, well outside noise (roughly 500 games per decile)

**3. Not just regression to the mean**

Top scorers drop off anyway. To check the usage gap adds something beyond that, I compared players with the *same recent scoring level* who differed in how much their usage backed it up. In all 4 scoring tiers, the high-gap players scored 1.1 to 1.6 points less the next game.

### What didn't work

- **Adjusting for a player's own past efficiency** (the idea being stars like Smith-Njigba always beat their usage) did not improve predictions, even with 6 seasons of data
- **First version of the flags** labeled almost every top WR "sell-high" because top scorers regress by default. Fixed by flagging on the usage effect (relative to scoring level) instead of the raw projected drop

### Example output (as of Oct 4, 2026)

| Player | Flag | Recent avg | Projection | Usage effect |
|---|---|---|---|---|
| Garrett Wilson | buy-low | 9.91 | 10.10 | +1.41 |
| Wan'Dale Robinson | buy-low | 8.29 | 8.84 | +1.22 |
| Justin Jefferson | buy-low | 7.14 | 7.80 | +0.94 |
| Christian Watson | sell-high | 13.69 | 10.24 | -0.94 |
| Alec Pierce | sell-high | 9.55 | 7.29 | -1.16 |

Players with only 1-2 games from 2026 in their window are marked **"mostly 2025 data"**, since the model can't see role changes, injury returns or new offenses for them.

### Limitations

- Only uses recent scoring and usage. **No injury, quarterback, matchup or game-script info**
- Single games are very noisy (std is 5-6 points). Flags describe groups of players over many weeks, not a guarantee for any one game
- Standard scoring only for now
- "Games" = weeks where a player recorded a stat, not games he was on the roster for
- Flags are relative (top/bottom 10%), so some players are flagged every week
- Players with no recorded 2026 stats are excluded

---

## How to run

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install nflreadpy pandas scikit-learn pyarrow matplotlib jupyter
```

Notebooks, in order:

1. `01-wr-similarity.ipynb` -- Part 1, WR comparison
2. `02-wr-fantasy.ipynb` -- early fantasy exploration
3. `03-multiseason-backtest.ipynb` -- backtest, final model and flags

Data refreshes every time you rerun (pulled fresh from nflverse).

## Next up

- Streamlit page: pick a player, see projection, flag and closest comps in one place
- PPR / half-PPR switch
- Add quarterback and matchup info
- Other positions (RB, TE)
- Rerun Part 1 on a longer window once more of 2026 is played

---

*Data: [nflverse](https://github.com/nflverse) via `nflreadpy`.*
