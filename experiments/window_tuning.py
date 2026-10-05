"""Does the 'recent games' window matter? Compare windows on EXACTLY the same games."""
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.wrmodel import (load_wr, make_backtest_frame, evaluate, summarize, feature_names)

wr, season = load_wr()
feats = feature_names()

# 1) reproduce the headline result with the shared code
bt8 = make_backtest_frame(wr, window=8)
res8 = evaluate(bt8, {"direct": feats})
print("Reproduce (window=8):")
print(summarize(res8), "\n")

# 2) window sweep. Build every frame without a volume filter, then keep only the
#    games that are valid for ALL settings, so each setting is scored on identical rows.
settings = {f"roll{w}": dict(window=w) for w in [4, 6, 8, 10, 12, 16]}
settings.update({f"ewm{h}": dict(window=8, halflife=h) for h in [2, 3, 5]})

frames = {k: make_backtest_frame(wr, min_targets=0, **v) for k, v in settings.items()}
ref = make_backtest_frame(wr, window=8)                       # same relevance rule as before
common = ref.index
for f in frames.values():
    common = common.intersection(f.index)
print(f"{len(common)} games scored identically for every setting\n")

rows = {}
for name, f in frames.items():
    sub = f.loc[common]
    r = evaluate(sub, {"direct": feats})
    rows[name] = {
        "naive MAE": (r["actual"] - r["naive"]).abs().mean(),
        "direct MAE": (r["actual"] - r["direct"]).abs().mean(),
        "n": len(r),
    }
out = pd.DataFrame(rows).T
out["gain vs own naive %"] = (out["direct MAE"] / out["naive MAE"] - 1) * 100
print(out.round(3).sort_values("direct MAE"))

# 3) accuracy isn't the only job: do the flags still separate buy-low from sell-high?
from src.wrmodel import flag_deciles
print("\nFlag check (avg pts above what recent scoring alone predicts; decile 0 = sell, 9 = buy)")
for name, kw in [("roll8", dict(window=8)), ("ewm5", dict(halflife=5))]:
    d = flag_deciles(make_backtest_frame(wr, **kw), feats)
    print(f"{name}: {d['mean'].tolist()}  spread 9-0 = {d['mean'].iloc[-1] - d['mean'].iloc[0]:.2f}")
