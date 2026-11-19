"""Lightweight LODO re-classification for cascade-with-referees,
applying the SYMMETRIC "unknown" decision rule:

    a referee-bucket disambiguation step outputs "unknown" when
    every surviving referee scores <= 0 — exactly the same
    convention as the no-cascade architecture
    (max OvR score <= 0 -> "unknown").

Reads the existing panels + LDA weights from
honest_lodo_cascade_with_referees_panels.csv (produced by
honest_lodo_cascade_with_referees.py), so no panel search and no
LDA refit is needed.  Runtime: < 1 minute total.

Output:
    paper_figures/honest_lodo_cascade_with_referees_summary.csv
    (overwrites the previous summary)
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import load_cohort, panel_scores  # noqa: E402

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
TRIO = {"CRC", "LUNGC", "OVC"}
RES4 = {"BRC", "CVX", "ENDC", "PRC"}


# -------------------------- load cohort --------------------------------

# All 1273 patients x 1462 proteins, NaN kept (per-panel complete case).
X_use_df, y_use, BLOCKS = load_cohort()
X_raw = X_use_df.to_numpy(dtype=float)
DENSE = list(X_use_df.columns)          # name kept: all proteins
P_INDEX = {p: i for i, p in enumerate(DENSE)}
ALL_12 = sorted(set(y_use))
print(f"Cohort: {len(X_use_df)} samples")


# -------------------------- load panels with weights ------------------

panels_df = pd.read_csv(
    OUT / "honest_lodo_cascade_with_referees_panels.csv")


def cascade_predict_with_unknown(scores, held_out):
    """Cascade-with-referees + symmetric unknown rule."""
    n = next(iter(scores.values())).shape[0]
    preds = np.empty(n, dtype=object)

    trio_alive = [(t, c) for t, c in [
        ("T41_CRC", "CRC"), ("T42_LUNGC", "LUNGC"),
        ("T43_OVC", "OVC")] if c != held_out and t in scores]
    res4_alive = [(t, c) for t, c in [
        ("T71_BRC", "BRC"), ("T72_CVX", "CVX"),
        ("T73_ENDC", "ENDC"), ("T74_PRC", "PRC")]
        if c != held_out and t in scores]

    def s_of(tid, i):
        return scores.get(tid, np.full(n, -np.inf))[i]

    def disambiguate(alive, i):
        """argmax over surviving referees; unknown if all <= 0."""
        if not alive:
            return "unknown"
        # NaN = referee not available for this patient (not positive)
        rs = [(s_of(t, i), c) for t, c in alive
              if not np.isnan(s_of(t, i))]
        if not rs:
            return "unknown"
        best_score, best_class = max(rs)
        return best_class if best_score > 0 else "unknown"

    for i in range(n):
        if s_of("T1_MYEL", i) > 0:
            preds[i] = "MYEL"
        elif s_of("T3_CLL", i) > 0:
            preds[i] = ("LYMPH" if s_of("T6_CLLvsLYMPH", i) > 0
                        else "CLL")
        elif s_of("T2_AML", i) > 0:
            preds[i] = "AML"
        elif s_of("T4_trio", i) > 0:
            preds[i] = disambiguate(trio_alive, i)
        elif s_of("T5_GLIOM", i) > 0:
            preds[i] = "GLIOM"
        elif s_of("T7_LYMPH", i) > 0:
            preds[i] = "LYMPH"
        else:
            preds[i] = disambiguate(res4_alive, i)
    return preds


# -------------------------- per-held-out routing ----------------------

print("\n=== Reclassify with symmetric 'unknown' rule ===")
summary_rows = []
for held in ALL_12:
    rows = panels_df[panels_df["held_out"] == held]
    if rows.empty:
        continue
    scores = {}
    for _, r in rows.iterrows():
        panel = r["panel"].split(",")
        w = np.array([float(x) for x in r["weights"].split(",")])
        b = float(r["intercept"])
        scores[r["test_id"]] = panel_scores(
            X_raw, [P_INDEX[p] for p in panel], w, b)

    preds = cascade_predict_with_unknown(scores, held)
    # routing reported on the 1094 complete patients
    held_mask = (y_use == held) & (BLOCKS[1] == 0)
    held_preds = preds[held_mask]
    routing = pd.Series(held_preds).value_counts().to_dict()
    n_held = int(held_mask.sum())

    pct_unknown = 100 * routing.get("unknown", 0) / n_held

    # Bucket-graceful for trio/residual4 held-outs:
    # any non-held bucket sibling counts as correct
    if held in TRIO:
        n_correct = sum(routing.get(c, 0) for c in TRIO if c != held)
        pct_correct_bucket = 100 * n_correct / n_held
    elif held in RES4:
        n_correct = sum(routing.get(c, 0) for c in RES4 if c != held)
        pct_correct_bucket = 100 * n_correct / n_held
    else:
        pct_correct_bucket = None

    cnt = pd.Series(held_preds).value_counts()
    top_routes = ", ".join(
        f"{c} ({100*cnt[c]/n_held:.0f}%)"
        for c in cnt.index[:3])

    summary_rows.append({
        "held_out": held, "n": n_held,
        "n_samples": int((BLOCKS[1] == 0).sum()),
        "n_tests_retained": len(rows),
        "pct_unknown": round(pct_unknown, 1),
        "pct_correct_bucket": (round(pct_correct_bucket, 1)
                               if pct_correct_bucket is not None
                               else "—"),
        "top_routes": top_routes,
        "routing": str(routing),
    })
    cb_str = (f"correct_bucket {pct_correct_bucket:.1f}%"
              if pct_correct_bucket is not None else "(no bucket)")
    print(f"  {held:>6}: unknown {pct_unknown:5.1f}%  "
          f"{cb_str:>26}   top: {top_routes}")

pd.DataFrame(summary_rows).to_csv(
    OUT / "honest_lodo_cascade_with_referees_summary.csv", index=False)
print(f"\nSaved {OUT/'honest_lodo_cascade_with_referees_summary.csv'}")
