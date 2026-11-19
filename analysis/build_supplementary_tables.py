"""Assemble Supplementary Tables S1-S8 from the current analysis outputs
and write them to `Supplementary materials/`.

    S1  per-cancer AUC of the no-cascade k = 8 tests vs Alvez
            paper_figures/pairseed_v2_k8_per_cancer_auc.csv
    S2  panel stability (30 random 80% subsamples)
            paper_figures/pairseed_panel_stability_{runs,frequency}.csv
            paper_figures/pairseed_ovr_k8_panels.csv

    S3  PairSeed no-cascade family (k_target sweep)
            paper_figures/pairseed_no_cascade_family.csv
    S4  Alvez-mimicked family
            paper_figures/alvez_mimicked_family.csv
    S5  deployable formulas of all 14 cascade tests (T1-T7 + referees
        T41-T43, T71-T74): intercept + raw-NPX weight per protein
        (wide format, 0 = protein not in the panel)
            paper_figures/pairseed_cascade_k8_formulas.csv
            paper_figures/bucket_referee_panels.csv
    S6  PairSeed cascade-with-referees family (target_k sweep)
            PairSeed_cascades/cascade_with_referees_family.csv
    S7  per-fold panels of the no-cascade LODO experiment
            paper_figures/honest_lodo_pairseed_nocascade_panels.csv
    S8  per-fold panels of the cascade-with-referees LODO experiment
            paper_figures/honest_lodo_cascade_with_referees_panels.csv

Run from the repository root:  python analysis/build_supplementary_tables.py
"""
from pathlib import Path

import pandas as pd

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
SUPP = Path("Supplementary materials")
SUPP.mkdir(exist_ok=True)


def table_formulas():
    main = pd.read_csv(OUT / "pairseed_cascade_k8_formulas.csv")
    meta = ["test_id", "training_cohort", "positive_classes",
            "panel_size", "intercept"]
    main.insert(1, "role", "main")

    ref = pd.read_csv(OUT / "bucket_referee_panels.csv")
    rows = []
    for _, r in ref.iterrows():
        members = sorted(set(r["positives"].split(",")) |
                         set(r["negatives"].split(",")))
        row = {"test_id": r["test_id"], "role": "referee",
               "training_cohort": "only:" + ",".join(members),
               "positive_classes": r["positives"],
               "panel_size": int(r["panel_size"]),
               "intercept": float(r["intercept"])}
        for p, w in zip(r["panel"].split(","),
                        map(float, r["weights"].split(","))):
            row[p] = w
        rows.append(row)
    ref_w = pd.DataFrame(rows)

    df = pd.concat([main, ref_w], ignore_index=True)
    proteins = sorted(c for c in df.columns
                      if c not in meta + ["role"])
    df = df[["test_id", "role"] + meta[1:] + proteins]
    df[proteins] = df[proteins].fillna(0.0)
    # T6 is a conditional referee inside the CLL branch
    df.loc[df["test_id"] == "T6_CLLvsLYMPH", "role"] = "referee"
    return df


def table_auc():
    d = pd.read_csv(OUT / "pairseed_v2_k8_per_cancer_auc.csv")
    return d[["cancer", "panel_size", "panel", "pairseed_auc", "alvez_auc",
              "delta", "n_samples"]]


def table_stability():
    runs = pd.read_csv(OUT / "pairseed_panel_stability_runs.csv")
    freq = pd.read_csv(OUT / "pairseed_panel_stability_frequency.csv")
    main = (pd.read_csv(OUT / "pairseed_ovr_k8_panels.csv")
            .set_index("cancer").proteins.str.split(","))
    rows = []
    for c, g in runs.groupby("cancer"):
        fc = freq[freq.cancer == c].sort_values("frequency", ascending=False)
        pct = lambda q: int(round(100 * q))  # noqa: E731
        rows.append({
            "cancer": c,
            "n_repeats": len(g),
            "mean_panel_size": round(g.n_proteins.mean(), 1),
            "heldout_balanced_accuracy_mean": round(g.heldout_bal_acc.mean(), 1),
            "heldout_balanced_accuracy_sd": round(g.heldout_bal_acc.std(), 1),
            "n_distinct_proteins_selected": int((fc.frequency > 0).sum()),
            "proteins_selected_in_>=50%_of_repeats": "; ".join(
                f"{p} ({pct(q)}%)" for p, q in zip(fc.protein, fc.frequency)
                if q >= 0.5),
            "main_panel_(Figure_1)_selection_frequency": "; ".join(
                f"{p} ({pct(q)}%)" for p, q in zip(fc.protein, fc.frequency)
                if p in main[c]),
        })
    return pd.DataFrame(rows)


tables = {
    "Table_S1_per_cancer_auc.csv": table_auc(),
    "Table_S2_panel_stability.csv": table_stability(),
    "Table_S3_pairseed_no_cascade_family.csv":
        pd.read_csv(OUT / "pairseed_no_cascade_family.csv"),
    "Table_S4_alvez_mimicked_family.csv":
        pd.read_csv(OUT / "alvez_mimicked_family.csv"),
    "Table_S5_pairseed_cascade_k8_formulas.csv": table_formulas(),
    "Table_S6_pairseed_cascade_family.csv":
        pd.read_csv(Path("PairSeed_cascades") /
                    "cascade_with_referees_family.csv"),
    "Table_S7_lodo_no_cascade_panels.csv":
        pd.read_csv(OUT / "honest_lodo_pairseed_nocascade_panels.csv"),
    "Table_S8_lodo_cascade_panels.csv":
        pd.read_csv(OUT / "honest_lodo_cascade_with_referees_panels.csv"),
}

for name, df in tables.items():
    df.to_csv(SUPP / name, index=False)
    print(f"Saved {SUPP / name}  ({len(df)} rows)")
