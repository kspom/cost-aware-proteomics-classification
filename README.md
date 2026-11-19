# From Discovery to Deployment — code

Code for "From Discovery to Deployment: Cost-Aware Multi-Disease
Classification in Plasma Proteomics" (Zlobina). Run all scripts from the
repository root; outputs are written to `paper_figures/`,
`PairSeed_cascades/`, `PairSeed_no_cascade/` and `Supplementary materials/`.

## Input data

Download into the repository root:

- `pancancer_olink_data_biostudies_v2.txt` — BioStudies S-BSST935:
  https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/935/S-BSST935/Files/pancancer_olink_data_biostudies_v2.txt
- `41467_2023_39765_MOESM7_ESM.xlsx` — Supplementary Data 5 of Alvez et al.,
  Nat Commun 2023 (https://doi.org/10.1038/s41467-023-39765-y):
  https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-39765-y/MediaObjects/41467_2023_39765_MOESM7_ESM.xlsx

`PairSeed_cascades/architecture*.json` define the cascade structure and
are inputs, not results.

## Requirements

Python ≥ 3.10; `pip install -r requirements.txt`.

## Run order

1. Preprocessing: `explore_missingness.py`, `sex_voted_labels.py`.
2. PairSeed without cascade: `analysis/pairseed_ovr_k8.py`,
   `pairseed_ovr_k8_boxplots.py`, `pairseed_no_cascade_family.py`,
   `pairseed_no_cascade_family_extend_to_k24.py`,
   `pairseed_per_cancer_auc.py`, `backward_elimination_ovr_k8.py`,
   `pairseed_panel_stability.py`, `pairseed_no_cascade_composition.py`,
   `pairseed_search_trace.py MYEL`, `pairseed_search_trace.py LUNGC`.
3. Alvez classifier and Alvez-like family:
   `analysis/alvez_83_protein_classifier.py`, `alvez_mimicked_family.py`.
4. Cascade: `analysis/pairseed_cascade_k8.py`,
   `pairseed_bucket_referees.py`, `cascade_with_referees_full_cohort.py`,
   `pairseed_cascade_family.py`, `pairseed_cascade_with_referees_family.py`,
   `backward_elimination_cascade_k8.py`, `pairseed_cascade_composition.py`.
5. Leave-one-disease-out: `analysis/honest_lodo_pairseed_nocascade.py`,
   `honest_lodo_cascade_with_referees.py`,
   `reclassify_lodo_cascade_with_referees.py`,
   `open_set_typicality_check.py`.
6. Optional: `analysis/eval_complete_patients.py` re-evaluates saved panels
   without repeating the searches.
7. `analysis/build_supplementary_tables.py` and all scripts in `figures/`.

The stability and leave-one-disease-out searches take 2–3 h each.

## License

MIT (see `LICENSE`). The input data remain under the terms of their
original sources.
