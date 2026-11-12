# Data source and attribution

Dataset: SKEMPI 2.0, original CSV with 7,085 rows.

Authors: Jankauskaite J, Jimenez-Garcia B, Dapkunas J, Fernandez-Recio J, Moal IH.
Publication: SKEMPI 2.0: an updated benchmark of changes in protein-protein binding energy, kinetics and thermodynamics upon mutation. Bioinformatics 35, 462-469 (2019).
DOI: https://doi.org/10.1093/bioinformatics/bty635

Official download: https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv
Official documentation: https://life.bsc.es/pid/skempi2/info/faq_and_help
Terms: https://life.bsc.es/pid/skempi2/info/terms
License stated by the source: Creative Commons Attribution 4.0 International.
License text: https://creativecommons.org/licenses/by/4.0/
Downloaded: 2026-11-12.

The original CSV is retained unchanged. Cleaning, deduplication, labels, metadata-connected grouping, features, splits and predictions are project-derived outputs. This project is not endorsed by the data authors.

Target follows the official convention: delta-delta-G = R*T*ln(Kd_mut/Kd_wt), kcal/mol. Negative means tighter measured binding after mutation, positive means weaker binding. This is not a prediction of absolute Kd.

Primary task: protein-protein interaction mutation-effect regression. Antibody-related rows are a diagnostic subset, not the entire training dataset. No sequences or measurements are newly generated experiments in this project.
