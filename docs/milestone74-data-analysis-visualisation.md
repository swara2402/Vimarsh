# Milestone 74: Data analysis and visualisation workspace

## Delivered

The React dashboard now includes a browser-local exploratory data analysis workspace. An operator can upload a CSV (up to 15 MB) with a header row and inspect:

- Dataset dimensions, numeric feature count, missing-cell count, and complete-row count.
- Ten-bin histograms for detected numeric features.
- Category frequency / class-balance view for a selectable column.
- Per-column missing-value counts and percentages.
- Pearson correlation matrix for up to six numeric columns, using pairwise complete observations.
- Descriptive statistics: valid count, missing count, mean, median, population standard deviation, minimum, and maximum.

The workspace is responsive and does not require a new charting dependency. CSV parsing and analysis run in the browser. The uploaded dataset is not sent to the FastAPI service, stored in the VIMARSH database, or used to retrain a model. Clearing the dataset removes it from the component state.

## Demonstration

1. Start the application using the existing [demo-day runbook](demo-day-runbook.md).
2. Scroll to **Data Analysis & Visualisation**.
3. Upload a CSV benchmark export (Wine Quality, MAGIC Gamma Telescope, or HTRU2).
4. Change the histogram feature and category column to show the interactive views.
5. Point out missingness, class balance, correlations, and the numerical summary.

Use a header row with unique column names. Numeric columns are detected when at least 85% of their non-empty values parse as numbers. The category view shows the twelve most frequent values. The correlation matrix shows at most six detected numeric columns.

## Interpretation and limitations

This is descriptive exploratory analysis, not a substitute for the repository's controlled experiment pipeline. It does not compute model accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, calibration curves, or drift metrics because those require aligned model predictions, labels, or a defined reference population. It does not infer causal relationships. Correlation is descriptive and may be affected by missingness, outliers, and confounding.

The histogram uses ten equal-width bins. Standard deviation is the population form. Empty strings are treated as missing; other special missing-value spellings (for example, `NA` or `null`) are treated as category/text values unless exported as blank cells. The CSV parser supports quoted fields and escaped double quotes, strips a UTF-8 byte-order mark, rejects unclosed quoted fields and rows with more fields than the header, and expects comma-separated UTF-8 text. Short rows are padded with blank cells. Correlations that are undefined because a feature is constant or has too few paired observations are displayed as N/A rather than as zero.

## Validation status

The implementation is committed on the milestone branch. Run `npm run build` from `frontend/` and the existing frontend CI workflow before merging. This environment has not executed the build, so the milestone must not be described as CI-verified until GitHub Actions completes.

## Saved VIMARSH policy evaluation visualisation

The dashboard also presents the committed verification evaluation snapshots dated 31 October 2026. It reads the aggregate values computed from these exact source files:

- `results/verification/conservative_evaluation_table_20261031_190449.json`
- `results/verification/balanced_evaluation_table_20261031_190449.json`
- `results/verification/coverage_oriented_evaluation_table_20261031_190450.json`

Each snapshot contains 975 rows. The current dashboard selector compares coverage, error among covered cases, abstention count, action distribution, and a confusion matrix for covered predictions. Confusion-matrix-derived precision, recall, and F1 are calculated on covered rows only; they must not be described as full-population classifier metrics.

| Policy snapshot | Coverage | Error among covered | Abstentions | Recorded introduced errors |
|---|---:|---:|---:|---:|
| Conservative | 6.87% | 16.42% | 908 | 2 |
| Balanced | 29.44% | 23.00% | 688 | 0 |
| Coverage-oriented | 100.00% | 22.46% | 0 | 12 |

These are historical saved evaluation outcomes, not a fresh model run. The coverage-oriented snapshot includes alternate-model and additional-evidence actions among covered cases; coverage is not synonymous with accepting the original prediction. The comparison is descriptive and does not establish causal harm reduction or real-world effectiveness.
