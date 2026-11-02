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

The histogram uses ten equal-width bins. Standard deviation is the population form. Empty strings are treated as missing; other special missing-value spellings (for example, `NA` or `null`) are treated as category/text values unless exported as blank cells. The CSV parser supports quoted fields and escaped double quotes, but expects comma-separated UTF-8 text.

## Validation status

The implementation is committed on the milestone branch. Run `npm run build` from `frontend/` and the existing frontend CI workflow before merging. This environment has not executed the build, so the milestone must not be described as CI-verified until GitHub Actions completes.
