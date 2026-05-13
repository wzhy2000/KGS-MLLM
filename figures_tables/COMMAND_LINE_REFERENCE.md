# Figures and Tables Command-Line Reference

This document summarizes the command-line entry points used to regenerate the manuscript figures and tables.

Run commands from the `figures_tables/` directory unless otherwise specified:

```bash
cd KGS_MLLM_Project/figures_tables
```

Default paths:

```text
PROJECT_ROOT        = parent directory of figures_tables/
RESULT_ROOT         = PROJECT_ROOT/result
DATASET_ROOT        = PROJECT_ROOT/dataset
IMAGE_DIR           = figures_tables/image
TABLE_DIR           = figures_tables/table
```

If `dataset/` and `result/` are stored outside the repository, pass explicit paths through the options shown below.

## Notebooks

Generate and display all figures:

```bash
jupyter notebook generate_figures.ipynb
```

Generate and display all tables:

```bash
jupyter notebook generate_tables.ipynb
```

The notebooks show the exact command used for each figure or table block.

## Figure Scripts

### `figure_scripts/nshot_best.py`

Draws the best-fold n-shot 7-class accuracy curve.

Example:

```bash
python figure_scripts/nshot_best.py \
  --result-root ../result/nshot \
  --score-root ../dataset/nshotdataset/scoredata \
  --output image/nshot_best_7class.png
```

Options:

```text
--result-root RESULT_ROOT
--score-root SCORE_ROOT
--output OUTPUT
```

### `figure_scripts/ICLtestdraw.py`

Draws the prompt/perturbation accuracy comparison.

Example:

```bash
python figure_scripts/ICLtestdraw.py \
  --result-root ../result/ICLperturbresult \
  --output image/ICLtestdraw.png
```

Options:

```text
--result-root RESULT_ROOT
--output OUTPUT
```

### `figure_scripts/nshot_average_all.py`

Draws mean n-shot accuracy curves for all MLLMs.

Example:

```bash
python figure_scripts/nshot_average_all.py \
  --result-root ../result/nshot \
  --score-root ../dataset/nshotdataset/scoredata \
  --output-dir image
```

Options:

```text
--result-root RESULT_ROOT
--score-root SCORE_ROOT
--output-dir OUTPUT_DIR
```

### `figure_scripts/nshot_boxplot_all.py`

Draws n-shot 7-class overall boxplots.

Example:

```bash
python figure_scripts/nshot_boxplot_all.py \
  --result-set both \
  --result-root ../result \
  --score-root ../dataset/nshotdataset/scoredata \
  --output-dir image
```

Options:

```text
--result-set {nshot,nshot_latest,both}
--result-root RESULT_ROOT
--score-root SCORE_ROOT
--output-dir OUTPUT_DIR
```

### `figure_scripts/overall_classification_performance.py`

Draws the overall classification performance comparison.

Example:

```bash
python figure_scripts/overall_classification_performance.py \
  --model-report table/model_group_average_report.txt \
  --doctor-report table/doctor_group_average_report.txt \
  --output image/overall_classification_performance.png
```

Options:

```text
--model-report MODEL_REPORT
--doctor-report DOCTOR_REPORT
--output OUTPUT
--show-title
```

### `figure_scripts/classification_performance_radar.py`

Draws radar plots across the five KGS endoscopic findings.

Example:

```bash
python figure_scripts/classification_performance_radar.py \
  --model-report table/model_group_average_report.txt \
  --doctor-report table/doctor_group_average_report.txt \
  --output image/classification_performance_radar.png
```

Options:

```text
--model-report MODEL_REPORT
--doctor-report DOCTOR_REPORT
--output OUTPUT
--show-title
```

### `figure_scripts/subgroup_accuracy_vs_dlkgs.py`

Draws subgroup accuracy comparisons between one MLLM and DLKGs.

KGS test set example:

```bash
python figure_scripts/subgroup_accuracy_vs_dlkgs.py \
  --dataset test \
  --model all \
  --result-root ../result/testdataset_result_batch \
  --p-value-path table/mcnemar_pvalues_vs_DLKGS.txt \
  --output-dir image
```

External validation example:

```bash
python figure_scripts/subgroup_accuracy_vs_dlkgs.py \
  --dataset external \
  --model gemini \
  --result-root ../result/externaldataset_result_batch \
  --output-dir image
```

Options:

```text
--dataset {test,external}
--model {all,gemini,gpt,claude,grok}
--result-root RESULT_ROOT
--p-value-path P_VALUE_PATH
--output-dir OUTPUT_DIR
--output OUTPUT
```

### `figure_scripts/likert_score_violinplot.py`

Plots Likert score distributions for correct and incorrect cases.

Example:

```bash
python figure_scripts/likert_score_violinplot.py \
  --score0 table/score0.xlsx \
  --score1 table/score1.xlsx \
  --output image/violinplot.png
```

Options:

```text
--score0 SCORE0
--score1 SCORE1
--output OUTPUT
--random-seed RANDOM_SEED
```

### `figure_scripts/likert_score_cumulative_frequency.py`

Plots cumulative frequency curves for Likert scores.

Example:

```bash
python figure_scripts/likert_score_cumulative_frequency.py \
  --score0 table/score0.xlsx \
  --score1 table/score1.xlsx \
  --output image/cumulative_frequency_plot.png
```

Options:

```text
--score0 SCORE0
--score1 SCORE1
--output OUTPUT
```

### `figure_scripts/misclassification_donut_error.py`

Plots donut charts for model misclassification error types.

Example:

```bash
python figure_scripts/misclassification_donut_error.py \
  --stats-file ../result/testdataset_doctorscore/misclassification_statistics.xlsx \
  --output-dir image
```

Options:

```text
--stats-file STATS_FILE
--output-dir OUTPUT_DIR
--flat-output
```

### `figure_scripts/nshot_token_time_curves.py`

Plots response-time and token-consumption curves for n-shot experiments.

Example:

```bash
python figure_scripts/nshot_token_time_curves.py \
  --result-root ../result/nshot_latest \
  --output-dir image
```

Options:

```text
--result-root RESULT_ROOT
--output-dir OUTPUT_DIR
```

### `figure_scripts/nshot_accuracy_overall.py`

Generates n-shot accuracy line plots for 7-class Kimura-Takemoto and KGS-A 3-class tasks.

Initial n-shot example:

```bash
python figure_scripts/nshot_accuracy_overall.py \
  --result-set nshot \
  --result-root ../result \
  --model all \
  --class-mode both \
  --score-root ../dataset/nshotdataset/scoredata \
  --output-root image
```

Two-week interval example:

```bash
python figure_scripts/nshot_accuracy_overall.py \
  --result-set nshot_latest \
  --result-root ../result \
  --model all \
  --class-mode both \
  --score-root ../dataset/nshotdataset/scoredata \
  --output-root image
```

Options:

```text
--result-set {nshot,nshot_latest}
--result-root RESULT_ROOT
--model {all,gemini,gpt,claude,grok}
--class-mode {both,7,3}
--score-root SCORE_ROOT
--output-root OUTPUT_ROOT
--group-by-result-set
```

### `figure_scripts/batch_confusion_matrices.py`

Generates confusion matrices for the KGS test set and external validation set.

KGS test set example:

```bash
python figure_scripts/batch_confusion_matrices.py \
  --dataset test \
  --result-root ../result/testdataset_result_batch \
  --dataset-root ../dataset/testdataset \
  --output-root image \
  --model all \
  --category all
```

External validation example:

```bash
python figure_scripts/batch_confusion_matrices.py \
  --dataset external \
  --result-root ../result/externaldataset_result_batch \
  --dataset-root ../dataset/externaldataset \
  --output-root image \
  --model all \
  --category all
```

Options:

```text
--dataset {test,external}
--result-root RESULT_ROOT
--dataset-root DATASET_ROOT
--output-root OUTPUT_ROOT
--model {all,gemini,gpt,claude,grok,dlkgs}
--category {all,A,DR,H,IM,N}
--matrix-folder MATRIX_FOLDER
```

### `figure_scripts/dlkgs_roc_pr_curves.py`

Draws DLKGs micro-average ROC and precision-recall curves.

Example:

```bash
python figure_scripts/dlkgs_roc_pr_curves.py \
  --json-path ../result/testdataset_result_batch/DLKGS/test_results.json \
  --output-root image/DLKGS \
  --curve all
```

Options:

```text
--json-path JSON_PATH
--output-root OUTPUT_ROOT
--curve {all,roc,pr}
```

## Table Scripts

### `table_scripts/build_overall_performance_tables.py`

Builds the model and doctor group average reports used by Table 1 and multiple figures.

Example:

```bash
python table_scripts/build_overall_performance_tables.py \
  --result-root ../result \
  --dataset-root ../dataset/testdataset \
  --output-dir table
```

Options:

```text
--result-root RESULT_ROOT
--dataset-root DATASET_ROOT
--output-dir OUTPUT_DIR
```

Outputs:

```text
table/model_group_average_report.txt
table/doctor_group_average_report.txt
```

### `table_scripts/build_score_tables.py`

Builds `score0.xlsx` and `score1.xlsx` from doctor-score workbooks.

Example:

```bash
python table_scripts/build_score_tables.py \
  --source-root ../result/testdataset_doctorscore \
  --output-dir table
```

Options:

```text
--source-root SOURCE_ROOT
--output-dir OUTPUT_DIR
```

### `table_scripts/build_mcnemar_pvalues_vs_dlkgs.py`

Builds McNemar p-value tables for subgroup accuracy plots.

KGS test set example:

```bash
python table_scripts/build_mcnemar_pvalues_vs_dlkgs.py \
  --dataset test \
  --result-root ../result/testdataset_result_batch \
  --output table/mcnemar_pvalues_vs_DLKGS.txt
```

External validation example:

```bash
python table_scripts/build_mcnemar_pvalues_vs_dlkgs.py \
  --dataset external \
  --result-root ../result/externaldataset_result_batch \
  --output table/external_mcnemar_pvalues_Gemini_vs_DLKGs.txt
```

Options:

```text
--dataset {test,external}
--result-root RESULT_ROOT
--output OUTPUT
```

### `table_scripts/build_pairwise_accuracy_comparison_table.py`

Builds pairwise image-level accuracy comparison tables.

KGS test set example:

```bash
python table_scripts/build_pairwise_accuracy_comparison_table.py \
  --dataset test \
  --result-root ../result/testdataset_result_batch \
  --output-dir table
```

External validation example:

```bash
python table_scripts/build_pairwise_accuracy_comparison_table.py \
  --dataset external \
  --result-root ../result/externaldataset_result_batch \
  --output-dir table
```

Options:

```text
--dataset {test,external}
--result-root RESULT_ROOT
--output-dir OUTPUT_DIR
--output-prefix OUTPUT_PREFIX
--adjustment {fdr_bh,bonferroni}
```

### `table_scripts/build_pairwise_other_metrics_comparison_table.py`

Builds pairwise comparison tables for specificity, precision, recall, and F1 score.

KGS test set example:

```bash
python table_scripts/build_pairwise_other_metrics_comparison_table.py \
  --dataset test \
  --result-root ../result/testdataset_result_batch \
  --dataset-root ../dataset/testdataset \
  --output-dir table
```

External validation example:

```bash
python table_scripts/build_pairwise_other_metrics_comparison_table.py \
  --dataset external \
  --result-root ../result/externaldataset_result_batch \
  --dataset-root ../dataset/externaldataset \
  --output-dir table
```

Options:

```text
--dataset {test,external}
--result-root RESULT_ROOT
--dataset-root DATASET_ROOT
--output-dir OUTPUT_DIR
--output-prefix OUTPUT_PREFIX
--n-bootstraps N_BOOTSTRAPS
```

### `table_scripts/build_subcategory_performance_table.py`

Builds performance tables for 18 endoscopic subcategories.

KGS test set example:

```bash
python table_scripts/build_subcategory_performance_table.py \
  --dataset test \
  --result-root ../result \
  --dataset-root ../dataset/testdataset \
  --output-dir table
```

External validation example:

```bash
python table_scripts/build_subcategory_performance_table.py \
  --dataset external \
  --result-root ../result \
  --dataset-root ../dataset/externaldataset \
  --output-dir table
```

Options:

```text
--dataset {test,external}
--result-root RESULT_ROOT
--dataset-root DATASET_ROOT
--output-dir OUTPUT_DIR
--output-prefix OUTPUT_PREFIX
```

## Rebuilding All Tables Needed by Figures

Some figure scripts read intermediate tables. A typical order is:

```bash
python table_scripts/build_overall_performance_tables.py --result-root ../result --dataset-root ../dataset/testdataset --output-dir table
python table_scripts/build_score_tables.py --source-root ../result/testdataset_doctorscore --output-dir table
python table_scripts/build_mcnemar_pvalues_vs_dlkgs.py --dataset test --result-root ../result/testdataset_result_batch --output table/mcnemar_pvalues_vs_DLKGS.txt
python table_scripts/build_mcnemar_pvalues_vs_dlkgs.py --dataset external --result-root ../result/externaldataset_result_batch --output table/external_mcnemar_pvalues_Gemini_vs_DLKGs.txt
```

After these intermediates are available, the figure scripts can be run independently.
