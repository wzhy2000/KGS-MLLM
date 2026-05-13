# KGS-MLLM Project

This repository contains code for evaluating multimodal large language models (MLLMs), DLKGs, and endoscopists on Kyoto Gastritis Score (KGS)-related endoscopic findings.

The project is organized into two parts:

- `experiments/`: experiment scripts used to generate model outputs.
- `figures_tables/`: scripts and notebooks used to regenerate the manuscript figures and tables from saved results.

The five KGS endoscopic findings are:

- `A`: Atrophy
- `DR`: Diffuse redness
- `H`: Hypertrophy gastric fold
- `IM`: Intestinal metaplasia
- `N`: Nodular gastritis

## Repository Structure

```text
KGS_MLLM_Project/
|-- config.py
|-- core/
|   |-- llm_client.py
|   `-- task_config.py
|-- utils/
|   |-- data_utils.py
|   `-- image_utils.py
|-- experiments/
|   |-- exp_nshot.py
|   |-- exp_perturb.py
|   |-- exp_re_pipeline.py
|   |-- exp_batch_features.py
|   `-- exp_DLKGS.py
|-- DLKGS/
|   `-- five_model/
|-- figures_tables/
|   |-- generate_figures.ipynb
|   |-- generate_tables.ipynb
|   |-- COMMAND_LINE_REFERENCE.md
|   |-- figure_scripts/
|   |-- table_scripts/
|   |-- image/
|   `-- table/
|-- requirements.txt
`-- environment.yml
```

`dataset/` and `result/` are expected data/result directories. They may be stored outside this repository. If they are not placed under `KGS_MLLM_Project/`, pass their paths through the command-line options or edit the path variables in the notebooks.

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate kgs_mllm
```

Install additional Python dependencies if needed:

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` before running experiment scripts:

```python
DEFAULT_MODEL = "gemini-3-pro-preview"
API_KEY = os.environ.get("API_KEY", "your_default_api_key_here")
PROXY_API_URL = os.environ.get("PROXY_API_URL", "your_default_api_url_here")
BASE_WORKSPACE = r"your_actual_workspace_root_directory"
```

`BASE_WORKSPACE` should point to a directory containing the required datasets and output folders.

For figure and table regeneration, most scripts can also receive explicit input/output paths through command-line options, so `dataset/` and `result/` do not need to be inside the repository.

## Experiment Scripts

The experiment scripts are located in `experiments/`.

### N-shot Experiment

Script:

```bash
python experiments/exp_nshot.py
```

Purpose:

- Runs the n-shot in-context learning experiment for the atrophy task.
- Uses increasing shot settings from 2-shot to 22-shot.
- Saves Excel outputs under `BASE_WORKSPACE/result_nshot/<DEFAULT_MODEL>/`.

Notes:

- This script currently uses `BASE_WORKSPACE` and `DEFAULT_MODEL` from `config.py`.
- It does not define a command-line parser.
- To change `batch_size`, edit the call at the bottom of the script:

```python
run_nshot_pipeline(batch_size=8)
```

### Perturbation Experiment

Script:

```bash
python experiments/exp_perturb.py
```

Purpose:

- Runs perturbation experiments at the 14-shot setting.
- Conditions include baseline, short prompt, reversed category order, reversed example order, rotated images, and random image augmentation.
- Saves Excel outputs under `BASE_WORKSPACE/ICLA25/result/`.

Notes:

- This script currently uses `BASE_WORKSPACE` and `DEFAULT_MODEL` from `config.py`.
- It does not define a command-line parser.
- To change the tested conditions or `batch_size`, edit the `if __name__ == "__main__"` block.

### Repeated/Incremental ICL Pipeline

Script:

```bash
python experiments/exp_re_pipeline.py
```

Purpose:

- Runs repeated cross-validation-style ICL experiments.
- Saves outputs under `BASE_WORKSPACE/result_re/<DEFAULT_MODEL>/`.
- Includes `response_time` and `tokens` columns where available.

Notes:

- This script currently uses `BASE_WORKSPACE` and `DEFAULT_MODEL` from `config.py`.
- It does not define a command-line parser.
- To change `batch_size` or `num_groups`, edit the call:

```python
run_cv_pipeline()
```

### Batch Feature Experiment

Script:

```bash
python experiments/exp_batch_features.py --dataset testdataset --tasks A DR H IM N --batch-size 8
```

Purpose:

- Runs MLLM batch prediction for the five KGS findings.
- Supports both the KGS test set and the external validation set.
- Produces result workbooks in the same output structure used by the table and figure scripts.

Options:

```text
--dataset {batch,external,externaldataset,test,testdataset}
--tasks {A,DR,H,IM,N} [...]
--batch-size BATCH_SIZE
```

External validation example:

```bash
python experiments/exp_batch_features.py --dataset externaldataset --tasks A DR H IM N --batch-size 8
```

### DLKGs Prediction Experiment

Script:

```bash
python experiments/exp_DLKGS.py --dataset testdataset
```

Purpose:

- Runs DLKGs five-model prediction.
- Saves an intermediate JSON file and an Excel prediction workbook.

Options:

```text
--dataset {batch,external,externaldataset,test,testdataset}
--base-dir BASE_DIR
--model-file MODEL_FILE
--weights-dir WEIGHTS_DIR
--json-output JSON_OUTPUT
--excel-output EXCEL_OUTPUT
--device DEVICE
```

External validation example:

```bash
python experiments/exp_DLKGS.py --dataset externaldataset
```

## Figure and Table Reproduction

The figure and table reproduction code is in `figures_tables/`.

Two executed notebooks are provided:

```text
figures_tables/generate_figures.ipynb
figures_tables/generate_tables.ipynb
```

Each notebook:

- lists the command used for every figure/table block;
- runs the corresponding script;
- displays the generated figure or table;
- prints elapsed time for each code cell.

If `dataset/` and `result/` are stored outside the project root, edit the path variables in the first code cell of each notebook.

For a full command-line reference, see:

```text
figures_tables/COMMAND_LINE_REFERENCE.md
```

## Figure/Table Directory Layout

```text
figures_tables/
|-- generate_figures.ipynb
|-- generate_tables.ipynb
|-- COMMAND_LINE_REFERENCE.md
|-- figure_scripts/
|   `-- *.py
|-- table_scripts/
|   `-- *.py
|-- image/
|   `-- generated figure files
`-- table/
    `-- generated table files
```

## Data and Result Paths

The table and figure scripts assume the following default project-level paths:

```text
KGS_MLLM_Project/dataset/
KGS_MLLM_Project/result/
KGS_MLLM_Project/figures_tables/image/
KGS_MLLM_Project/figures_tables/table/
```

Most scripts also expose options such as:

```text
--result-root
--dataset-root
--output-dir
--output-root
--output
```

Use these options when the downloaded data and result files are stored elsewhere.

## License

This project is released for research use. See the repository license file if provided.

## Citation

Please cite the associated publication when using this code or derived results.
