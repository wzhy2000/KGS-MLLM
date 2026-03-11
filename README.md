# Multimodal Large Language Models for Kyoto Gastritis Score (KGS) via In-Context Learning

This repository contains the official code implementation for the study evaluating the feasibility and effectiveness of Multimodal Large Language Models (MLLMs) in automating the Kyoto Gastritis Score (KGS) through In-Context Learning (ICL).

The study systematically evaluates state-of-the-art MLLMs (including Gemini 3, GPT-5.2, Claude 4.5, and Grok 4) on grading five key endoscopic features: Atrophy (A), Diffuse Redness (DR), Enlarged Folds (H), Intestinal Metaplasia (IM), and Nodularity (N).

## 📁 Repository Structure

The basic directory structure of the project is as follows:

```text
KGS_MLLM_Project/
├── config.py                  # Global configuration (API keys, model params, workspace paths)
├── core/
│   ├── llm_client.py          # Core LLM API wrapper (dialogue assembly)
│   └── task_config.py         # Centralized configuration for KGS features (A, DR, H, IM, N)
├── utils/
│   ├── data_utils.py          # Data parsing and JSON/Regex extraction tools
│   └── image_utils.py         # Image encoding and dynamic perturbation/augmentation tools
├── experiments/
│   ├── exp_fig2a_nshot.py     # Evaluating n-shot scaling effects (2 to 22-shot)
│   ├── exp_fig2b_perturb.py   # Prompt engineering and perturbation robustness
│   ├── exp_cv_pipeline.py     # Incremental ICL cycle experiment
│   └── exp_batch_features.py  # Batch generalization testing across five key endoscopic features
└── requirements.txt
```

## ⚙️ Installation & Setup
### Install dependencies:

Create a conda environment and install the requirements
```shell
conda env create -f environment.yml
```
Activate the msamil environment
```shell
conda activate kgs_mllm
```
then run:
```shell
pip install -r requirements.txt
```
### Configure the Environment:

Open config.py and update the following variables to match your local environment:

API_KEY: Your LLM API key.

PROXY_API_URL: Your API endpoint (default is provided in the code).

DEFAULT_MODEL: Choose the model you want to evaluate (e.g., 'gemini-3-pro-preview', 'gpt-5.2-chat-latest').

BASE_WORKSPACE: The absolute path to your root dataset folder.

## 🚀 Reproducing Experiments
All results are automatically saved as Excel .xlsx files with breakpoint continuation support.

n-shot Scaling Analysis

Evaluates the marginal utility of increasing the number of examples in the prompt (from 2-shot to 22-shot).

```shell
python -m experiments.exp_nshot.py
```
Perturbation & Robustness Analysis

Assesses model stability under 6 prompt engineering and input perturbation conditions (Baseline, Short Prompt, Reversed Categories, Reversed Examples, Rotated Images, and Random Augmentation) at the peak 14-shot configuration.

```shell
python -m experiments.exp_perturb.py
```
Incremental ICL cycle experiment

Performs a ICL cycle experiment with an incremental feedback learning loop to evaluate self-reinforcement capabilities.

```shell
python -m experiments.exp_re_pipeline.py
```
Cross-Feature Generalization Assessment

Evaluates the optimal ICL strategy across the complete set of five KGS endoscopic features (A, DR, H, IM, N). You can switch the target feature by modifying the task_name parameter in the __main__ block of the script.

```shell
python -m experiments.exp_batch_features.py
```
## 📊 Data Preparation Requirement
To run the pipelines, ensure your local workspace matches the directory structure expected in config.py. The typical structure expects a base directory containing subfolders like imgdata/ (for endoscopic images) and scoredata/ (for JSON files containing expert ground truth scores).

You can download our data from this website:   //......

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📝 Citation
If you find this code or our research helpful in your work, please cite our paper:   //......