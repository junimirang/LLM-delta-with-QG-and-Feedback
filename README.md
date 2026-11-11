# LLM Synthetic Traffic Generation using QG and Feedback system

Research code for generating synthetic network-session feature records from statistical metadata, screening generated records with a Mahalanobis-distance quality gate (QG), and evaluating their use as training data for remote-access traffic classification.

The LLM receives a statistical profile rather than individual source records. The profile includes univariate statistics, correlations, and principal components. The quality gate screens generated samples against a reference subset of the training data. In the feedback variant, summaries of rejected samples guide subsequent generation batches.

The experiments use DARPA 99 session features and distinguish SSH/TELNET remote-access sessions from other sessions. The scripts produce tabular feature records for offline evaluation.

## Repository contents

| Stage | Main scripts |
|---|---|
| Statistical profiling | `1. meta feature extractor.py` |
| Baseline synthesis | `2. Traffic Generator for SMOTE.py`, `2. Traffic Generator for wgan-gp.py` |
| LLM synthesis and ablations | `2_1. Traffic_Generator for LLM without delta.py`, `2_2. Traffic Generator for LLM.py`, `2_3. Traffic Generator for LLM with QG.py`, `2_4. Traffic Generator for LLM with QG and Feedback.py` |
| Classification | `3. Detection_DARPA99_RF_batch_revision_new.py`, `3. Detection_DARPA99_MLP_batch_revision_new.py` |
| Experiment orchestration | `4_Revision_Sequential_Orchestrator.py`, `4_Run_LLM_Generators_Sequential.py` |
| Sample quality and proximity | `5_Evaluate_DCR.py`, `5_Evaluate_JSD,SH-E.py` |
| Reporting and figures | `6_Create_Table4_Mean_SD.py`, `7_Graph_Detection_Combined.py` |

## Setup

Install the Python packages used across the scripts:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib requests imbalanced-learn openpyxl torch
```

The code archive contains scripts only. Before running the full pipeline, provide the following inputs:

- **Session feature CSVs:** Supply the preprocessed DARPA 99 training and test files with the feature columns and labels expected by the scripts.
- **Project helpers:** Some scripts import local modules `common_config.py` and `input_helper.py`. These are absent from the archive. Add the project copies and configure the input, output, and model paths before running the scripts that import them.
- **WGAN-GP checkpoints:** Provide the pretrained model files expected by the WGAN-GP generator.
- **Local LLM:** Run a compatible local inference server and configure its endpoint and model in the LLM scripts. The paper uses Qwen3.5-9B (GGUF, Q4_K_M) with LM Studio.

## Reproducing the experiments

1. Extract the statistical profile from the training data.
2. Generate the synthetic pools with SMOTE, WGAN-GP, and the selected LLM conditions. Keep the quality-gated variants in separate output files.
3. Train the Random Forest and MLP classifiers with and without synthetic augmentation. Use the held-out test data for evaluation.
4. Evaluate distributional similarity and diversity with Jensen–Shannon divergence and Shannon entropy, visualize the groups with t-SNE, and assess sample proximity with distance to closest record (DCR).
5. Export the summary table and figures. Check the file paths and options in each script before execution.
