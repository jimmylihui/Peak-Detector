# Peak-Detector

The repository is organized around three reproducible pieces:

1. Raw/processed data preparation for baseline arrays.
2. Dataset formatter scripts for prompt-style peak-representation data.
3. Baseline method code for ECG, PPG, BCG, and BSG peak detection benchmarks.

## Repository Layout


| Folder             | Purpose                                                                                                                                            |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data processing/` | Code-only pipeline scripts for downloading/checking datasets, extracting raw data, processing signals, and generating baseline-ready NumPy arrays. |
| `formatter/`       | Dataset formatter scripts copied only from `project-BCG-LLM/combined_data/formatter`, assigned into dataset folders.                               |
| `based baseline/`  | Cleaned baseline method implementations and dataset test entrypoints.                                                                              |
| `LLaMA-Factory/`   | Training framework area from the original project.                                                                                                 |


Generated arrays, checkpoints, plots, prediction files, result tables, notebooks, caches, and copied datasets are intentionally excluded from the cleaned folders.

## Dependencies

Install the repository-level dependencies with:

```bash
pip install -r requirements.txt
```

The dependency file combines the LLaMA-Factory training requirements with the biomedical signal-processing packages used by `data processing`, `formatter`, and `based baseline`.

## Data Flow

1. Download or locate raw datasets:

```bash
python "data processing/download_data.py" --dataset mitbih
python "data processing/download_data.py" --dataset incart
python "data processing/download_data.py" --dataset bidmc
python "data processing/download_data.py" --dataset capnobase
python "data processing/download_data.py" --dataset kansas
python "data processing/download_data.py" --dataset arrhythmia
```

1. Run dataset-specific data-processing scripts to create baseline arrays such as `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, and `y_test.npy`.
2. Use `formatter/` scripts when prompt-style formatted datasets are needed.
3. Run baseline methods from `based baseline/`.

## Peak-Detector LLaMA-Factory Experiment

The MIT-BIH processed ECG peak SFT experiment is configured here:

```text
LLaMA-Factory/examples/train_full/qwen2.5_lora_sft_ds3_3B_full_9000_J_peak_ECG_peaks_mitbih_processed.yaml
```

Run it from the `LLaMA-Factory` folder:

```bash
cd LLaMA-Factory
PYTHONPATH=src WANDB_MODE=offline python -m llamafactory.cli train \
  examples/train_full/qwen2.5_lora_sft_ds3_3B_full_9000_J_peak_ECG_peaks_mitbih_processed.yaml
```

If W&B is configured, replace `WANDB_MODE=offline` with the normal W&B environment. The dataset entry is `mitbih_ecg_peaks_train_processed`, which expects:

```text
LLaMA-Factory/data/mitbih_ecg_peaks_train_processed.json
```

This LLaMA-Factory checkout expects compatible package versions from its `requirements.txt`, including `transformers<=4.52.3`, `peft<=0.15.2`, `trl<=0.9.6`, and `accelerate<=1.7.0`. A broken optional `flash_attn` install can prevent imports; remove or rebuild it if the CLI fails while importing `flash_attn_2_cuda`.

## Included Datasets


| Modality | Dataset folders        |
| -------- | ---------------------- |
| ECG      | MIT-BIH, INCART        |
| PPG      | BIDMC, CapnoBase       |
| BCG      | Kansas, BCG Arrhythmia |
| BSG      | ICU BSG, Hospital BSG  |


## Manual Dataset Links

Some datasets require manual access or do not have stable direct-download URLs. The most important manual paths are also listed in `data processing/README.md`.


| Dataset                              | Source                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CapnoBase                            | [https://doi.org/10.5683/SP2/NLB8IT](https://doi.org/10.5683/SP2/NLB8IT)                                                                                                                                                                                                                                                                     |
| CapnoBase                            | [https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT)                                                                                                                                                                                       |
| Kansas BCG original                  | [https://doi.org/10.21227/77hc-py84](https://doi.org/10.21227/77hc-py84)                                                                                                                                                                                                                                                                     |
| Kansas BCG original                  | [https://ieee-dataport.org/open-access/bed-based-ballistocardiography-dataset](https://ieee-dataport.org/open-access/bed-based-ballistocardiography-dataset)                                                                                                                                                                                 |
| Kansas BCG processed copy            | [https://springernature.figshare.com/articles/dataset/BCG_dataset/20496234](https://springernature.figshare.com/articles/dataset/BCG_dataset/20496234)                                                                                                                                                                                       |
| BCG Arrhythmia / Multi-Pathology BCG | [https://doi.org/10.6084/m9.figshare.28416896](https://doi.org/10.6084/m9.figshare.28416896)                                                                                                                                                                                                                                                 |
| BCG Arrhythmia / Multi-Pathology BCG | [https://springernature.figshare.com/articles/dataset/A_Multi-Pathology_Ballistocardiogram_Dataset_for_Cardiac_Function_Monitoring_and_Arrhythmia_Assessment/28416896](https://springernature.figshare.com/articles/dataset/A_Multi-Pathology_Ballistocardiogram_Dataset_for_Cardiac_Function_Monitoring_and_Arrhythmia_Assessment/28416896) |


## Verification Status

The cleaned folders were checked after organization:

- `data processing/`: all Python scripts compile.
- `formatter/`: 8 formatter scripts compile.
- `based baseline/`: 162 baseline Python scripts compile.

