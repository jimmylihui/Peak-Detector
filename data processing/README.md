# Data Processing

Code-only scripts that generate the NumPy arrays consumed by the baseline methods in `based baseline`.

Included scripts write baseline-facing files such as `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, and `y_test.npy` for the ECG, PPG, BCG, and BSG benchmark datasets.

Dataset split metadata is kept beside the matching processing scripts when available, for example `ECG/MITBIH/split_info.json`, `PPG/BIDMC/split_info.json`, and `PPG/CapnoBase/split_metadata.json`.

Use `download_data.py` to download public WFDB datasets (`mitbih`, `incart`, `bidmc`) or to check/copy local-only datasets (`capnobase`, `kansas`, `arrhythmia`, `icu`).

## Manual Download Paths

If `download_data.py` cannot download or access a dataset, use these source pages directly:

| Dataset | Download paths |
| --- | --- |
| CapnoBase | https://doi.org/10.5683/SP2/NLB8IT |
| CapnoBase | https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT |
| Kansas BCG original | https://doi.org/10.21227/77hc-py84 |
| Kansas BCG original | https://ieee-dataport.org/open-access/bed-based-ballistocardiography-dataset |
| Kansas BCG processed copy | https://springernature.figshare.com/articles/dataset/BCG_dataset/20496234 |
| BCG Arrhythmia / Multi-Pathology BCG | https://doi.org/10.6084/m9.figshare.28416896 |
| BCG Arrhythmia / Multi-Pathology BCG | https://springernature.figshare.com/articles/dataset/A_Multi-Pathology_Ballistocardiogram_Dataset_for_Cardiac_Function_Monitoring_and_Arrhythmia_Assessment/28416896 |

CapnoBase benchmark processing only needs the 42 files ending in `*_8min.mat`. For Kansas, IEEE DataPort is the original source and the Figshare page is a smaller processed copy. The BCG Arrhythmia Figshare dataset is about 2.36 GB.

Excluded: LLM prompt formatters, explanation-data builders, notebooks, training loops, evaluation code, plots, caches, checkpoints, generated arrays, prediction files, and result artifacts.
