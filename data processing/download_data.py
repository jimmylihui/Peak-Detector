#!/usr/bin/env python3
"""Download raw datasets used by the baseline data-processing scripts.

Public WFDB datasets are downloaded directly. Datasets without a stable public
direct-download URL are reported with the local path expected by this project.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path("/path/to/workspace/project-BCG-LLM")

WFDB_DATASETS = {
    "mitbih": {
        "wfdb_name": "mitdb",
        "out_dir": PROJECT_ROOT / "ECG_peak/dataset/mitbih_database",
        "description": "MIT-BIH Arrhythmia Database",
    },
    "incart": {
        "wfdb_name": "incartdb",
        "out_dir": PROJECT_ROOT / "ECG_peak/dataset/incart/files",
        "description": "St Petersburg INCART 12-lead Arrhythmia Database",
    },
    "bidmc": {
        "wfdb_name": "bidmc",
        "out_dir": PROJECT_ROOT / "PPG_peaks/dataset/BIDMC/bidmc-ppg-and-respiration-dataset-1.0.0",
        "description": "BIDMC PPG and Respiration Dataset",
    },
}

LOCAL_DATASETS = {
    "capnobase": {
        "out_dir": PROJECT_ROOT / "PPG_peaks/dataset/capnobase/data",
        "description": "CapnoBase IEEE TBME Respiratory Rate Benchmark",
        "paths": [
            "https://doi.org/10.5683/SP2/NLB8IT",
            "https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT",
        ],
        "note": "Download the 42 files ending in *_8min.mat for the benchmark.",
    },
    "kansas": {
        "out_dir": PROJECT_ROOT / "combined_data/combined_splitted_data/BCG",
        "description": "Kansas bed-based ballistocardiography dataset",
        "paths": [
            "https://doi.org/10.21227/77hc-py84",
            "https://ieee-dataport.org/open-access/bed-based-ballistocardiography-dataset",
            "https://springernature.figshare.com/articles/dataset/BCG_dataset/20496234",
        ],
        "note": "The Figshare path is a smaller processed copy; IEEE DataPort is the original source.",
    },
    "arrhythmia": {
        "out_dir": PROJECT_ROOT / "combined_data/combined_splitted_data/BCG_Arrhythmia",
        "description": "Multi-Pathology BCG dataset for arrhythmia assessment",
        "paths": [
            "https://doi.org/10.6084/m9.figshare.28416896",
            "https://springernature.figshare.com/articles/dataset/A_Multi-Pathology_Ballistocardiogram_Dataset_for_Cardiac_Function_Monitoring_and_Arrhythmia_Assessment/28416896",
        ],
        "note": "Figshare dataset download is about 2.36 GB.",
    },
    "icu": {
        "out_dir": PROJECT_ROOT / "ICU_3d_hr/benchmark/dataset/split_icu_dataset",
        "description": "ICU BSG split dataset",
        "paths": [],
        "note": "No public download path was added for this local split dataset.",
    },
}


def download_wfdb_dataset(name: str) -> None:
    try:
        import wfdb
    except ImportError as exc:
        raise SystemExit("wfdb is required: pip install wfdb") from exc

    meta = WFDB_DATASETS[name]
    out_dir = meta["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {meta['description']} to {out_dir}")
    wfdb.dl_database(meta["wfdb_name"], dl_dir=str(out_dir))


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(out_dir)
        return
    if archive_path.suffixes[-2:] in [[".tar", ".gz"], [".tar", ".xz"], [".tar", ".bz2"]] or archive_path.suffix == ".tar":
        with tarfile.open(archive_path) as archive:
            archive.extractall(out_dir)
        return
    raise ValueError(f"Unsupported archive type: {archive_path}")


def download_archive(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / Path(url.split("?")[0]).name
    print(f"Downloading {url}")
    urlretrieve(url, archive_path)
    print(f"Extracting {archive_path} to {out_dir}")
    extract_archive(archive_path, out_dir)


def copy_local_source(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Local source does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already exists, skipping: {destination}")
        return
    print(f"Copying {source} -> {destination}")
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def report_local_dataset(name: str) -> None:
    meta = LOCAL_DATASETS[name]
    path = meta["out_dir"]
    status = "found" if path.exists() else "missing"
    print(f"{name}: {meta['description']}")
    print(f"  local status: {status}")
    print(f"  expected path: {path}")
    if meta["paths"]:
        print("  download paths:")
        for url in meta["paths"]:
            print(f"    - {url}")
    if meta["note"]:
        print(f"  note: {meta['note']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *WFDB_DATASETS.keys(), *LOCAL_DATASETS.keys()],
        default="all",
        help="Dataset to download or check.",
    )
    parser.add_argument(
        "--capnobase-url",
        default=os.environ.get("CAPNOBASE_ARCHIVE_URL"),
        help="Optional direct CapnoBase archive URL. Can also be set with CAPNOBASE_ARCHIVE_URL.",
    )
    parser.add_argument(
        "--local-source",
        type=Path,
        help="Optional source path to copy for a local-only dataset such as kansas, arrhythmia, or icu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(WFDB_DATASETS) + list(LOCAL_DATASETS) if args.dataset == "all" else [args.dataset]

    for name in selected:
        if name in WFDB_DATASETS:
            download_wfdb_dataset(name)
        elif name == "capnobase" and args.capnobase_url:
            download_archive(args.capnobase_url, LOCAL_DATASETS[name]["out_dir"])
        elif args.local_source and name in LOCAL_DATASETS:
            copy_local_source(args.local_source, LOCAL_DATASETS[name]["out_dir"])
        elif name in LOCAL_DATASETS:
            report_local_dataset(name)


if __name__ == "__main__":
    main()
