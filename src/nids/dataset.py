"""Reproducible reconstruction of the Week 1 split used by the Week 3 models.

These constants mirror the modelling notebook exactly.  They are used only for
the offline reproducibility check and for generating evaluation assets; the
application itself never retrains or re-splits data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack_type",
    "difficulty_level",
]

ATTACK_MAPPING = {
    "normal": "Normal",
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    "satan": "Probe",
    "ipsweep": "Probe",
    "portsweep": "Probe",
    "nmap": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "sendmail": "R2L",
    "xlock": "R2L",
    "snmpgetattack": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "httptunnel": "R2L",
    "named": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    raw_train_rows: int
    raw_test_rows: int
    duplicates_removed: int


def read_raw_files(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(raw_dir)
    train_path = directory / "KDDTrain+_20Percent.txt"
    test_path = directory / "KDDTest+.txt"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Fichiers NSL-KDD absents : " + ", ".join(missing))

    train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS, on_bad_lines="skip")
    test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS, on_bad_lines="skip")
    for frame in (train, test):
        frame["attack_category"] = (
            frame["attack_type"].str.strip().str.lower().map(ATTACK_MAPPING).fillna("Normal")
        )
    return train, test


def reproduce_split(raw_dir: str | Path, seed: int = SEED) -> DatasetSplit:
    """Rebuild the exact 70/15/15 stratified split from the notebook."""

    train, test = read_raw_files(raw_dir)
    raw_train_rows, raw_test_rows = len(train), len(test)

    train_clean = train.drop_duplicates()
    test_clean = test.drop_duplicates()
    duplicates_removed = (raw_train_rows - len(train_clean)) + (
        raw_test_rows - len(test_clean)
    )

    full = pd.concat([train_clean, test_clean], axis=0, ignore_index=True)
    full = full.drop(columns=["difficulty_level", "attack_type"])
    y = full["attack_category"]
    X = full.drop(columns=["attack_category"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        raw_train_rows=raw_train_rows,
        raw_test_rows=raw_test_rows,
        duplicates_removed=duplicates_removed,
    )
