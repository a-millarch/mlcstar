import os
import hashlib
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from rich.logging import RichHandler
from datetime import datetime


class ProjectManager:
    """
    Manages project directory settings.

    Simplified version for standalone local use (no Azure compute instance
    path requirements).

    Example:
        >>> pm = ProjectManager()
        >>> logger = pm.setup_logging(print_only=True)
    """
    def __init__(self, workdir=None):
        self.init_dir = os.getcwd()
        self.workdir = workdir if workdir else self.init_dir
        self.project_name = "mlcstar"

        if workdir and os.path.exists(workdir):
            os.chdir(workdir)
            print(f"Working directory changed to: {workdir}")
        elif workdir:
            print(f"Directory does not exist: {workdir}")

    def setup_logging(self, print_only: bool = False):
        """
        Configures logging for the project.

        Args:
            print_only (bool): If True, only print to console, don't save to file.

        Returns:
            logging.Logger: The configured logger instance.
        """
        logger_name = self.project_name
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # Remove any existing handlers
        if logger.hasHandlers():
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()

        formatter = logging.Formatter(f'%(levelname)s - %(asctime)s - [{self.project_name}]\n%(message)s')

        if not print_only:
            log_dir = Path(self.workdir) / 'logging'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / 'app.log'

            if log_file_path.exists() and log_file_path.stat().st_size > 0:
                self.clear_log(log_file_path)

            file_handler = RotatingFileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        rich_handler = RichHandler(markup=True)
        rich_handler.setFormatter(formatter)
        logger.addHandler(rich_handler)

        logging.getLogger('matplotlib.font_manager').disabled = True
        self.logger = logger
        return logger

    def clear_log(self, log_file_path=None):
        if log_file_path is None:
            log_file_path = Path(self.workdir) / 'logging' / 'app.log'

        current_date = datetime.now()
        archive_folder = (
            Path(self.workdir) / 'logging' / 'archive'
            / current_date.strftime('%Y/%m/%d')
        )
        archive_folder.mkdir(parents=True, exist_ok=True)
        archive_file_path = archive_folder / f"log_{current_date.strftime('%H%M')}.txt"

        with open(log_file_path, 'r') as f:
            log_contents = f.read()
        with open(archive_file_path, 'w') as f:
            f.write(log_contents)
        with open(log_file_path, 'w'):
            pass


# ---------------------------------------------------------------------------
# Module-level logger and config
# ---------------------------------------------------------------------------

try:
    _pm = ProjectManager()
    logger = _pm.setup_logging(print_only=True)
except Exception:
    logger = logging.getLogger('mlcstar')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(RichHandler(markup=True))

pd.options.mode.chained_assignment = None


def save_figure(fig, filename, save_dir='reports/studyfigs'):
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, f'{filename}.png')
    fig.savefig(png_path, dpi=1200, bbox_inches='tight')


def get_cfg(cfg_path="configs/defaults.yaml"):
    with open(cfg_path) as file:
        return yaml.safe_load(file)


cfg = get_cfg()


def count_csv_rows(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        return sum(1 for line in f)


def get_base_df(base_df_path=None):
    if base_df_path is None:
        base_df_path = cfg["base_df_path"]
    return pd.read_pickle(base_df_path)


def get_bin_df(bin_df_path=None):
    if bin_df_path is None:
        bin_df_path = cfg["bin_df_path"]
    bin_df = pd.read_pickle(bin_df_path)
    expected_freqs = set(cfg["bin_intervals"].values())
    actual_freqs = set(bin_df["bin_freq"].unique())
    if expected_freqs != actual_freqs:
        raise ValueError(
            f"bin_df is stale — frequencies don't match cfg['bin_intervals']. "
            f"Expected: {sorted(expected_freqs)}, Got: {sorted(actual_freqs)}. "
            f"Regenerate bin_df with the current bin_intervals config."
        )
    return bin_df


def get_train_test_split(cfg, base_df=None, return_indices=False):
    """
    Get train/test split using the strategy from config.

    Args:
        cfg: Configuration dictionary
        base_df: Base dataframe (if None, will load using get_base_df())
        return_indices: If True, return indices instead of dataframes

    Returns:
        If return_indices=False: (train_df, test_df)
        If return_indices=True: (train_indices, test_indices)
    """
    if base_df is None:
        base_df = get_base_df()

    # Apply exclusion if specified
    exclusion = cfg["dataset"].get("exclusion", 0)
    if exclusion == "lvl1tc":
        base_df = base_df[base_df.LVL1TC == 1]

    # Sort by date to ensure temporal ordering
    base_df = base_df.sort_values('ServiceDate').reset_index(drop=True)

    holdout_type = cfg.get("holdout_type", "temporal")

    if holdout_type == "temporal":
        split_date = pd.to_datetime(cfg["holdout_split_date"])
        train_mask = base_df.ServiceDate <= split_date
        test_mask = base_df.ServiceDate > split_date

        if return_indices:
            return np.where(train_mask)[0], np.where(test_mask)[0]
        return base_df[train_mask].copy(), base_df[test_mask].copy()

    elif holdout_type == "random":
        holdout_fraction = cfg.get("holdout_fraction", 0.2)
        seed = cfg.get("seed", 2024)
        try:
            train_df, test_df = train_test_split(
                base_df, test_size=holdout_fraction, random_state=seed,
                stratify=base_df[cfg["target"]], shuffle=True
            )
        except ValueError:
            train_df, test_df = train_test_split(
                base_df, test_size=holdout_fraction, random_state=seed, shuffle=True
            )
        if return_indices:
            return train_df.index.values, test_df.index.values
        return train_df, test_df

    else:
        raise ValueError(f"Unknown holdout_type: {holdout_type}. Use 'temporal' or 'random'.")


def align_dataframes(df_a, df_b, fill_value=0.0):
    all_columns = set(df_a.columns) | set(df_b.columns)
    missing_in_a = all_columns - set(df_a.columns)
    missing_in_b = all_columns - set(df_b.columns)
    for col in missing_in_a:
        df_a[col] = fill_value
    for col in missing_in_b:
        df_b[col] = fill_value
    non_numeric_columns = [col for col in all_columns if not col.isdigit()]
    numeric_columns = sorted([col for col in all_columns if col.isdigit()], key=int)
    sorted_columns = non_numeric_columns + numeric_columns
    return df_a.reindex(columns=sorted_columns), df_b.reindex(columns=sorted_columns)


def create_enumerated_id(df, string_col, datetime_col):
    df["unique_id"] = df[string_col].astype(str) + df[datetime_col].astype(str)
    df_unique = df.drop_duplicates(subset=["unique_id"]).copy()
    df_unique["PID"] = range(1, len(df_unique) + 1)
    df = df.merge(df_unique[["unique_id", "PID"]], on="unique_id", how="left")
    df.drop(columns=["unique_id"], inplace=True)
    return df


def ensure_datetime(df, column_name):
    if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
    return df


def is_file_present(file_path: str):
    return os.path.isfile(file_path)


def are_files_present(directory, filenames, extension):
    return all(
        os.path.isfile(os.path.join(directory, f"{filename}{extension}"))
        for filename in filenames
    )


def md5_checksum(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# Unit conversion helpers
def inches_to_cm(inches):
    return inches * 2.54


def feet_to_cm(feet):
    return feet * 30.48


def pounds_to_kg(pounds):
    return pounds * 0.45359237


def ounces_to_kg(ounces):
    return ounces * 0.0283495231


def mark_keywords_in_df(df, text_column, keywords, timestamp=None,
                         base_timestamp=None, t_delta=12, new_column="keyword_present"):
    keyword_pattern = "|".join(keywords)
    df[new_column] = df[text_column].str.contains(keyword_pattern, case=False, na=False)
    if timestamp and base_timestamp:
        df[timestamp] = pd.to_datetime(df[timestamp], errors="coerce")
        df[base_timestamp] = pd.to_datetime(df[base_timestamp], errors="coerce")
        df[f"within_{t_delta}_hours"] = (
            df[timestamp] - df[base_timestamp]
        ) <= pd.Timedelta(hours=t_delta)
    return df


def get_concept(concept, cfg) -> dict:
    """Get concept from mapped files."""
    drop_cols = cfg["drop_features"].get(concept, [])
    concept_dict = {}
    for agg_func in cfg["agg_func"][concept]:
        logger.debug(f"Loading {concept}.agg_func: {agg_func}")
        df = pd.read_csv(f"data/interim/mapped/{concept}_{agg_func}.csv")
        try:
            df = df[~df.FEATURE.isin(drop_cols + [np.nan])]
        except Exception:
            df = df[~df.FEATURE.isin([np.nan])]
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
        concept_dict[agg_func] = df
    return concept_dict
