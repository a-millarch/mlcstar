"""
make_data.py — Data preparation pipeline for mlcstar.

Steps:
1. Collect raw concept files from Azure (collect_subsets)
2. Filter raw files to in-hospital records (filter_subsets_inhospital)
3. Map/bin concepts to temporal grid (map_data_optimized)

Run directly to execute the full pipeline:
    python -m mlcstar.make_data
"""

import os
import pandas as pd
import numpy as np

from mlcstar.utils import ProjectManager, cfg, logger
from mlcstar.utils import is_file_present, are_files_present

from mlcstar.data.collectors import collect_subsets
import mlcstar.data.build_patient_info as bpi
from mlcstar.data.filters import filter_subsets_inhospital
from mlcstar.data.mapper import map_concept, map_concept_optimized


def proces_raw_concepts(cfg, base=None, reset=False):
    """
    Collect raw concept CSV files from Azure if not already present.

    Covers both default_load_filenames (small, direct download) and
    large_load_filenames (chunked download + filter).

    Args:
        cfg: Configuration dictionary.
        base: Optional pre-loaded base dataframe (used for population filter).
        reset: If True, re-collect even if files are already present.
    """
    subsets_filenames = cfg["default_load_filenames"] + cfg["large_load_filenames"]
    if (
        are_files_present("data/raw", subsets_filenames, extension=".csv")
        and not reset
    ):
        logger.info("All raw subsets found, continuing")
    else:
        logger.info("Raw subsets missing, collecting")
        collect_subsets(cfg, base=base)


def proces_inhospital_concepts(cfg, reset=False):
    """
    Filter raw concept files to in-hospital records and save as pkl.

    Reads data/external/metadata.csv for datetime column names and offset info.
    Saves filtered files to data/interim/concepts/<filename>.pkl.

    Args:
        cfg: Configuration dictionary.
        reset: If True, re-filter even if interim files are already present.
    """
    subsets_filenames = cfg["default_load_filenames"] + cfg["large_load_filenames"]
    if (
        are_files_present("data/interim/concepts", subsets_filenames, extension=".pkl")
        and not reset
    ):
        logger.info("Interim subsets found, continuing")
    else:
        logger.info("Filtering subsets to in-hospital records")
        filter_subsets_inhospital(cfg)


def map_data(cfg):
    """
    Map and bin all concepts to the temporal grid (non-optimized version).

    Saves CSV and pkl files to data/interim/mapped/.
    """
    logger.info("Mapping data to bins")
    map_dir = "data/interim/mapped/"
    for concept in cfg["concepts"]:
        for agg_func in cfg["agg_func"][concept]:
            if is_file_present(
                f"{map_dir}{concept}_{agg_func}.csv"
            ) and is_file_present(f"{map_dir}{concept}_{agg_func}.pkl"):
                pass
            else:
                logger.debug(f"Binning and mapping {concept} with agg_func: {agg_func}")
                is_categorical = concept in cfg["dataset"]["ts_cat_names"]
                is_multi_label = concept in cfg["dataset"]["ts_categorical_multi_label"]
                map_concept(cfg, concept, agg_func, is_categorical, is_multi_label)


def map_data_optimized(cfg):
    """
    Map and bin all concepts to the temporal grid (optimized version).

    Skips concepts that already have output files.
    Saves output to data/interim/mapped/<concept>_<agg_func>.csv.
    """
    logger.info("Mapping data to bins (optimized)")
    map_dir = "data/interim/mapped/"

    for concept in cfg["concepts"]:
        for agg_func in cfg["agg_func"][concept]:
            output_file = f"{map_dir}{concept}_{agg_func}.csv"

            if os.path.exists(output_file):
                logger.info(f"Skipping {concept}_{agg_func} (already exists)")
                continue

            logger.info(f"Processing {concept} with {agg_func}")

            is_categorical = concept in cfg["dataset"]["ts_cat_names"]
            is_multi_label = concept in cfg["dataset"]["ts_categorical_multi_label"]

            map_concept_optimized(
                cfg,
                concept,
                agg_func,
                is_categorical,
                is_multi_label
            )


if __name__ == '__main__':
    pm = ProjectManager()
    logger = pm.setup_logging(print_only=True)

    # Step 1: Create base_df (cohort construction)
    if is_file_present(cfg['base_df_path']):
        logger.info(f"base_df found at {cfg['base_df_path']}, skipping creation")
    else:
        base = bpi.create_base_df(cfg)

    # Step 2: Create bin_df (temporal grid)
  #  if is_file_present(cfg['bin_df_path']):
  #      logger.info(f"bin_df found at {cfg['bin_df_path']}, skipping creation")
  #  else:
  #      bpi.create_bin_df(cfg)

    # Step 3: Filter raw concepts to in-hospital records
    proces_inhospital_concepts(cfg, reset=False)

    # Step 4: Map/bin concepts to temporal grid
    #map_data_optimized(cfg)

    logger.info("Data preparation pipeline complete.")

