import pandas as pd
import numpy as np

from mlcstar.utils import logger
from typing import List, Dict, Optional, Union
from mlcstar.utils import get_bin_df
from mlcstar.data.filters import collect_filter

import time


def expand_interval_to_bins(
    events_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Expand interval-based events to one row per overlapping bin.

    Events must have TIMESTAMP (start) and END_TIMESTAMP (end).
    For each event, creates a row for every bin where the event is active:
        event_start < bin_end AND event_end > bin_start

    Returns a DataFrame with standard columns (PID, FEATURE, VALUE, TIMESTAMP)
    where TIMESTAMP is set to the bin_start of each overlapping bin.
    """
    events_df = events_df.copy()
    bin_df = bin_df.copy()
    events_df["PID"] = events_df["PID"].astype("int32")
    bin_df["PID"] = bin_df["PID"].astype("int32")

    bin_pids = set(bin_df["PID"].unique())
    events_df = events_df[events_df["PID"].isin(bin_pids)]

    if len(events_df) == 0:
        logger.warning("No interval events match any bin PIDs")
        return events_df.drop(columns=["END_TIMESTAMP"], errors="ignore")

    unique_pids = events_df["PID"].unique()
    results = []

    for i in range(0, len(unique_pids), chunk_size):
        chunk_pids = set(unique_pids[i: i + chunk_size])
        chunk_events = events_df[events_df["PID"].isin(chunk_pids)]
        chunk_bins = bin_df[bin_df["PID"].isin(chunk_pids)]

        merged = chunk_events.merge(
            chunk_bins[["PID", "bin_start", "bin_end"]],
            on="PID",
        )

        overlap = merged[
            (merged["TIMESTAMP"] < merged["bin_end"])
            & (merged["END_TIMESTAMP"] > merged["bin_start"])
        ].copy()

        results.append(overlap)

    if len(results) == 0:
        logger.warning("No interval events overlap any bins")
        return events_df.drop(columns=["END_TIMESTAMP"], errors="ignore").iloc[0:0]

    result = pd.concat(results, ignore_index=True)

    result["TIMESTAMP"] = result["bin_start"]
    result = result.drop(columns=["END_TIMESTAMP", "bin_start", "bin_end"])

    logger.info(
        f"Interval expansion: {len(events_df)} events -> {len(result)} bin-aligned rows"
    )
    return result


def merge_and_aggregate(
    bin_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    agg_func: str = "mean",
    is_categorical: bool = False,
    is_multi_label: bool = False
) -> pd.DataFrame:
    """
    Enhanced merge and aggregate that handles both continuous and categorical data.
    """
    bin_df["PID"] = bin_df["PID"].astype("int32")
    subset_df["PID"] = subset_df["PID"].astype("int32")

    if not is_categorical:
        subset_df["VALUE"] = subset_df["VALUE"].astype("float")

    merged_df = pd.merge(bin_df, subset_df, on="PID", how="left")

    filtered_df = merged_df[
        (merged_df["TIMESTAMP"] >= merged_df["bin_start"])
        & (merged_df["TIMESTAMP"] <= merged_df["bin_end"])
    ]

    if is_categorical:
        if is_multi_label:
            aggregated_df = filtered_df[
                ["PID", "bin_counter", "bin_start", "bin_end", "FEATURE", "VALUE"]
            ].drop_duplicates()
        else:
            if agg_func in ["mode"]:
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan})
                    .reset_index()
                )
            elif agg_func in ["last", "first"]:
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": agg_func})
                    .reset_index()
                )
            else:
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": "last"})
                    .reset_index()
                )
    else:
        aggregation = {
            "first": "first", "mean": "mean", "max": "max", "min": "min",
            "std": "std", "sum": "sum", "count": "count", "last": "last",
        }
        agg_function = aggregation.get(agg_func, "mean")

        aggregated_df = (
            filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
            .agg({"VALUE": agg_function})
            .reset_index()
        )

    result_df = pd.merge(
        bin_df,
        aggregated_df,
        on=["PID", "bin_counter", "bin_start", "bin_end"],
        how="left",
    )

    return result_df


def map_concept(
    cfg: Dict,
    concept: str,
    agg_func: str,
    is_categorical: bool = False,
    is_multi_label: bool = False
) -> None:
    """Map concept data to time bins and save to disk."""
    output_path = f"data/interim/mapped/{concept}"

    bin_df = get_bin_df()
    logger.info(f"Prepared bin df for {concept} (categorical={is_categorical}, multi_label={is_multi_label})")

    concept_df = pd.read_pickle(f"data/interim/concepts/{concept}.pkl")
    filter_function = collect_filter(concept)
    concept_df = filter_function(concept_df)

    dfs = []
    logger.info(f"Processing {len(concept_df.FEATURE.unique())} features")

    for feat in concept_df.FEATURE.unique():
        logger.info(f"Processing feature: {feat}")
        subset = concept_df[concept_df.FEATURE == feat]

        result_df = merge_and_aggregate(
            bin_df,
            subset,
            agg_func=agg_func,
            is_categorical=is_categorical,
            is_multi_label=is_multi_label
        )

        dfs.append(result_df)

    logger.info("Concatenating feature dataframes")

    if len(dfs) < 1:
        logger.warning(f"Concept {concept} failed - no features processed")
        bin_df["FEATURE"] = np.nan
        bin_df["VALUE"] = np.nan
        bin_df.to_pickle(f"{output_path}_{agg_func}.pkl")
        bin_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)
    else:
        result_df = (
            pd.concat(dfs)
            .drop_duplicates()
            .sort_values(["PID", "bin_counter"])
            .reset_index(drop=True)
        )

        if is_categorical and is_multi_label:
            logger.info("Expanding multi-label values")
            expanded_rows = []
            for idx, row in result_df.iterrows():
                if pd.notna(row['VALUE']):
                    if isinstance(row['VALUE'], list):
                        for val in row['VALUE']:
                            new_row = row.copy()
                            new_row['VALUE'] = val
                            expanded_rows.append(new_row)
                    else:
                        expanded_rows.append(row)
                else:
                    expanded_rows.append(row)
            result_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
            logger.info(f"Expanded to {len(result_df)} rows (from multi-label)")

        grouped = result_df.groupby(["PID", "bin_counter"])

        def filter_rows(group):
            if group["FEATURE"].isna().all() and group["VALUE"].isna().all():
                return group
            else:
                return group.dropna(subset=["FEATURE", "VALUE"])

        logger.info(f"Cleaning binned {concept} dataframe")
        filtered_df = grouped.apply(filter_rows).reset_index(drop=True)

        logger.info(f"Final shape: {filtered_df.shape}")

        logger.info(f"Saving file to {output_path}")
        filtered_df.to_pickle(f"{output_path}_{agg_func}.pkl")
        filtered_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)


def merge_and_aggregate_optimized(
    bin_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    agg_func: str = "mean",
    is_categorical: bool = False,
    is_multi_label: bool = False
) -> pd.DataFrame:
    """ULTRA-OPTIMIZED merge and aggregate."""
    start_time = time.time()

    if 'TIMESTAMP' not in subset_df.columns:
        raise ValueError(f"subset_df missing TIMESTAMP column. Available: {subset_df.columns.tolist()}")

    if not pd.api.types.is_datetime64_any_dtype(subset_df['TIMESTAMP']):
        subset_df = subset_df.copy()
        subset_df['TIMESTAMP'] = pd.to_datetime(subset_df['TIMESTAMP'])

    bin_df = bin_df.copy()
    subset_df = subset_df.copy()
    bin_df["PID"] = bin_df["PID"].astype("int32")
    subset_df["PID"] = subset_df["PID"].astype("int32")

    bin_pids = set(bin_df['PID'].unique())
    subset_df = subset_df[subset_df['PID'].isin(bin_pids)]

    if len(subset_df) == 0:
        logger.warning("No data after PID filtering")
        result_df = bin_df.copy()
        result_df['FEATURE'] = np.nan
        result_df['VALUE'] = np.nan
        return result_df

    logger.debug(f"  Assigning bins for {len(subset_df)} rows...")
    t0 = time.time()

    subset_grouped = subset_df.groupby('PID', sort=False)
    bin_grouped = bin_df.groupby('PID', sort=False)

    results = []
    unique_pids = subset_df['PID'].unique()

    batch_size = 1000

    for i, pid in enumerate(unique_pids):
        if i > 0 and i % batch_size == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(unique_pids) - i) / rate
            logger.debug(f"    Progress: {i}/{len(unique_pids)} ({100*i/len(unique_pids):.0f}%) - ETA: {eta:.0f}s")

        try:
            patient_data = subset_grouped.get_group(pid)
            patient_bins = bin_grouped.get_group(pid)
        except KeyError:
            continue

        if len(patient_bins) == 0:
            continue

        timestamps = patient_data['TIMESTAMP'].values
        bin_starts = patient_bins['bin_start'].values
        bin_ends = patient_bins['bin_end'].values

        indices = np.searchsorted(bin_starts, timestamps, side='right') - 1

        valid_mask = (indices >= 0) & (indices < len(patient_bins))

        if not valid_mask.any():
            continue

        valid_indices = indices[valid_mask]
        valid_timestamps = timestamps[valid_mask]

        within_bin = valid_timestamps < bin_ends[valid_indices]

        if not within_bin.any():
            continue

        final_valid_mask = np.zeros(len(patient_data), dtype=bool)
        final_valid_mask[np.where(valid_mask)[0][within_bin]] = True

        patient_result = patient_data[final_valid_mask].copy()

        final_indices = valid_indices[within_bin]
        patient_result['bin_counter'] = patient_bins['bin_counter'].iloc[final_indices].values
        patient_result['bin_start'] = patient_bins['bin_start'].iloc[final_indices].values
        patient_result['bin_end'] = patient_bins['bin_end'].iloc[final_indices].values
        patient_result['bin_freq'] = patient_bins['bin_freq'].iloc[final_indices].values

        results.append(patient_result)

    if len(results) == 0:
        logger.warning("  No data matched any bins!")
        result_df = bin_df.copy()
        result_df['FEATURE'] = np.nan
        result_df['VALUE'] = np.nan
        return result_df

    filtered_df = pd.concat(results, ignore_index=True)

    elapsed = time.time() - t0
    logger.debug(f"  Bin assignment: {len(subset_df)} → {len(filtered_df)} rows in {elapsed:.1f}s")

    t0 = time.time()

    if is_categorical:
        if is_multi_label:
            aggregated_df = filtered_df[
                ["PID", "bin_counter", "bin_start", "bin_end", "FEATURE", "VALUE"]
            ].drop_duplicates()
        else:
            if agg_func == "mode":
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan})
                    .reset_index()
                )
            elif agg_func in ["last", "first"]:
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": agg_func})
                    .reset_index()
                )
            else:
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": "last"})
                    .reset_index()
                )
    else:
        aggregation = {
            "first": "first", "mean": "mean", "max": "max", "min": "min",
            "std": "std", "sum": "sum", "count": "count", "last": "last",
        }
        agg_function = aggregation.get(agg_func, "mean")

        filtered_df['VALUE'] = pd.to_numeric(filtered_df['VALUE'], errors='coerce')
        filtered_df = filtered_df[filtered_df['VALUE'].notna()]

        if len(filtered_df) == 0:
            logger.warning("  No numeric values!")
            result_df = bin_df.copy()
            result_df['FEATURE'] = np.nan
            result_df['VALUE'] = np.nan
            return result_df

        aggregated_df = (
            filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
            .agg({"VALUE": agg_function})
            .reset_index()
        )

    logger.debug(f"  Aggregation: {time.time()-t0:.1f}s")

    t0 = time.time()
    result_df = pd.merge(
        bin_df,
        aggregated_df,
        on=["PID", "bin_counter", "bin_start", "bin_end"],
        how="left",
    )
    logger.debug(f"  Merge back: {time.time()-t0:.1f}s")

    total_time = time.time() - start_time
    logger.debug(f"  Total feature processing: {total_time:.1f}s")

    return result_df


def clean_dataframe_ultra_fast(result_df):
    """Ultra-fast cleaning using agg + merge."""
    t0 = time.time()

    logger.debug(f"  Cleaning {len(result_df)} rows...")

    group_info = result_df.groupby(['PID', 'bin_counter'], sort=False).agg({
        'FEATURE': lambda x: x.notna().any(),
        'VALUE': lambda x: x.notna().any()
    }).reset_index()

    group_info.columns = ['PID', 'bin_counter', '_has_feature', '_has_value']
    group_info['_has_any_data'] = group_info['_has_feature'] | group_info['_has_value']

    result_with_info = result_df.merge(
        group_info[['PID', 'bin_counter', '_has_any_data']],
        on=['PID', 'bin_counter'],
        how='left'
    )

    mask = (~result_with_info['_has_any_data']) | (
        result_with_info['FEATURE'].notna() & result_with_info['VALUE'].notna()
    )

    filtered_df = result_with_info[mask].drop(columns=['_has_any_data']).reset_index(drop=True)

    total = time.time() - t0
    logger.info(f"[{total:.1f}s] Cleaned: {len(filtered_df)} rows (removed {len(result_df) - len(filtered_df)} rows)")

    return filtered_df


def map_concept_optimized(
    cfg,
    concept: str,
    agg_func: str,
    is_categorical: bool = False,
    is_multi_label: bool = False
) -> None:
    """FINAL ultra-optimized mapper with all fixes."""
    from mlcstar.utils import get_bin_df
    from mlcstar.data.filters import collect_filter

    overall_start = time.time()

    logger.info(f"{'='*80}")
    logger.info(f"MAPPING: {concept} with {agg_func} (ULTRA-OPTIMIZED)")
    logger.info(f"{'='*80}")

    output_path = f"data/interim/mapped/{concept}"

    t0 = time.time()
    bin_df = get_bin_df()
    logger.info(f"[{time.time()-t0:.1f}s] Loaded bin_df: {bin_df['PID'].nunique()} patients")

    t0 = time.time()
    concept_df = pd.read_pickle(f"data/interim/concepts/{concept}.pkl")
    filter_function = collect_filter(concept)
    concept_df = filter_function(concept_df)
    logger.info(f"[{time.time()-t0:.1f}s] Loaded concept: {len(concept_df)} rows, {concept_df['PID'].nunique()} patients")

    if 'END_TIMESTAMP' in concept_df.columns:
        logger.info("Expanding interval events to per-bin rows...")
        t1 = time.time()
        concept_df = expand_interval_to_bins(concept_df, bin_df)
        logger.info(f"[{time.time()-t1:.1f}s] Expanded to {len(concept_df)} rows")

    if 'TIMESTAMP' not in concept_df.columns:
        raise ValueError(f"Concept {concept} missing TIMESTAMP column. Available: {concept_df.columns.tolist()}")

    dfs = []
    features = concept_df.FEATURE.unique()
    logger.info(f"Processing {len(features)} features...")

    feature_times = []

    for i, feat in enumerate(features):
        t0 = time.time()

        logger.info(f"  Feature {i+1}/{len(features)}: {feat}")
        subset = concept_df[concept_df.FEATURE == feat]

        result_df = merge_and_aggregate_optimized(
            bin_df,
            subset,
            agg_func=agg_func,
            is_categorical=is_categorical,
            is_multi_label=is_multi_label
        )

        elapsed = time.time() - t0
        feature_times.append(elapsed)
        logger.info(f"    Time: {elapsed:.1f}s")

        dfs.append(result_df)

    avg_time = np.mean(feature_times)
    logger.info(f"Average time per feature: {avg_time:.1f}s")

    logger.info("Concatenating feature dataframes...")
    t0 = time.time()

    if len(dfs) < 1:
        logger.warning(f"Concept {concept} failed - no features processed")
        bin_df["FEATURE"] = np.nan
        bin_df["VALUE"] = np.nan
        bin_df.to_pickle(f"{output_path}_{agg_func}.pkl")
        bin_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)
        return

    result_df = pd.concat(dfs, ignore_index=True).drop_duplicates().sort_values(["PID", "bin_counter"]).reset_index(drop=True)

    logger.info(f"[{time.time()-t0:.1f}s] Concatenated: {len(result_df)} rows")

    if is_categorical and is_multi_label:
        t0 = time.time()
        logger.info("Expanding multi-label values...")

        is_list = result_df['VALUE'].apply(lambda x: isinstance(x, list) if pd.notna(x) else False)

        if is_list.any():
            list_rows = result_df[is_list].copy()
            non_list_rows = result_df[~is_list].copy()
            list_rows_expanded = list_rows.explode('VALUE').reset_index(drop=True)
            result_df = pd.concat([non_list_rows, list_rows_expanded], ignore_index=True)
            logger.info(f"[{time.time()-t0:.1f}s] Expanded {is_list.sum()} multi-label rows")
        else:
            logger.info(f"[{time.time()-t0:.1f}s] No multi-label values to expand")

    logger.info(f"Cleaning dataframe...")
    filtered_df = clean_dataframe_ultra_fast(result_df)

    logger.info(f"Final shape: {filtered_df.shape}")

    t0 = time.time()
    logger.info(f"Saving to {output_path}...")
    filtered_df.to_pickle(f"{output_path}_{agg_func}.pkl")
    filtered_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)
    logger.info(f"[{time.time()-t0:.1f}s] Saved")

    total_time = time.time() - overall_start
    logger.info(f"{'='*80}")
    logger.info(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"{'='*80}\n")
