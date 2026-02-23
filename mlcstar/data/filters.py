# filters.py
"""
Data filtering functions for mlcstar.

Infrastructure functions (filter_subsets_inhospital, filter_inhospital,
collect_filter) are domain-agnostic and ready to use.

Concept-specific filter functions are provided as templates.
Replace the body of each function with logic matching your EHR data schema.

# TODO: ADAPT FOR mlcstar
# 1. Replace column name strings with your EHR's actual column names.
# 2. Import only the mappings you actually define in mappings.py.
# 3. Add/remove concept filter functions to match your cfg["concepts"].
# 4. Update the collect_filter() dict to map concept names to filter functions.
"""

import pandas as pd
import numpy as np
import gc

from mlcstar.utils import logger, get_cfg, get_base_df
from mlcstar.utils import ensure_datetime, is_file_present

from mlcstar.data.mappings import (
    VITALS_MAP, BP_TYPES, HEIGHT_WEIGHT_MAP,
    LABS_FEATURE_MAP, LABS_REVERSE_MAP,
    ICU_MAP,
    ATC_LVL3_MAP, ATC_LVL4_MAP, MEDICATION_ACTION_LIST,
    PROCEDURE_MAP, PROCEDURE_REVERSE_MAP, PROCEDURE_INCLUDE_LIST,
    ADT_PATTERNS, classify_department,
)


# ============================================================================
# INFRASTRUCTURE FUNCTIONS (domain-agnostic — do not modify)
# ============================================================================

def filter_subsets_inhospital(cfg, base=None):
    """
    Filter all raw concept files to in-hospital records and save to interim.

    Reads metadata from data/external/metadata.csv which must contain:
        - filename: concept name (matches cfg['default_load_filenames'])
        - dt_colname: name of the datetime column in that file
        - ts_offset: number of days offset allowed outside admission window

    Saves filtered files to data/interim/concepts/<filename>.pkl.
    """
    metadata = pd.read_csv("data/external/metadata.csv")

    missing_files = [
        file
        for file in cfg["default_load_filenames"]
        if file not in metadata["filename"].values
    ]
    assert (
        len(missing_files) == 0
    ), f"{missing_files} are not present in data/external/metadata.csv"

    df = pd.DataFrame()

    if base is None:
        base = get_base_df()

    for filename in metadata.filename:
        del df
        gc.collect()
        logger.debug(f"Filtering {filename}")
        df = pd.read_csv(f"data/raw/{filename}.csv", low_memory=False, index_col=0)

        dt_name = str(
            metadata.loc[metadata["filename"] == filename]["dt_colname"].iat[0]
        )
        offset = int(
            metadata.loc[metadata["filename"] == filename]["ts_offset"].iat[0]
        )

        filtered_df = filter_inhospital(base, df, cfg, dt_name, offset=offset)
        filtered_df.to_pickle(f"data/interim/concepts/{filename}.pkl")


def filter_inhospital(
    base: pd.DataFrame, df: pd.DataFrame, cfg, dt_name: str, offset=1
) -> pd.DataFrame:
    """
    Filter a dataframe to records falling within each patient's admission window.

    Args:
        base: Base dataframe with columns [PID, CPR_hash, start, end].
        df: Raw concept dataframe (must contain CPR_hash and dt_name columns).
        cfg: Configuration dictionary.
        dt_name: Name of the datetime column in df.
        offset: Days of slack outside admission window (default 1).

    Returns:
        Filtered dataframe with original columns plus PID.
    """
    colnames = df.columns.to_list()
    df = ensure_datetime(df, dt_name)
    merged_df = base[["PID", "CPR_hash", "start", "end"]].merge(
        df, on="CPR_hash", how="left"
    )

    filtered_df = merged_df[
        (merged_df[dt_name] >= merged_df["start"] - pd.DateOffset(days=offset))
        & (merged_df[dt_name] <= merged_df["end"] + pd.DateOffset(days=offset))
    ]
    filtered_df = filtered_df.drop_duplicates().reset_index(drop=True)

    logger.debug(f">Original df len: {len(df)}, new df len: {len(filtered_df)}")
    return filtered_df[colnames + ["PID"]]


# ============================================================================
# CONCEPT-SPECIFIC FILTER FUNCTIONS (templates — adapt for mlcstar)
# ============================================================================
# Each function receives the output of filter_inhospital() for that concept
# and must return a DataFrame with columns: [PID, FEATURE, VALUE, TIMESTAMP].
# Optionally include END_TIMESTAMP for interval-type concepts (e.g. ADT).


def filter_vitals(vit):
    """
    Filter and standardize vital signs data.

    # TODO: ADAPT FOR mlcstar
    # 1. Rename columns to VALUE, FEATURE, TIMESTAMP.
    # 2. Apply BP splitting if needed (uses BP_TYPES from mappings).
    # 3. Map feature names using VITALS_MAP.
    # 4. Filter to numeric values and known feature names.
    """
    vit = vit.copy()

    # TODO: rename columns to standard names
    # Example:
    # vit.rename(columns={
    #     'Value': 'VALUE',
    #     'ParameterName': 'FEATURE',
    #     'MeasurementTime': 'TIMESTAMP',
    # }, inplace=True)

    # TODO: split combined BP strings (e.g. '120/80') using BP_TYPES
    # for bt in BP_TYPES:
    #     mask = vit['FEATURE'] == bt
    #     ...

    # TODO: map feature names using VITALS_MAP
    # vit['FEATURE'] = vit['FEATURE'].replace(to_replace=VITALS_MAP)

    # TODO: filter to known features and non-null numeric values
    # vit = vit[vit.FEATURE.isin(list(set(VITALS_MAP.values())))]

    vit = vit[["TIMESTAMP", "PID", "FEATURE", "VALUE"]]
    return vit


def filter_labs(lab):
    """
    Filter and standardize laboratory result data.

    # TODO: ADAPT FOR mlcstar
    # 1. Clean numeric strings (remove commas, asterisks, <, > prefixes).
    # 2. Filter to tests in LABS_FEATURE_MAP.
    # 3. Rename columns and apply LABS_REVERSE_MAP.
    """
    # TODO: clean result string column
    # lab['ResultValue'] = lab['ResultValue'].str.replace(',', '.')
    # lab['ResultValue'] = lab['ResultValue'].str.replace('*', '')

    # TODO: filter to relevant tests
    # include_list = [name for names in LABS_FEATURE_MAP.values() for name in names]
    # lab = lab[lab['TestName'].isin(include_list)].copy()

    # TODO: rename columns
    # lab.rename(columns={
    #     'TestName': 'FEATURE',
    #     'ResultValue': 'VALUE',
    #     'SampleTime': 'TIMESTAMP',
    # }, inplace=True)

    # TODO: apply reverse map and convert to numeric
    # lab.FEATURE = lab.FEATURE.replace(LABS_REVERSE_MAP)
    # lab['VALUE'] = pd.to_numeric(lab['VALUE'], errors='coerce')
    # lab = lab.dropna(subset=['VALUE'])

    logger.info(f"Using {len(lab)} observations of labs")
    return lab


def filter_ita(ita):
    """
    Filter and standardize ICU/ward score data.

    # TODO: ADAPT FOR mlcstar (or remove if not applicable)
    # 1. Rename columns to FEATURE, VALUE, TIMESTAMP.
    # 2. Map feature names using ICU_MAP.
    """
    ita = ita.copy()

    # TODO: rename columns
    # ita.rename(columns={
    #     'ScoreName': 'FEATURE',
    #     'ScoreValue': 'VALUE',
    #     'MeasurementTime': 'TIMESTAMP',
    # }, inplace=True)

    # TODO: apply ICU_MAP
    # ita['FEATURE'] = ita['FEATURE'].replace(to_replace=ICU_MAP)

    return ita


def reverse_dict_replace(original_dict, df, atc_level):
    """Invert a dict and replace ATC codes in df with category names."""
    inverted_dict = {}
    for key, value in original_dict.items():
        if isinstance(value, list):
            for item in value:
                inverted_dict[item] = key
        else:
            inverted_dict[value] = key
    df["FEATURE"] = (
        df[f"ATC{atc_level}"]
        .replace(inverted_dict)
        .where(df[f"ATC{atc_level}"].isin(inverted_dict.keys()), np.nan)
    )
    logger.info(
        f">>Medicine: found {len(df[df.FEATURE.notnull()])} administrations "
        f"of a ATC level {atc_level} drug"
    )
    return df


def filter_medicin(med):
    """
    Filter and standardize medication administration data.

    # TODO: ADAPT FOR mlcstar
    # 1. Filter to valid administration actions (MEDICATION_ACTION_LIST).
    # 2. Extract ATC3/ATC4 prefixes and map to category names.
    # 3. Rename timestamp column to TIMESTAMP.
    """
    med = med[med.Handling.isin(MEDICATION_ACTION_LIST)].copy()  # TODO: rename 'Handling' column
    med["ATC3"] = med.ATC.str[:3]   # TODO: rename 'ATC' column if different
    med["ATC4"] = med.ATC.str[:4]

    med3 = reverse_dict_replace(ATC_LVL3_MAP, med.copy(deep=True), 3)
    med4 = reverse_dict_replace(ATC_LVL4_MAP, med.copy(deep=True), 4)
    med = pd.concat([med3, med4]).drop_duplicates().copy()
    med = med[med["FEATURE"].notnull()].copy()
    med["VALUE"] = med["FEATURE"]
    med["FEATURE"] = "medication"

    # TODO: rename your administration start/end timestamp columns
    # med.rename(columns={'AdminStart': 'start', 'AdminEnd': 'end'}, inplace=True)
    med["TIMESTAMP"] = med["start"]
    logger.info(f"Using {len(med)} observations of medicine")
    return med


def filter_procedures(proc):
    """
    Filter and standardize procedure data.

    # TODO: ADAPT FOR mlcstar
    # 1. Filter to procedure codes in PROCEDURE_INCLUDE_LIST.
    # 2. Rename columns to VALUE and TIMESTAMP.
    # 3. Apply PROCEDURE_REVERSE_MAP to map codes to category names.
    """
    proc = proc[proc["ProcedureCode"].isin(PROCEDURE_INCLUDE_LIST)].copy()  # TODO: rename column

    proc.rename(
        columns={"ProcedureCode": "VALUE", "ServiceDatetime": "TIMESTAMP"},  # TODO: adapt column names
        inplace=True,
    )

    proc.VALUE = proc.VALUE.replace(PROCEDURE_REVERSE_MAP)
    logger.info(f"Using {len(proc)} observations of procedures")
    proc["FEATURE"] = "procedures"
    return proc


def filter_adt(adt, base_df=None):
    """
    Filter ADT (Admission-Discharge-Transfer) events and classify departments.

    # TODO: ADAPT FOR mlcstar
    # 1. Parse your admit/discharge datetime columns.
    # 2. Apply classify_department() from mappings (after filling ADT_PATTERNS).
    # 3. Rename to TIMESTAMP (admit) and END_TIMESTAMP (discharge).

    Args:
        adt: Raw ADT DataFrame (after filter_inhospital).
        base_df: Optional base DataFrame for filling missing discharge times.
    """
    adt = adt.copy()

    # TODO: rename your admit/discharge datetime columns
    # adt['Flyt_ind'] = pd.to_datetime(adt['AdmitTime'], errors='coerce')
    # adt['Flyt_ud'] = pd.to_datetime(adt['DischargeTime'], errors='coerce')

    # Classify departments
    adt["VALUE"] = adt["Afsnit"].apply(classify_department)  # TODO: rename 'Afsnit' column
    adt["FEATURE"] = "ADT"

    adt = adt[adt["VALUE"].notna()].copy()
    logger.info(f"ADT: {len(adt)} events after department classification")

    adt = adt.sort_values(["PID", "Flyt_ind"]).reset_index(drop=True)

    # Handle missing end times: fill from next event's start
    adt["next_flyt_ind"] = adt.groupby("PID")["Flyt_ind"].shift(-1)
    mask_missing_end = adt["Flyt_ud"].isna()
    adt.loc[mask_missing_end, "Flyt_ud"] = adt.loc[mask_missing_end, "next_flyt_ind"]

    if mask_missing_end.any() and adt["Flyt_ud"].isna().any():
        if base_df is None:
            base_df = get_base_df()
        end_map = base_df.set_index("PID")["end"]
        still_missing = adt["Flyt_ud"].isna()
        adt.loc[still_missing, "Flyt_ud"] = adt.loc[still_missing, "PID"].map(end_map).values

    adt = adt.drop(columns=["next_flyt_ind"])
    adt = adt.dropna(subset=["Flyt_ud"])

    adt["TIMESTAMP"] = adt["Flyt_ind"]
    adt["END_TIMESTAMP"] = adt["Flyt_ud"]

    logger.info(f"Using {len(adt)} ADT observations")
    return adt


# ============================================================================
# FILTER REGISTRY
# ============================================================================

def collect_filter(concept: str):
    """
    Return the filter function for a given concept name.

    # TODO: ADAPT FOR mlcstar
    # Update this dict to map your cfg['concepts'] names to filter functions.
    # Add new filter functions above as needed.
    """
    filter_funcs = {
        # 'VitaleVaerdier': filter_vitals,
        # 'Labsvar': filter_labs,
        # 'ITAOversigtsrapport': filter_ita,
        # 'Medicin': filter_medicin,
        # 'Procedurer': filter_procedures,
        # 'ADTHaendelser': filter_adt,
    }

    if concept not in filter_funcs:
        raise KeyError(
            f"No filter function registered for concept '{concept}'. "
            f"Add it to collect_filter() in filters.py."
        )
    return filter_funcs[concept]


if __name__ == "__main__":
    pass
