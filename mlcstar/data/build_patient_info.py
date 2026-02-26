import pandas as pd
import numpy as np
import subprocess

from sklearn import base

from mlcstar.utils import logger, get_cfg, get_base_df, create_enumerated_id, is_file_present
from mlcstar.utils import ensure_datetime, count_csv_rows, inches_to_cm, ounces_to_kg
try:
    from mlcstar.data.collectors import population_filter_parquet
except ImportError:
    population_filter_parquet = None
from mlcstar.data.mapper import map_concept
from mlcstar.data.mappings import (
    standardize_hospital as _standardize_hospital,
    first_hospital as _first_hospital,
)

from typing import List, Dict, Optional, Union
from datetime import timedelta

try:
    from azureml.core import Dataset
except ImportError:
    Dataset = None


# ============================================================================
# MAIN PIPELINE ENTRY POINT
# ============================================================================

def create_base_df(cfg, result_path=None):
    """
    Create the base patient cohort dataframe.

    # TODO: ADAPT FOR mlcstar
    # Replace each step below with your domain's equivalent.
    # The output must contain at minimum:
    #   - PID: int — unique patient identifier
    #   - CPR_hash (or equivalent ID): str
    #   - ServiceDate (or equivalent index date): datetime
    #   - start: datetime — trajectory/episode start time
    #   - end: datetime — trajectory/episode end time
    #   - <target>: int — your prediction target (0/1)
    #   - Any static features configured in cfg["dataset"]["num_cols"] and
    #     cfg["dataset"]["cat_cols"]
    """
    if result_path is None:
        result_path = cfg["base_df_path"]
    logger.info("Creating base dataframe")

    population = load_or_collect_population(cfg)
    logger.info(f'pop loaded: {population.CPR_hash.nunique()} unique CPR hashes')
    df_ad = load_or_collect_adt(population)
    of = build_trajectories(df_ad)

    population = ensure_datetime(population, "ServiceDate")
    matched = match_population_to_trajectories(of, population)
    # For patients without a trajectory, fall back to ServiceDate for start/end
    matched["start"] = matched["start"].fillna(matched["ServiceDate"])
    matched["end"] = matched["end"].fillna(matched["ServiceDate"])
    logger.info(f'matched: {matched.CPR_hash.nunique()} unique CPR hashes')

    merged_df = add_first_contacts(matched, df_ad)
    merged_df = add_first_hospital(merged_df)

    # Add case info
    merged_df = add_case_knife_time(merged_df, population)

    result = add_patient_info(merged_df, population)
    result = add_patient_id(result)
    #result = mask_mortality(result)
    logger.info(f'before clean: {result.CPR_hash.nunique()} unique CPR hashes')
    result = final_cleanup(result)
    logger.info(f'after clean: {result.CPR_hash.nunique()} unique CPR hashes')
    # Add static features
    result = add_to_base(result)
    logger.info(f'after static added: {result.CPR_hash.nunique()} unique CPR hashes')
    # Add comorbidity (optional — remove if not applicable)
    result = add_comorbidity(result)

    logger.info(f"Saving file at {result_path}")
    result.to_pickle(result_path, protocol=4)
    return result


# ============================================================================
# POPULATION
# ============================================================================

def load_or_collect_population(cfg):
    """
    Load or collect the patient population.
    """
    while True:
        try:
            return pd.read_csv(cfg["population_file_path"], index_col=0)
        except FileNotFoundError:
            logger.warning("Population seed file not found! Creating")
            
            from azureml.core import Dataset
            path = f'{cfg["raw_file_path"]}CPMI_Procedurer.parquet'
            ds_procedure = Dataset.Tabular.from_parquet_files(path=path)
            df_procedure = ds_procedure.to_pandas_dataframe()
            df_procedure['ProcedureCode'] = df_procedure['ProcedureCode'].astype('category')
            df_procedure = df_procedure[df_procedure.ProcedureCode.str.startswith('KJJ')]
            df_procedure.to_csv(cfg["population_file_path"])
            raise


# ============================================================================
# ADT / ADMISSION DATA
# ============================================================================

def load_or_collect_adt(population):
    """
    Load or collect admission/discharge/transfer (ADT) events.

    # TODO: ADAPT FOR mlcstar
    # Replace column names and file path with your domain's equivalent.
    # Required output columns: CPR_hash, Flyt_ind (start), Flyt_ud (end),
    #   ADT_haendelse (event type), Afsnit (department/unit name)
    """
    path = "data/raw/ADTHaendelser.csv"   # TODO: update filename
    while True:
        try:
            logger.debug("Loading ADT")
            df_ad = pd.read_csv(path, dtype={"CPR_hash": str}, index_col=0)
            break
        except FileNotFoundError:
            logger.warning("ADT file not found. Loading.")
            if population_filter_parquet:
                population_filter_parquet("ADTHaendelser", base=population)  # TODO: update filename
            else:
                raise

    df_ad[["Flyt_ind", "Flyt_ud"]] = df_ad[["Flyt_ind", "Flyt_ud"]].apply(
        pd.to_datetime, format="mixed", errors="coerce"
    )
    # TODO: adapt shift logic to your event type column name
    df_ad.loc[df_ad.ADT_haendelse == "Flyt Ind", "Flyt_ind"] += pd.Timedelta(seconds=1)

    return df_ad.sort_values(["CPR_hash", "Flyt_ind"]).reset_index(drop=True)


# ============================================================================
# TRAJECTORY CONSTRUCTION
# ============================================================================

def build_trajectories(df_ad):
    logger.info(">Building trajectories")
    df_ad["trajectory"] = (
        df_ad[df_ad["ADT_haendelse"] == "Indlæggelse"]
        .groupby("CPR_hash")
        .cumcount() + 1
    )
    df_ad["trajectory"] = df_ad["trajectory"].ffill()

    df_ad = df_ad[["CPR_hash", "trajectory", "Flyt_ind", "Flyt_ud"]].copy()
    df_ad = df_ad.rename(columns={"Flyt_ind": "start", "Flyt_ud": "end"})

    of = collapse_admissions(df_ad, time_gap_hours=1)

    return of


def match_population_to_trajectories(of, population):
    logger.info("Matching procedures to trajectories.")
    fdf = find_forløb(of, population, "ServiceDate")
    fdf.to_csv('data/processed/admissions.csv')

    # Pivot multiple procedures per trajectory into wide columns
    group_key = ["CPR_hash", "trajectory"]
    proc_cols = [c for c in fdf.columns if c.startswith("Procedure")]
    fdf = fdf.sort_values(group_key + ["ServiceDate"])
    fdf["_proc_num"] = fdf.groupby(group_key).cumcount() + 1
    max_procs = int(fdf["_proc_num"].max())
    logger.info(f"  Max procedures per trajectory: {max_procs}")

    # Pivot procedure-specific columns wide
    wide_parts = []
    for col in proc_cols:
        pivoted = fdf.pivot_table(
            index=group_key, columns="_proc_num", values=col, aggfunc="first"
        )
        pivoted.columns = [f"{col}_{int(n)}" for n in pivoted.columns]
        wide_parts.append(pivoted)

    if wide_parts:
        wide = pd.concat(wide_parts, axis=1).reset_index()
    else:
        wide = fdf[group_key].drop_duplicates().reset_index(drop=True)

    # Keep non-procedure columns from first row per group (earliest ServiceDate)
    non_proc = (
        fdf.drop(columns=proc_cols + ["_proc_num"])
        .drop_duplicates(subset=group_key, keep="first")
    )

    df = non_proc.merge(wide, on=group_key, how="left")

    # Merge with trajectory data for start/end
    df = pd.merge(
        df, of, on=["CPR_hash", "trajectory"],
        how="left", suffixes=("", "_traj"),
    )
    df = df.drop(columns=[c for c in df.columns if c.endswith("_traj")])

    # Preserve population patients that had no matching trajectory
    pop_cols = ["CPR_hash", "ServiceDate"]
    unmatched = population.merge(
        df[pop_cols].drop_duplicates(), on=pop_cols,
        how="left", indicator=True
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"].drop(columns=["_merge"])
    if len(unmatched) > 0:
        # Rename procedure columns to _1 for consistency with wide format
        for col in proc_cols:
            if col in unmatched.columns:
                unmatched = unmatched.rename(columns={col: f"{col}_1"})
        logger.info(
            f"  {unmatched.CPR_hash.nunique()} patients had no trajectory match, "
            "keeping with NaN trajectory"
        )
        df = pd.concat([df, unmatched], ignore_index=True)
    return df


# ============================================================================
# CARE PATHWAY / CONTACT ENRICHMENT
# ============================================================================

def add_first_contacts(df, df_adt):
    """
    Find first contacts within each trajectory.

    # TODO: ADAPT FOR mlcstar
    # Replace "RH " with your trauma center / site identifier.
    # Remove or replace "type_visitation" if your domain doesn't use it.
    """
    logger.info("Finding first contact and site.")
    merged = df_adt.merge(df[["CPR_hash", "ServiceDate", "start", "end"]], on="CPR_hash")
    filtered = merged[(merged["Flyt_ind"] >= merged["start"]) & (merged["Flyt_ind"] <= merged["end"])]

    first_afsnit = filtered.groupby(["CPR_hash", "ServiceDate", "start"]).first().reset_index()

    # TODO: replace "RH " with your site identifier
    first_site = filtered[
        filtered["Afsnit"].str.contains("RH ", case=False, na=False)
    ].groupby(["CPR_hash", "ServiceDate", "start"]).first().reset_index()

    first_site = first_site[["CPR_hash", "Flyt_ind", "ServiceDate", "start"]].rename(columns={"Flyt_ind": "first_RH"})

    enriched = pd.merge(first_afsnit, first_site, on=["CPR_hash", "ServiceDate", "start"], how="left")
    enriched = enriched.rename(columns={"Afsnit": "first_afsnit"})
    enriched["time_to_RH"] = enriched["first_RH"] - enriched["start"]

    # Left-join back to input df so patients without ADT events are preserved
    cols_to_add = [c for c in enriched.columns if c not in df.columns or c in ["CPR_hash", "ServiceDate", "start"]]
    result = df.merge(enriched[cols_to_add], on=["CPR_hash", "ServiceDate", "start"], how="left")
    logger.info(f"first_contacts: {result.CPR_hash.nunique()} unique CPR hashes ({enriched.CPR_hash.nunique()} had ADT match)")

    return result


# Delegated to mlcstar.data.mappings (single source of truth)
standardize_hospital = _standardize_hospital
first_hospital = _first_hospital


def add_first_hospital(df):
    """
    Extract and standardize the first hospital from department name.

    # TODO: ADAPT FOR mlcstar
    # Update hospital classification to match your site names.
    """
    df['first_afsnit'] = df['first_afsnit'].str.replace(',', '', regex=False)
    df['FIRST_HOSPITAL'] = df['first_afsnit'].apply(first_hospital)
    df['FIRST_HOSPITAL'] = df['FIRST_HOSPITAL'].apply(standardize_hospital)
    print(df.FIRST_HOSPITAL.value_counts())
    return df


def add_case_knife_time(df, procedures):
    """
    Add 'knife_time' from the Cases file — the Knivtid/procedure start timestamp.

    Matches Cases to the population seed (Procedurer) on:
      - CPR_hash (both sides)
      - SKS_Kode (Cases) == ProcedureCode (population)
      - Operationshændelses_Tidspunkt date (Cases) == ServiceDate date (population)
    filtered to rows where Operationshændelse == "Knivtid/procedure start".
    """
    logger.info("Adding knife time from Cases.")
    from mlcstar.data.collectors import population_filter_parquet
    population_filter_parquet('Cases', base=df)
    cases = pd.read_csv("data/raw/Cases.csv", index_col=0, dtype={"CPR_hash": str})
    cases["Operationshændelses_Tidspunkt"] = pd.to_datetime(
        cases["Operationshændelses_Tidspunkt"], errors="coerce"
    )

    knife_start = (
        cases[(cases.Operationshændelse == "Knivtid/procedure start") & (cases.Status == "Fuldført")]
        [["CPR_hash", "SKS_Kode", "Case_ID", "Operationshændelses_Tidspunkt"]]
        .rename(columns={"Operationshændelses_Tidspunkt": "knife_time"})
    )
    knife_end = (
        cases[cases.Operationshændelse == "Sidste sutur/procedure slut"]
        [["Case_ID", "Operationshændelses_Tidspunkt"]]
        .rename(columns={"Operationshændelses_Tidspunkt": "knife_time_end"})
    )
    knife = knife_start.merge(knife_end, on="Case_ID", how="left")
    knife["elapsed_knife_time_minutes"] = knife["knife_time_end"] - knife["knife_time"]
    knife["elapsed_knife_time_minutes"] = knife["elapsed_knife_time_minutes"].dt.total_seconds() / 60

    pop = procedures[["CPR_hash", "ProcedureCode", "ServiceDate"]].copy()
    pop["ServiceDate"] = pd.to_datetime(pop["ServiceDate"], errors="coerce")

    candidates = knife.merge(
        pop,
        left_on=["CPR_hash", ],
        right_on=["CPR_hash", ],
        how="inner",
    )
    candidates["_diff"] = (candidates["knife_time"] - candidates["ServiceDate"]).abs()
    candidates = candidates[candidates["_diff"] <= pd.Timedelta(days=1)]

    matched = (
        candidates.sort_values("_diff")
        .drop_duplicates(subset=["CPR_hash", "ServiceDate"])
        
    )
    matched.to_csv('data/processed/ProcedurerCases_merged.csv')
    logger.info(f"Matched knife_time for {len(matched)} of {len(df)} rows.")
    matched = matched[["CPR_hash", "ServiceDate", "Case_ID", "knife_time", "knife_time_end", "elapsed_knife_time_minutes"]]
    return df.merge(matched, on=["CPR_hash", "ServiceDate"], how="left")


# ============================================================================
# PATIENT DEMOGRAPHICS
# ============================================================================

def add_patient_info(df, population):
    """
    Add patient demographics (DOB, DOD, SEX).

    # TODO: ADAPT FOR mlcstar
    # Update column names and CSV file path. If your domain uses different
    # demographic fields, update accordingly.
    """
    logger.info("Adding patient information.")
    if population_filter_parquet:
        population_filter_parquet("PatientInfo", base=population)  # TODO: update filename
    pi = pd.read_csv("data/raw/PatientInfo.csv", index_col=0)  # TODO: update filename
    # TODO: update column rename mapping
    pi = pi.rename(columns={"Fødselsdato": "DOB", "Dødsdato": "DOD", "Køn": "SEX"})
    pi["SEX"] = pi["SEX"].replace({"Mand": "Male", "Kvinde": "Female"})

    df = df.merge(pi[["CPR_hash", "DOB", "DOD", "SEX"]], on="CPR_hash", how="left")
    df["overlap"] = df.groupby("CPR_hash", group_keys=False).apply(check_overlaps).explode().values

    return df


def add_patient_id(df):
    logger.info("Creating PID.")
    return create_enumerated_id(df, "CPR_hash", "ServiceDate")


def final_cleanup(df):
    logger.info("Cleaning up dataframe.")
    #df = df[df["start"].notnull() & df["end"].notnull()]
    df = df.drop(columns=["Flyt_ind", "Flyt_ud", "ADT_haendelse"], errors='ignore')
    df = df.drop_duplicates(subset="PID").reset_index(drop=True)
    df = df.drop_duplicates(subset=["CPR_hash", "start", "end"]).reset_index(drop=True)
    return df


# ============================================================================
# STATIC FEATURES
# ============================================================================

def add_to_base(base):
    """
    Add static features to the base dataframe.

    # TODO: ADAPT FOR mlcstar
    # Replace mortality flags (deceased_30d, deceased_90d) and LVL1TC with
    # your target variable and domain-specific binary indicators.
    # Update AGE, HEIGHT, WEIGHT as appropriate.
    """
    base["DURATION"] = (base.end - base.start) / np.timedelta64(1, "D")

    base["AGE"] = (
        np.floor(
            (pd.to_datetime(base["start"]) - pd.to_datetime(base.DOB)).dt.days / 365.25
        )
    ).astype("Int64")

    base = add_height_weight(base)

    base.loc[
        (pd.to_datetime(base.DOD) - pd.to_datetime(base.start))
        <= pd.Timedelta(days=30),
        "DECEASED_30d",
    ] = 1
    base["DECEASED_30d"] = base["DECEASED_30d"].fillna(0)

    base.loc[
        (pd.to_datetime(base.DOD) - pd.to_datetime(base.start))
        <= pd.Timedelta(days=90),
        "DECEASED_90d",
    ] = 1
    base["DECEASED_90d"] = base["DECEASED_90d"].fillna(0)


    return base


### Height weight

def prepare_height_weight(base):

    path = "data/raw/VitaleVaerdier.csv"
    if is_file_present(path):
        logger.info("Vitals file found, loading.")
    else:
        logger.info("Vitals file not found, processing.")
        from mlcstar.data.collectors import population_filter_parquet
        population_filter_parquet('VitaleVaerdier', base=base)

    vit_raw = pd.read_csv(path) 
    hw_map = {"Højde": "HEIGHT", "Vægt": "WEIGHT"}
    vit_raw.rename(
        columns={
            "Værdi": "VALUE",
            "Vital_parametre": "FEATURE",
            "Registreringstidspunkt": "TIMESTAMP",
        },
        inplace=True,
    )
    vit_raw["FEATURE"] = vit_raw["FEATURE"].replace(to_replace=hw_map)
    vit_raw["VALUE"] = pd.to_numeric(vit_raw["VALUE"], errors="coerce")
    vit_raw = vit_raw.dropna(subset=["VALUE"])
    vit_raw.loc[vit_raw.FEATURE == "HEIGHT", "VALUE"] = inches_to_cm(
        vit_raw[vit_raw.FEATURE == "HEIGHT"].VALUE.astype(float)
    )
    vit_raw.loc[vit_raw.FEATURE == "WEIGHT", "VALUE"] = ounces_to_kg(
        vit_raw[vit_raw.FEATURE == "WEIGHT"].VALUE.astype(float)
    )
    hw = vit_raw[(vit_raw.FEATURE.isin(list(set(hw_map.values()))))]
    assert len(hw)>0
    hw = hw.merge(base[["PID", "CPR_hash", "start", "end"]], on="CPR_hash", how="left")
    hw["TIMESTAMP"] = pd.to_datetime(hw.TIMESTAMP)
    hw = hw[hw.TIMESTAMP <= hw.end]
    hw = hw.sort_values(["CPR_hash", "TIMESTAMP"], ascending=False).drop_duplicates(
        subset=["CPR_hash", "FEATURE"], keep="first"
    )
    #hw = hw[hw.delta.dt.days < 365 * 2]
    return hw[["TIMESTAMP", "PID", "FEATURE", "VALUE"]]


def add_height_weight(base):

    hw= prepare_height_weight(base)

    hw_df = hw.sort_values("TIMESTAMP").drop_duplicates(
        subset=["PID", "FEATURE"], keep="first"
    )
    pivot_df = hw_df.pivot(
        index=["PID"], columns="FEATURE", values="VALUE"
    ).reset_index()
    base = base.merge(pivot_df, how="left", on="PID")

    return base



# ============================================================================
# COMORBIDITY (OPTIONAL)
# ============================================================================

def add_comorbidity(base, cols_to_add=["ASMT_ELIX"]):
    while True:
        try:
            elix = pd.read_csv("data/interim/computed_elix_df.csv", low_memory=False)
            logger.info("Elixhauser df dataframe found, continuing")
            baselen = len(base)
            elix = elix.rename(columns={'elixscore': 'ASMT_ELIX'})
            base = base.merge(elix[["PID"] + cols_to_add], how="left", on="PID")
            assert baselen - len(base) == 0
            logger.info("Merged Elix onto base")
            return base
        except FileNotFoundError:
            logger.info("Elixhauser DF missing. Create it or remove this call.")
            base["ASMT_ELIX"] = np.nan
            return base
        break


# ============================================================================
# DOMAIN-AGNOSTIC UTILITY FUNCTIONS
# ============================================================================

def collapse_admissions(df: pd.DataFrame, time_gap_hours: int = 1) -> pd.DataFrame:
    """Collapse consecutive admissions within a time gap threshold."""
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(["CPR_hash", "start"]).reset_index(drop=True)

    collapsed = df.groupby("CPR_hash").apply(
        lambda group: _collapse_patient_admissions(group, time_gap_hours),
        include_groups=True
    ).reset_index(drop=True)

    return collapsed


def _collapse_patient_admissions(group: pd.DataFrame, time_gap_hours: int) -> pd.DataFrame:
    group = group.sort_values("start").copy()
    group["prev_end"] = group["end"].shift()
    group["gap"] = group["start"] - group["prev_end"]

    gap_thresh = pd.Timedelta(hours=time_gap_hours)
    group["group_id"] = (group["gap"] >= gap_thresh).cumsum()

    collapsed = (
        group.groupby("group_id")
        .agg({
            "CPR_hash": "first",
            "start": "min",
            "end": "max",
            "trajectory": lambda x: ",".join(sorted(set(str(int(v)) for v in x if pd.notna(v)))),
        })
        .reset_index(drop=True)
    )
    collapsed["duration"] = collapsed["end"] - collapsed["start"]
    return collapsed


def find_forløb(
    base: pd.DataFrame, df: pd.DataFrame, dt_name: str, offset=1
) -> pd.DataFrame:
    """
    Match observations to trajectories based on date overlap with optional offset.
    """
    colnames = df.columns.to_list()
    df = ensure_datetime(df, dt_name)
    merged_df = base.merge(df, on="CPR_hash", how="left")

    filtered_df = merged_df[
        (merged_df[dt_name] >= merged_df["start"] - pd.DateOffset(days=offset))
        & (merged_df[dt_name] <= merged_df["end"] + pd.DateOffset(days=offset))
    ]
    filtered_df = filtered_df.drop_duplicates().reset_index(drop=True)

    return filtered_df[colnames + ["trajectory"]]


def check_overlaps(group):
    """Check for overlapping trajectories for the same patient."""
    overlaps = []
    for i in range(len(group) - 1):
        if group.iloc[i]["end"] > group.iloc[i + 1]["start"]:
            overlaps.append(True)
        else:
            overlaps.append(False)
    overlaps.append(False)
    return overlaps


# ============================================================================
# BIN GENERATION (DOMAIN-AGNOSTIC)
# ============================================================================

def create_bin_df(cfg):
    """
    Generate time bins for each patient trajectory based on configurable binning intervals.

    This function is domain-agnostic. It reads base_df and creates time bins
    according to cfg["bin_intervals"]. No adaptation needed.
    """
    logger.info("Generating bin_df")
    bin_list = []
    base = get_base_df()

    bin_intervals = cfg["bin_intervals"]

    for _, row in base.iterrows():
        start_time = row["start"]
        end_time = row["end"] + pd.Timedelta(minutes=10)
        pid = row["PID"]

        current_time = start_time
        bin_counter = 1

        for interval, freq in bin_intervals.items():
            if current_time >= end_time:
                break

            if interval == "end":
                interval_end = end_time
            else:
                interval_end = start_time + pd.Timedelta(interval)

            bins = pd.date_range(
                start=current_time,
                end=min(interval_end, end_time),
                freq=freq,
                inclusive="left",
            )

            bin_list.extend(
                [
                    (pid, bin_start, bin_end, bin_counter + i, freq)
                    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:]))
                ]
            )

            current_time = bins[-1]
            bin_counter += len(bins) - 1

    bin_df = pd.DataFrame(
        bin_list, columns=["PID", "bin_start", "bin_end", "bin_counter", "bin_freq"]
    )

    bin_df.to_pickle(cfg["bin_df_path"], protocol=4)
    logger.info(f'>> Saved at {cfg["bin_df_path"]}')

    return bin_df


def create_bin_df_with_mortality_masking(cfg, base):
    """
    Create bin_df with option to drop last bin for deceased patients.
    Domain-agnostic — no adaptation needed.
    """
    logger.info("Generating bin_df with mortality masking")

    bin_list = []
    bin_intervals = cfg["bin_intervals"]

    drop_last_bin = base.get('drop_last_bin', pd.Series([False] * len(base)))

    for idx, row in base.iterrows():
        pid = row["PID"]
        start_time = row["start"]
        end_time = row["end"] + pd.Timedelta(minutes=10)
        should_drop_last = drop_last_bin.iloc[idx] if idx < len(drop_last_bin) else False

        if pd.isna(start_time) or pd.isna(end_time):
            logger.warning(f"Patient {pid} has NULL timestamps, skipping")
            continue

        if end_time <= start_time:
            logger.warning(f"Patient {pid} has invalid trajectory (end <= start), skipping")
            continue

        current_time = start_time
        bin_counter = 1
        patient_bins = []

        for interval, freq in bin_intervals.items():
            if current_time >= end_time:
                break

            if interval == "end":
                interval_end = end_time
            else:
                interval_end = start_time + pd.Timedelta(interval)

            bins = pd.date_range(
                start=current_time,
                end=min(interval_end, end_time),
                freq=freq,
                inclusive="left",
            )

            if len(bins) < 2:
                continue

            for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                patient_bins.append((pid, bin_start, bin_end, bin_counter + i, freq))

            current_time = bins[-1]
            bin_counter += len(bins) - 1

        if should_drop_last and len(patient_bins) > 1:
            logger.debug(f"Dropping last bin for deceased patient {pid}")
            patient_bins = patient_bins[:-1]

        bin_list.extend(patient_bins)

    bin_df = pd.DataFrame(
        bin_list, columns=["PID", "bin_start", "bin_end", "bin_counter", "bin_freq"]
    )

    base_pids = set(base['PID'].unique())
    bin_pids = set(bin_df['PID'].unique())
    missing = base_pids - bin_pids

    logger.info(f"Created bins for {len(bin_pids)}/{len(base_pids)} patients")

    if len(missing) > 0:
        logger.error(f"  {len(missing)} patients missing from bin_df!")

    bin_df.to_pickle(cfg["bin_df_path"])
    logger.info(f'Saved to {cfg["bin_df_path"]}')

    return bin_df


# ============================================================================
# MORTALITY MASKING (DOMAIN-AGNOSTIC)
# ============================================================================

def mask_mortality(df, method='percentage', min_duration_hours=0.5):
    """
    Adjust patient trajectory end times based on DOD and trajectory duration.
    Domain-agnostic — no adaptation needed if your base_df has a DOD column.
    Remove this call from create_base_df() if your domain doesn't use mortality.
    """
    logger.info(f"Masking mortality using method: {method}")

    for col in ["start", "end", "DOD"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    dod_mask = df["DOD"].notnull()
    n_deaths = dod_mask.sum()

    if not dod_mask.any():
        logger.info("No patients with DOD, skipping masking")
        return df

    logger.info(f"Masking {n_deaths} patients with DOD")

    duration = df["end"] - df["start"]
    duration_hours = duration.dt.total_seconds() / 3600

    original_end = df["end"].copy()

    if method == 'percentage':
        cond_short = dod_mask & (duration_hours <= 6)
        df.loc[cond_short, "end"] = df.loc[cond_short, "start"] + 0.9 * duration.loc[cond_short]

        cond_medium = dod_mask & (duration_hours > 6) & (duration_hours <= 72)
        df.loc[cond_medium, "end"] = df.loc[cond_medium, "start"] + 0.95 * duration.loc[cond_medium]

        cond_long = dod_mask & (duration_hours > 72)
        df.loc[cond_long, "end"] = df.loc[cond_long, "start"] + 0.98 * duration.loc[cond_long]

    elif method == 'absolute':
        cond1 = dod_mask & (duration_hours < 3)
        df.loc[cond1, "end"] = df.loc[cond1, "DOD"] - pd.Timedelta(minutes=10)

        cond2 = dod_mask & (duration_hours >= 3) & (duration_hours <= 72)
        df.loc[cond2, "end"] = df.loc[cond2, "DOD"] - pd.Timedelta(minutes=30)

        cond3 = dod_mask & (duration_hours > 72) & (duration_hours <= 168)
        df.loc[cond3, "end"] = df.loc[cond3, "DOD"] - pd.Timedelta(hours=3)

        cond4 = dod_mask & (duration_hours > 168)
        df.loc[cond4, "end"] = df.loc[cond4, "DOD"] - pd.Timedelta(days=1)

    elif method == 'drop_last_bin':
        df['drop_last_bin'] = dod_mask
        logger.info(f"Marked {n_deaths} patients to drop last bin during bin creation")
        return df

    else:
        raise ValueError(f"Unknown method: {method}")

    min_duration = pd.Timedelta(hours=min_duration_hours)
    invalid_mask = dod_mask & (df["end"] <= df["start"] + min_duration)
    n_invalid = invalid_mask.sum()

    if n_invalid > 0:
        logger.warning(f"  {n_invalid} patients would have end <= start after masking!")
        df.loc[invalid_mask, "end"] = df.loc[invalid_mask, "start"] + min_duration

    time_removed = (original_end - df["end"]).loc[dod_mask]
    avg_removed_hours = time_removed.dt.total_seconds().mean() / 3600

    logger.info(f"Masking complete: {n_deaths} patients masked, avg {avg_removed_hours:.1f}h removed")

    return df


if __name__ == "__main__":
    cfg = get_cfg()
    create_base_df(cfg)
    create_bin_df(cfg)
