"""
Shared clinical data mappings for mlcstar.

Single source of truth for all feature name mappings, code-to-category
lookups, and clinical classification logic.

No heavy dependencies (no mlcstar.utils, no azureml) so this module can
be imported anywhere.

# TODO: ADAPT FOR mlcstar
# This file is a template derived from the astra project.
# Replace all mapping dicts and patterns with values relevant to your
# clinical domain. Keep the structure (dict names, utility functions)
# the same so that filters.py and build_patient_info.py continue to work.
"""

import re
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union


# ============================================================================
# Vital signs: raw parameter names → standardized feature names
# (used by filter_vitals in filters.py)
# ============================================================================
# TODO: FILL IN FOR mlcstar — map your EHR's vital sign parameter names to
# standardized names (e.g. 'HR', 'SBP', 'DBP', 'MAP', 'SPO2', 'TEMP',
# 'RESPIRATORYRATE'). Add or remove entries as needed.

VITALS_MAP: Dict[str, str] = {
    # 'RawParameterName': 'STANDARDIZED_NAME',
}

# Blood pressure parameter names that carry "sys/dia" strings to split
# TODO: FILL IN FOR mlcstar — list any BP parameter names that contain
# combined systolic/diastolic strings (e.g. "120/80") so they get split
# into separate SBP and DBP measurements.
BP_TYPES: List[str] = []

# Height/weight parameter names → standardized
# TODO: FILL IN FOR mlcstar
HEIGHT_WEIGHT_MAP: Dict[str, str] = {
    # 'RawHeightParam': 'HEIGHT',
    # 'RawWeightParam': 'WEIGHT',
}


# ============================================================================
# Lab tests: raw test names → standardized feature names
# (used by filter_labs in filters.py)
# ============================================================================
# TODO: FILL IN FOR mlcstar — map standardized lab names to tuples of raw
# test name strings as they appear in your data source.
# Format: 'STANDARDIZED_NAME': ('raw_name_1', 'raw_name_2', ...)

LABS_FEATURE_MAP: Dict[str, Tuple[str, ...]] = {
    # 'LACTATE': ('LAC;P(AB)', 'LAKTAT;P'),
    # 'HEMOGLOBIN': ('HGB;B',),
}

# Reverse lookup: raw test name → standardized name (auto-built)
LABS_REVERSE_MAP: Dict[str, str] = {}
for _std_name, _raw_names in LABS_FEATURE_MAP.items():
    for _raw in _raw_names:
        LABS_REVERSE_MAP[_raw] = _std_name


# ============================================================================
# ICU / ward scores: measurement names → standardized feature names
# (used by filter_ita in filters.py, if applicable)
# ============================================================================
# TODO: FILL IN FOR mlcstar — map your ICU/ward score parameter names to
# standardized names. Remove this section if not applicable.

ICU_MAP: Dict[str, str] = {
    # 'GLASGOW COMA SCORE': 'GCS',
    # 'SOFA total score': 'SOFA',
}


# ============================================================================
# Medications: ATC code prefixes → category names
# (used by filter_medicin in filters.py)
# ============================================================================
# TODO: FILL IN FOR mlcstar — define ATC level 3 (3-char prefix) and
# level 4 (4-char prefix) groupings relevant to your clinical question.
# Keys are category names; values are lists of ATC prefixes.

ATC_LVL3_MAP: Dict[str, List[str]] = {
    # 'antibiotics': ['J01'],
    # 'cardiovascular_drugs': ['C01', 'C02', 'C07'],
}

ATC_LVL4_MAP: Dict[str, List[str]] = {
    # 'opiods': ['N02A'],
    # 'infusion': ['B05B', 'B05X'],
}

# Reverse lookups (auto-built)
ATC_LVL3_REVERSE: Dict[str, str] = {}
for _cat, _codes in ATC_LVL3_MAP.items():
    for _code in _codes:
        ATC_LVL3_REVERSE[_code] = _cat

ATC_LVL4_REVERSE: Dict[str, str] = {}
for _cat, _codes in ATC_LVL4_MAP.items():
    for _code in _codes:
        ATC_LVL4_REVERSE[_code] = _cat

# Valid medication administration actions
# TODO: FILL IN FOR mlcstar — list the action types in your medication data
# that represent actual administration (not orders, cancellations, etc.)
MEDICATION_ACTION_LIST: List[str] = [
    # 'Administered',
    # 'Given',
]


# ============================================================================
# Procedures: procedure codes → surgical category names
# (used by filter_procedures in filters.py)
# ============================================================================
# TODO: FILL IN FOR mlcstar — map procedure code groups to category names.
# Keys are category names; values are tuples of procedure code strings.

PROCEDURE_MAP: Dict[str, Tuple[str, ...]] = {
    # 'major_abdominal': ('CODE1', 'CODE2'),
    # 'major_thoracic': ('CODE3',),
}

# Reverse lookup: procedure code → category name (auto-built)
PROCEDURE_REVERSE_MAP: Dict[str, str] = {}
for _cat, _codes in PROCEDURE_MAP.items():
    for _code in _codes:
        PROCEDURE_REVERSE_MAP[_code] = _cat

# Flat include list of all procedure codes (auto-built)
PROCEDURE_INCLUDE_LIST: List[str] = [
    code for codes in PROCEDURE_MAP.values() for code in codes
]


# ============================================================================
# ADT (department) classification patterns
# (used by filter_adt in filters.py)
#
# Each entry: (location_type, list_of_patterns)
# A pattern is either:
#   - a string: simple regex match (case-insensitive)
#   - a tuple of strings: compound AND match (all patterns must match)
# ============================================================================
# TODO: FILL IN FOR mlcstar — define department classification rules using
# regex patterns that match department names in your ADT data.
# Common location types: 'TC' (trauma center), 'OR' (operating room),
# 'ICU' (intensive care), 'BED' (ward bed), 'AMB' (ambulatory/outpatient)

ADT_PATTERNS: List[Tuple[str, list]] = [
    # ('ICU', [r'\bintensive\b', r'\bICU\b']),
    # ('OR', ['operating room', 'surgery']),
    # ('BED', ['ward', 'bed']),
]


# ============================================================================
# Hospital name standardization
# (used by standardize_hospital, first_hospital in build_patient_info.py)
# ============================================================================
# TODO: FILL IN FOR mlcstar — list canonical hospital identifiers for your
# data. These are matched against extracted hospital name tokens.

VALID_HOSPITALS: List[str] = [
    # 'HOSPITAL_A',
    # 'HOSPITAL_B',
]

# Sex value normalization
# TODO: ADAPT FOR mlcstar if your data uses different sex value strings
SEX_MAP: Dict[str, str] = {
    'Male': 'Male',
    'Female': 'Female',
    'M': 'Male',
    'F': 'Female',
}


# ============================================================================
# Utility functions (pure — no file I/O, no heavy deps)
# ============================================================================

def classify_department(dept_name: str) -> Optional[str]:
    """
    Classify a department name into a location type.

    Uses ADT_PATTERNS. Supports both simple regex patterns and compound
    (tuple) patterns where all sub-patterns must match.

    Returns one of the location type strings from ADT_PATTERNS, or None.
    """
    if pd.isna(dept_name) or not dept_name:
        return None
    for location_type, patterns in ADT_PATTERNS:
        for pattern in patterns:
            if isinstance(pattern, tuple):
                if all(re.search(p, dept_name, re.IGNORECASE) for p in pattern):
                    return location_type
            else:
                if re.search(pattern, dept_name, re.IGNORECASE):
                    return location_type
    return None


def first_hospital(name: str) -> str:
    """
    Extract hospital identifier from a department name.

    Takes the first word, or "SJ <second_word>" for multi-word hospital
    identifiers (adapt if your naming convention differs).
    """
    if pd.isna(name):
        return name
    words = str(name).strip().split()
    if words and words[0] == 'SJ':  # TODO: adapt prefix logic for mlcstar
        return ' '.join(words[:2])
    elif words:
        return words[0]
    return ''


def standardize_hospital(
    name: str,
    valid_hospitals: List[str] = VALID_HOSPITALS,
) -> str:
    """
    Standardize a hospital name to a canonical form.

    Returns 'MISC' for unrecognized names.
    TODO: ADAPT FOR mlcstar — add any partial match logic needed for your
    hospital naming conventions.
    """
    if pd.isna(name):
        return np.nan
    name = str(name).strip().upper()

    for h in valid_hospitals:
        if name == h.upper():
            return h

    return 'MISC'


def derive_first_hospital(dept_name: str) -> str:
    """Extract and standardize hospital name from a department name.

    Convenience wrapper: first_hospital() → standardize_hospital().
    """
    if pd.isna(dept_name) or not dept_name:
        return np.nan
    cleaned = str(dept_name).strip().replace(',', '')
    extracted = first_hospital(cleaned)
    return standardize_hospital(extracted)


def parse_numeric(value_str: str) -> Optional[float]:
    """Parse a numeric value from a string, stripping <, > prefixes."""
    if not value_str:
        return None
    cleaned = re.sub(r'^[<>]\s*', '', str(value_str).strip())
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def classify_atc(atc_code: str) -> Optional[str]:
    """Map an ATC code to a medication category.

    Tries level 3 first (first 3 chars), then level 4 (first 4 chars).
    Returns None if no match.
    """
    if not atc_code:
        return None
    atc = str(atc_code)
    cat = ATC_LVL3_REVERSE.get(atc[:3])
    if cat is None:
        cat = ATC_LVL4_REVERSE.get(atc[:4])
    return cat
