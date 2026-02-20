import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import warnings

from mlcstar.utils import logger, get_bin_df
from mlcstar.data.filters import collect_filter


class AggregatedDS:
    """
    High-performance dataset class that aggregates time series data into tabular format.

    Optimizations:
    - Vectorized operations (no patient loops)
    - GPU acceleration with cuDF/CuPy (if available)
    - Efficient memory management
    - Parallel aggregations

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing dataset parameters
    base_df : pd.DataFrame
        Base dataframe containing patient IDs, target, and baseline features
    masking_point : str or pd.Timedelta, optional
        Time offset from patient start time to mask data
    agg_funcs : list of str, optional
        Aggregation functions to apply
    concepts : list of str, optional
        Concept names to aggregate
    use_gpu : bool, optional
        Whether to use GPU acceleration if available. Default: True
    default_mode : bool, optional
        If True, automatically loads and aggregates concepts. Default: True
    """

    def __init__(
        self,
        cfg: dict,
        base_df: pd.DataFrame,
        masking_point: Optional[Union[str, pd.Timedelta]] = None,
        agg_funcs: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        use_gpu: bool = True,
        default_mode: bool = True,
    ):
        self.cfg = cfg
        self.target = cfg["target"]
        # reorder by date for temporal split
        self.base = base_df.sort_values('start').reset_index(drop=True).copy(deep=True)
        self.masking_point = masking_point

        # Try to import GPU libraries
        try:
            import cudf
            import cupy as cp
            GPU_AVAILABLE = True
            logger.info("GPU support available (cuDF/CuPy)")
        except ImportError:
            GPU_AVAILABLE = False
            logger.info("GPU support not available, using CPU-only optimizations")

        self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            logger.info("Using GPU acceleration")

        # Default aggregation functions
        if agg_funcs is None:
            self.agg_funcs = ['first', 'last', 'min', 'max', 'mean', 'std']
        else:
            self.agg_funcs = agg_funcs

        # Default concepts
        if concepts is None:
            self.concepts = self.cfg["concepts"]
        else:
            self.concepts = concepts

        # Track feature types
        self.continuous_features = []
        self.categorical_features = []

        # Get categorical configuration
        self.cat_concepts = self.cfg["dataset"]["ts_cat_names"]
        self.multi_label_concepts = self.cfg["dataset"]["ts_categorical_multi_label"]

        # Cache PID set once for fast lookups
        self._base_pids = set(self.base['PID'].unique())

        if default_mode:
            self.set_tab_df()
            self.collect_and_aggregate_concepts()
            self.create_final_dataset()

    def set_tab_df(self):
        """Initialize the base tabular dataframe."""
        id_col = self.cfg["dataset"]["id_col"]
        num_cols = self.cfg["dataset"]["num_cols"]
        cat_cols = self.cfg["dataset"]["cat_cols"]

        self.tab_df = self.base[[id_col, self.target] + num_cols + cat_cols].copy()
        self.tab_df[num_cols] = self.tab_df[num_cols].astype(float)

        self.continuous_features.extend(num_cols)
        self.categorical_features.extend(cat_cols)

        logger.debug(f"Base tabular columns: {self.tab_df.columns.tolist()}")

    def _parse_masking_point(self) -> Optional[pd.Timedelta]:
        """Convert masking point to pd.Timedelta."""
        if self.masking_point is None:
            return None
        if isinstance(self.masking_point, pd.Timedelta):
            return self.masking_point
        if isinstance(self.masking_point, str):
            return pd.Timedelta(self.masking_point)
        raise ValueError(f"Invalid masking_point type: {type(self.masking_point)}")

    def collect_and_aggregate_concepts(self):
        """Collect, filter, mask, and aggregate all concepts."""
        self.aggregated_concepts = {}
        masking_delta = self._parse_masking_point()

        for concept in self.concepts:
            logger.info(f"Processing concept: {concept}")

            try:
                # Load and filter
                concept_data = self._load_and_filter_concept(concept)

                if len(concept_data) == 0:
                    logger.warning(f"No data for {concept}")
                    continue

                # Apply masking (vectorized)
                if masking_delta is not None:
                    concept_data = self._apply_masking_vectorized(concept_data, masking_delta)
                    logger.debug(f"After masking: {len(concept_data)} rows")

                if len(concept_data) == 0:
                    logger.warning(f"No data after masking for {concept}")
                    continue

                # Determine if categorical
                is_categorical = concept in self.cat_concepts
                is_multi_label = concept in self.multi_label_concepts

                # Aggregate
                aggregated_df = self._aggregate_concept_optimized(
                    concept_data, concept, is_categorical, is_multi_label
                )

                self.aggregated_concepts[concept] = aggregated_df
                logger.info(f"Aggregated {concept}: {aggregated_df.shape}")

            except Exception as e:
                logger.error(f"Failed to process {concept}: {e}")
                continue

    def _load_and_filter_concept(self, concept: str) -> pd.DataFrame:
        """
        Load concept and apply filter function.

        Optimized: Filter to relevant PIDs BEFORE applying expensive
        concept-specific filters (regex, string ops, etc.).
        """
        concept_path = f"data/interim/concepts/{concept}.pkl"
        try:
            df = pd.read_pickle(concept_path)
        except FileNotFoundError:
            concept_path = f"data/interim/concepts/{concept}.csv"
            df = pd.read_csv(concept_path, low_memory=False)

        # Filter to PIDs in base FIRST (cheap operation)
        if 'PID' in df.columns:
            df = df[df['PID'].isin(self._base_pids)]

        # Apply concept-specific filter
        filter_function = collect_filter(concept)
        filtered_df = filter_function(df)

        # Ensure TIMESTAMP is datetime
        if 'TIMESTAMP' in filtered_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(filtered_df['TIMESTAMP']):
                filtered_df['TIMESTAMP'] = pd.to_datetime(filtered_df['TIMESTAMP'])

        logger.debug(f"Loaded & filtered {concept}: {len(filtered_df)} rows")
        return filtered_df[['PID', 'FEATURE', 'VALUE', 'TIMESTAMP']]

    def _apply_masking_vectorized(
        self,
        concept_data: pd.DataFrame,
        masking_delta: pd.Timedelta
    ) -> pd.DataFrame:
        """
        Apply masking using vectorized operations (NO LOOPS).

        This is the key optimization: instead of looping over patients,
        we merge start times and filter in one vectorized operation.
        """
        if 'start' not in self.base.columns:
            logger.warning("No 'start' column, using min timestamp per patient")
            patient_starts = concept_data.groupby('PID')['TIMESTAMP'].min().reset_index()
            patient_starts.columns = ['PID', 'start']
        else:
            patient_starts = self.base[['PID', 'start']].copy()
            if not pd.api.types.is_datetime64_any_dtype(patient_starts['start']):
                patient_starts['start'] = pd.to_datetime(patient_starts['start'])

        # Merge start times onto concept data (vectorized)
        with_starts = concept_data.merge(patient_starts, on='PID', how='left')

        # Calculate cutoff time (vectorized)
        with_starts['cutoff'] = with_starts['start'] + masking_delta

        # Filter in one vectorized operation
        masked = with_starts[with_starts['TIMESTAMP'] <= with_starts['cutoff']].copy()

        # Drop helper columns
        masked = masked[['PID', 'FEATURE', 'VALUE', 'TIMESTAMP']]

        return masked

    def _aggregate_concept_optimized(
        self,
        concept_data: pd.DataFrame,
        concept_name: str,
        is_categorical: bool = False,
        is_multi_label: bool = False
    ) -> pd.DataFrame:
        """Optimized aggregation using GPU or efficient pandas operations."""
        if "drop_features" in self.cfg and concept_name in self.cfg["drop_features"]:
            drop_features = self.cfg["drop_features"][concept_name]
            concept_data = concept_data[~concept_data['FEATURE'].isin(drop_features)]

        if len(concept_data) == 0:
            logger.warning(f"No data to aggregate for {concept_name}")
            return pd.DataFrame()

        if is_categorical:
            logger.info(f"Aggregating categorical concept: {concept_name}")
            return self._aggregate_categorical_optimized(concept_data, concept_name)
        else:
            logger.info(f"Aggregating numeric concept: {concept_name}")
            return self._aggregate_numeric_optimized(concept_data, concept_name)

    def _aggregate_categorical_optimized(
        self,
        concept_data: pd.DataFrame,
        concept_name: str
    ) -> pd.DataFrame:
        """Optimized categorical aggregation using pivot_table (no per-value loop)."""
        concept_data = concept_data[concept_data['VALUE'].notna()]
        unique_values = concept_data['VALUE'].unique()

        if len(unique_values) == 0:
            logger.warning(f"No unique values for categorical concept {concept_name}")
            return pd.DataFrame()

        # Count occurrences per patient-value pair
        value_counts = (
            concept_data.groupby(['PID', 'VALUE'])
            .size()
            .reset_index(name='count')
        )

        # Pivot counts: one column per VALUE, one row per PID
        count_pivot = value_counts.pivot_table(
            index='PID', columns='VALUE', values='count', fill_value=0
        )

        # Build given (binary) from counts
        given_pivot = (count_pivot > 0).astype(int)

        def _clean(v):
            return str(v).replace(' ', '_').replace('/', '_').replace('-', '_')

        count_cols = {v: f'{_clean(v)}_{concept_name}_count' for v in count_pivot.columns}
        given_cols = {v: f'{_clean(v)}_{concept_name}_given' for v in given_pivot.columns}

        count_pivot = count_pivot.rename(columns=count_cols)
        given_pivot = given_pivot.rename(columns=given_cols)

        self.categorical_features.extend(given_cols.values())
        self.continuous_features.extend(count_cols.values())

        result = pd.concat([given_pivot, count_pivot], axis=1).reset_index()
        result = result.fillna(0)
        return result

    def _aggregate_numeric_optimized(
        self,
        concept_data: pd.DataFrame,
        concept_name: str
    ) -> pd.DataFrame:
        """Highly optimized numeric aggregation."""
        concept_data = concept_data.copy()
        concept_data['VALUE_numeric'] = pd.to_numeric(concept_data['VALUE'], errors='coerce')

        valid_data = concept_data[concept_data['VALUE_numeric'].notna()].copy()

        if len(valid_data) == 0:
            return pd.DataFrame()

        if self.use_gpu:
            try:
                return self._aggregate_numeric_gpu(valid_data, concept_name)
            except Exception as e:
                logger.warning(f"GPU aggregation failed, falling back to CPU: {e}")

        return self._aggregate_numeric_cpu(valid_data, concept_name)

    def _aggregate_numeric_gpu(
        self,
        valid_data: pd.DataFrame,
        concept_name: str
    ) -> pd.DataFrame:
        """GPU-accelerated numeric aggregation using cuDF."""
        gdf = cudf.from_pandas(valid_data)

        agg_dict = {}
        for agg_func in self.agg_funcs:
            if agg_func in ['first', 'last']:
                if agg_func not in agg_dict:
                    gdf_sorted = gdf.sort_values('TIMESTAMP')
                    if agg_func == 'first':
                        agg_result = gdf_sorted.groupby(['PID', 'FEATURE'])['VALUE_numeric'].first()
                    else:
                        agg_result = gdf_sorted.groupby(['PID', 'FEATURE'])['VALUE_numeric'].last()
                    agg_dict[agg_func] = agg_result
            else:
                agg_dict[agg_func] = (
                    gdf.groupby(['PID', 'FEATURE'])['VALUE_numeric']
                    .agg(agg_func)
                )

        result_dfs = []
        for agg_func, agg_data in agg_dict.items():
            agg_df = agg_data.to_pandas().reset_index()
            pivoted = agg_df.pivot(index='PID', columns='FEATURE', values='VALUE_numeric')
            pivoted.columns = [f"{col}_{concept_name}_{agg_func}" for col in pivoted.columns]
            self.continuous_features.extend(pivoted.columns.tolist())
            result_dfs.append(pivoted)

        result = pd.concat(result_dfs, axis=1).reset_index()
        result = result.fillna(0.0)
        return result

    def _aggregate_numeric_cpu(
        self,
        valid_data: pd.DataFrame,
        concept_name: str
    ) -> pd.DataFrame:
        """CPU-optimized numeric aggregation."""
        valid_data_sorted = valid_data.sort_values('TIMESTAMP')

        agg_operations = {}
        for agg_func in self.agg_funcs:
            if agg_func == 'first':
                agg_operations['first'] = 'first'
            elif agg_func == 'last':
                agg_operations['last'] = 'last'
            elif agg_func in ['min', 'max', 'mean', 'std', 'count', 'median']:
                agg_operations[agg_func] = agg_func

        # Single groupby with multiple aggregations
        grouped = valid_data_sorted.groupby(['PID', 'FEATURE'])['VALUE_numeric'].agg(
            list(agg_operations.values())
        ).reset_index()

        grouped.columns = ['PID', 'FEATURE'] + list(agg_operations.keys())

        result_dfs = []
        for agg_func in agg_operations.keys():
            if agg_func not in grouped.columns:
                continue

            pivoted = grouped[['PID', 'FEATURE', agg_func]].pivot(
                index='PID',
                columns='FEATURE',
                values=agg_func
            )

            pivoted.columns = [f"{col}_{concept_name}_{agg_func}" for col in pivoted.columns]
            self.continuous_features.extend(pivoted.columns.tolist())
            result_dfs.append(pivoted)

        result = pd.concat(result_dfs, axis=1).reset_index()
        result = result.fillna(0.0)
        return result

    def create_final_dataset(self):
        """Merge all aggregated concepts with base tabular data."""
        final_df = self.tab_df.copy()
        id_col = self.cfg["dataset"]["id_col"]

        for concept_name, agg_df in self.aggregated_concepts.items():
            if len(agg_df) > 0:
                pre_merge_len = len(final_df)
                final_df = final_df.merge(agg_df, left_on=id_col, right_on='PID', how='left')

                if 'PID' in final_df.columns and id_col != 'PID':
                    final_df = final_df.drop(columns=['PID'])

                assert len(final_df) == pre_merge_len, f"Merge changed row count for {concept_name}"
                logger.info(f"Merged {concept_name}: {len(agg_df.columns)-1} features")

        # Fill NaN
        feature_cols = [col for col in final_df.columns if col not in [id_col, self.target]]
        final_df[feature_cols] = final_df[feature_cols].fillna(0.0)

        self.final_df = final_df

        # Remove duplicates
        self.continuous_features = list(dict.fromkeys(self.continuous_features))
        self.categorical_features = list(dict.fromkeys(self.categorical_features))

        logger.info(f"Final dataset: {final_df.shape}")
        logger.info(f"Features: {len(self.continuous_features)} cont + {len(self.categorical_features)} cat")

    def get_features_by_type(self) -> Dict[str, List[str]]:
        """Get feature names by type."""
        return {
            'continuous': self.continuous_features,
            'categorical': self.categorical_features
        }

    def get_X_y(self, include_id: bool = False):
        """Get feature matrix and target."""
        id_col = self.cfg["dataset"]["id_col"]

        if include_id:
            X = self.final_df.drop(columns=[self.target])
        else:
            X = self.final_df.drop(columns=[id_col, self.target])

        y = self.final_df[self.target]

        return X, y

    def to_csv(self, filepath: str):
        """Save to CSV."""
        self.final_df.to_csv(filepath, index=False)
        logger.info(f"Saved to {filepath}")

    def to_pickle(self, filepath: str):
        """Save to pickle."""
        self.final_df.to_pickle(filepath)
        logger.info(f"Saved to {filepath}")
