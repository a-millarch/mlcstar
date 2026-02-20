# preprocessing.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict


# ============================================================================
# Multi-Hot Encoding
# ============================================================================

class MultiHotCategoricalEncoder:
    """
    Converts multi-label categorical time series to multi-hot vectors.

    Input format (wide with lists):
        TIMESTEP  PID  FEATURE     0                    1                           2
        0         1    medication  aspirin              [aspirin, ibuprofen]        NaN
        1         2    medication  [insulin, aspirin]   NaN                         insulin

    Output shape: [n_samples, n_categories, seq_len]
        Example: [17390, 12, 114]
        - 17390 patients
        - 12 medication types (channels)
        - 114 timesteps
    """

    def __init__(self):
        self.encoders_ = {}
        self.n_classes_ = {}

    def fit(self, df: pd.DataFrame, sample_col: str, timestep_cols: List[str],
            cat_col: str, feature_names: Optional[List[str]] = None):
        """
        Fit encoder on multi-label categorical data in WIDE format.

        Args:
            df: DataFrame with columns [sample_col, cat_col, timestep_0, timestep_1, ...]
                Cells can contain single values, lists, or NaN
            sample_col: Column identifying each sample (e.g., 'PID')
            timestep_cols: List of timestep column names (e.g., ['0', '1', '2', ...])
            cat_col: Column containing categorical feature name (e.g., 'FEATURE')
            feature_names: Optional list of categorical feature names to process
        """
        if feature_names is None:
            feature_names = df[cat_col].unique().tolist()

        for feat_name in feature_names:
            feat_df = df[df[cat_col] == feat_name]

            all_values = set()
            for ts_col in timestep_cols:
                for cell in feat_df[ts_col].dropna():
                    if isinstance(cell, list):
                        all_values.update(cell)
                    else:
                        all_values.add(cell)

            sorted_values = sorted(list(all_values))
            value_to_idx = {val: idx for idx, val in enumerate(sorted_values)}

            self.encoders_[feat_name] = {
                'value_to_idx': value_to_idx,
                'idx_to_value': {idx: val for val, idx in value_to_idx.items()},
                'n_classes': len(sorted_values),
                'category_labels': sorted_values
            }
            self.n_classes_[feat_name] = len(sorted_values)

        return self

    def transform(self, df: pd.DataFrame, sample_col: str, timestep_cols: List[str],
                  cat_col: str) -> Tuple[np.ndarray, Dict]:
        """
        Transform multi-label categorical data to multi-hot encoding.

        Returns:
            X_multi_hot: Array of shape [n_samples, n_categories, seq_len]
            encoding_info: Dictionary with encoding details including category_labels
        """
        samples = sorted(df[sample_col].unique())
        n_samples = len(samples)
        seq_len = len(timestep_cols)

        cat_features = list(self.encoders_.keys())
        total_dim = sum(self.n_classes_[feat] for feat in cat_features)

        X_multi_hot = np.zeros((n_samples, total_dim, seq_len), dtype=np.float32)

        sample_to_idx = {sample: idx for idx, sample in enumerate(samples)}

        dim_offset = 0
        encoding_info = {
            'feature_ranges': {},
            'feature_names': cat_features,
            'category_labels': {}
        }

        for feat_name in cat_features:
            encoder = self.encoders_[feat_name]
            n_classes = encoder['n_classes']

            encoding_info['feature_ranges'][feat_name] = (dim_offset, dim_offset + n_classes)
            encoding_info['category_labels'][feat_name] = encoder['category_labels']

            feat_df = df[df[cat_col] == feat_name].set_index(sample_col)

            for sample_id in samples:
                if sample_id not in feat_df.index:
                    continue

                sample_idx = sample_to_idx[sample_id]
                sample_data = feat_df.loc[sample_id]

                if isinstance(sample_data, pd.DataFrame):
                    for ts_idx, ts_col in enumerate(timestep_cols):
                        values = set()
                        for cell in sample_data[ts_col].dropna():
                            if isinstance(cell, list):
                                values.update(cell)
                            else:
                                values.add(cell)
                        for val in values:
                            if val in encoder['value_to_idx']:
                                val_idx = encoder['value_to_idx'][val]
                                X_multi_hot[sample_idx, dim_offset + val_idx, ts_idx] = 1.0
                else:
                    for ts_idx, ts_col in enumerate(timestep_cols):
                        cell = sample_data[ts_col]
                        if isinstance(cell, (list, tuple, np.ndarray)):
                            vals = cell
                        elif pd.isna(cell):
                            vals = []
                        else:
                            vals = [cell]
                        for v in vals:
                            if v in encoder['value_to_idx']:
                                val_idx = encoder['value_to_idx'][v]
                                X_multi_hot[sample_idx, dim_offset + val_idx, ts_idx] = 1.0

            dim_offset += n_classes

        return X_multi_hot, encoding_info

    def fit_transform(self, df: pd.DataFrame, sample_col: str, timestep_cols: List[str],
                      cat_col: str, feature_names: Optional[List[str]] = None):
        """Fit and transform in one step."""
        self.fit(df, sample_col, timestep_cols, cat_col, feature_names)
        return self.transform(df, sample_col, timestep_cols, cat_col)

    def get_category_labels(self) -> Dict[str, List[str]]:
        """Get category labels for all features."""
        return {
            feat_name: encoder['category_labels']
            for feat_name, encoder in self.encoders_.items()
        }
