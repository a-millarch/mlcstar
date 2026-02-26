"""
train.py — Single-dataset model training with 5-fold CV hyperparameter optimisation.

Trains three binary classifiers on a single AggregatedDS:
  - ExplainableBoostingClassifier (EBM, via interpret)
  - RandomForestClassifier        (sklearn)
  - SVC                           (sklearn)

Workflow
--------
1. Load base_df and apply the configured holdout split (20 % held out, 80 % train).
2. Build a single AggregatedDS across the full base_df (optionally with a masking
   point so only data before a given time offset is used).
3. Partition the aggregated X/y into train and holdout sets by patient ID.
4. For each model: run 5-fold stratified cross-validation to optimise AUROC and
   identify the best hyperparameters.
5. Refit on the full 80 % training set using those best hyperparameters.
6. Evaluate on the 20 % holdout (AUROC, AUPRC).
7. Save each trained pipeline and a summary CSV.

Usage
-----
    python mlcstar/training/train.py
    python mlcstar/training/train.py --masking_point 24h --save_dir models/training
    python mlcstar/training/train.py --n_folds 5 --ebm_n_iter 30 --rf_n_iter 50
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from interpret.glassbox import ExplainableBoostingClassifier

from mlcstar.data.datasets import AggregatedDS
from mlcstar.utils import get_cfg, get_base_df, get_train_test_split, logger

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------

EBM_PARAM_GRID = {
    "model__interactions": [0, 3, 5, 10],
    "model__max_leaves": [2, 3, 5],
    "model__learning_rate": [0.01, 0.05, 0.1],
}

RF_PARAM_GRID = {
    "model__n_estimators": [100, 200, 500],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__max_features": ["sqrt", "log2"],
}

SVM_PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "model__kernel": ["rbf", "linear"],
    "model__gamma": ["scale", "auto"],
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(
    categorical_features: list,
    continuous_features: list,
    scale_continuous: bool = False,
    encode_categoricals: bool = False,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that imputes (and optionally scales) features.

    Categorical features: mode imputation, with optional ordinal encoding
                          (needed for RF / SVM; EBM handles strings natively).
    Continuous features:  mean imputation, with optional StandardScaler (needed
                          for SVM).
    """
    cat_steps: list = [("imputer", SimpleImputer(strategy="most_frequent"))]
    if encode_categoricals:
        cat_steps.append(("to_str", FunctionTransformer(np.ndarray.astype, kw_args={"dtype": str})))
        cat_steps.append(("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
        )))
    cont_steps: list = [("imputer", SimpleImputer(strategy="mean"))]
    if scale_continuous:
        cont_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline(cat_steps), categorical_features),
            ("cont", Pipeline(cont_steps), continuous_features),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# CV + final training
# ---------------------------------------------------------------------------

def run_cv_and_final_train(
    model_name: str,
    base_model,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    categorical_features: list,
    continuous_features: list,
    scale_continuous: bool = False,
    encode_categoricals: bool = False,
    n_iter: int = 30,
    n_cv_folds: int = 5,
    random_state: int = 42,
    search_n_jobs: int = -1,
) -> dict:
    """
    Run stratified k-fold CV for hyperparameter search, then evaluate on holdout.

    RandomizedSearchCV with refit=True already refits the best estimator on the
    full training set, so no separate retraining step is needed.

    Parameters
    ----------
    model_name        : Short identifier used in logs and output filenames.
    base_model        : Unfitted sklearn-compatible estimator.
    param_grid        : Parameter distributions for RandomizedSearchCV.
                        Keys must be prefixed with "model__".
    X_train / y_train : Training features and labels (80 % split).
    X_holdout / y_holdout : Held-out features and labels (20 % split).
    categorical_features  : Column names for categorical features.
    continuous_features   : Column names for continuous features.
    scale_continuous  : Whether to add StandardScaler to continuous pipeline.
    encode_categoricals : Whether to ordinal-encode categorical features
                          (needed for RF / SVM; EBM handles strings natively).
    n_iter            : Number of random hyperparameter combinations to try.
    n_cv_folds        : Number of cross-validation folds.
    random_state      : Seed for reproducibility.
    search_n_jobs     : Parallelism for RandomizedSearchCV (-1 = all cores).

    Returns
    -------
    dict with best_params, cv_auroc, holdout_auroc, holdout_auprc, and the
    fitted pipeline (keyed as "pipeline").
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  Model: {model_name.upper()}")
    logger.info(f"{'=' * 70}")

    preprocessor = build_preprocessor(
        categorical_features, continuous_features, scale_continuous,
        encode_categoricals,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", base_model),
    ])

    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        refit=True,        # best estimator is refitted on full X_train / y_train
        n_jobs=search_n_jobs,
        random_state=random_state,
        verbose=1,
        error_score="raise",
    )

    logger.info(
        f"Running {n_cv_folds}-fold stratified CV  "
        f"({n_iter} random hyperparameter combinations, scoring=AUROC)..."
    )
    search.fit(X_train, y_train)

    # Strip pipeline prefix from param names for cleaner logging/saving
    best_params = {
        k.replace("model__", ""): v for k, v in search.best_params_.items()
    }
    cv_auroc = search.best_score_

    logger.info(f"Best CV AUROC : {cv_auroc:.4f}")
    logger.info(f"Best params   : {best_params}")

    # Evaluate the already-refitted best estimator on the holdout set
    best_pipeline = search.best_estimator_
    y_proba = best_pipeline.predict_proba(X_holdout)[:, 1]

    holdout_auroc = roc_auc_score(y_holdout, y_proba)
    holdout_auprc = average_precision_score(y_holdout, y_proba)

    logger.info(f"Holdout AUROC : {holdout_auroc:.4f}")
    logger.info(f"Holdout AUPRC : {holdout_auprc:.4f}")

    return {
        "model_name": model_name,
        "best_params": best_params,
        "cv_auroc": cv_auroc,
        "holdout_auroc": holdout_auroc,
        "holdout_auprc": holdout_auprc,
        "n_train": len(X_train),
        "n_holdout": len(X_holdout),
        "n_positive_train": int(y_train.sum()),
        "n_positive_holdout": int(y_holdout.sum()),
        "pipeline": best_pipeline,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    logger.info("=" * 70)
    logger.info("TRAINING — EBM / RandomForest / SVM  (single AggregatedDS)")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load base_df and determine holdout split
    # ------------------------------------------------------------------
    cfg = get_cfg()
    logger.info("Loading base dataframe...")
    base_df_full = get_base_df()
    logger.info(f"Total patients: {len(base_df_full)}")

    train_df, holdout_df = get_train_test_split(cfg, base_df_full)
    logger.info(
        f"Train : {len(train_df)} patients  |  "
        f"Holdout : {len(holdout_df)} patients  "
        f"(strategy: {cfg.get('holdout_type', 'temporal')})"
    )

    # ------------------------------------------------------------------
    # 2. Build a single AggregatedDS over the entire base_df
    #    (concept data is filtered per-patient inside AggregatedDS, so
    #     building on the full base_df and splitting afterwards is safe
    #     and avoids loading raw concept files twice.)
    # ------------------------------------------------------------------
    masking_point: pd.Timedelta | None = (
        pd.Timedelta(args.masking_point) if args.masking_point else None
    )
    logger.info(
        f"Building AggregatedDS  "
        f"(masking_point={masking_point if masking_point is not None else 'none — all data'})..."
    )

    agg_ds = AggregatedDS(
        cfg=cfg,
        base_df=base_df_full,
        masking_point=masking_point,
        agg_funcs=["first", "last", "min", "max", "mean", "std"],
        concepts=cfg["concepts"],
        default_mode=True,
    )

    id_col = cfg["dataset"]["id_col"]
    X_full, y_full = agg_ds.get_X_y(include_id=True)   # X retains id_col

    categorical_features = agg_ds.categorical_features
    continuous_features = agg_ds.continuous_features
    feature_cols = [c for c in X_full.columns if c != id_col]

    logger.info(
        f"Full aggregated dataset : {X_full.shape}  "
        f"({len(categorical_features)} categorical + {len(continuous_features)} continuous)"
    )
    logger.info(f"Overall positive rate   : {y_full.mean() * 100:.1f} %")

    # ------------------------------------------------------------------
    # 3. Partition into train / holdout by patient ID
    # ------------------------------------------------------------------
    train_pids = set(train_df[id_col].values)
    holdout_pids = set(holdout_df[id_col].values)

    train_mask = X_full[id_col].isin(train_pids)
    holdout_mask = X_full[id_col].isin(holdout_pids)

    X_train = X_full.loc[train_mask, feature_cols].reset_index(drop=True)
    X_holdout = X_full.loc[holdout_mask, feature_cols].reset_index(drop=True)
    y_train = y_full[train_mask].to_numpy()
    y_holdout = y_full[holdout_mask].to_numpy()

    logger.info(
        f"Train set  : {len(X_train)} samples  ({int(y_train.sum())} positive)\n"
        f"Holdout set: {len(X_holdout)} samples  ({int(y_holdout.sum())} positive)"
    )

    if len(np.unique(y_train)) < 2 or len(np.unique(y_holdout)) < 2:
        raise ValueError(
            "Insufficient class diversity in train or holdout set — "
            "cannot compute AUROC."
        )

    # ------------------------------------------------------------------
    # 4. Define model configurations
    # ------------------------------------------------------------------
    model_configs = [
        {
            "name": "ebm",
            "model": ExplainableBoostingClassifier(random_state=args.random_state),
            "param_grid": EBM_PARAM_GRID,
            "scale_continuous": False,
            "encode_categoricals": False,  # EBM handles strings natively
            "n_iter": args.ebm_n_iter,
            # Limit CV parallelism to avoid over-subscription with EBM's own
            # internal joblib parallelism.
            "search_n_jobs": 1,
        },
        {
            "name": "random_forest",
            "model": RandomForestClassifier(
                random_state=args.random_state, n_jobs=-1
            ),
            "param_grid": RF_PARAM_GRID,
            "scale_continuous": False,
            "encode_categoricals": True,
            "n_iter": args.rf_n_iter,
            "search_n_jobs": -1,
        },
        {
            "name": "svm",
            # probability=True enables predict_proba (required for AUROC)
            "model": SVC(probability=True, random_state=args.random_state),
            "param_grid": SVM_PARAM_GRID,
            "scale_continuous": True,   # SVM requires feature scaling
            "encode_categoricals": True,
            "n_iter": args.svm_n_iter,
            "search_n_jobs": -1,
        },
    ]

    # ------------------------------------------------------------------
    # 5. Train each model and collect results
    # ------------------------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for mc in model_configs:
        result = run_cv_and_final_train(
            model_name=mc["name"],
            base_model=mc["model"],
            param_grid=mc["param_grid"],
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            categorical_features=categorical_features,
            continuous_features=continuous_features,
            scale_continuous=mc["scale_continuous"],
            encode_categoricals=mc["encode_categoricals"],
            n_iter=mc["n_iter"],
            n_cv_folds=args.n_folds,
            random_state=args.random_state,
            search_n_jobs=mc["search_n_jobs"],
        )

        # Save fitted pipeline
        model_path = save_dir / f"{mc['name']}_best.pkl"
        pipeline_to_save = result.pop("pipeline")
        with open(model_path, "wb") as fh:
            pickle.dump(
                {
                    "pipeline": pipeline_to_save,
                    "model_name": mc["name"],
                    "best_params": result["best_params"],
                    "categorical_features": categorical_features,
                    "continuous_features": continuous_features,
                    "masking_point": masking_point,
                    "cv_auroc": result["cv_auroc"],
                    "holdout_auroc": result["holdout_auroc"],
                    "holdout_auprc": result["holdout_auprc"],
                },
                fh,
            )
        logger.info(f"Model saved : {model_path}")

        result["model_path"] = str(model_path)
        results.append(result)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    header = f"{'Model':<18} | {'CV AUROC':>9} | {'Holdout AUROC':>13} | {'Holdout AUPRC':>13}"
    logger.info(header)
    logger.info("-" * 70)
    for r in results:
        logger.info(
            f"{r['model_name']:<18} | {r['cv_auroc']:>9.4f} | "
            f"{r['holdout_auroc']:>13.4f} | {r['holdout_auprc']:>13.4f}"
        )
    logger.info("=" * 70)

    # Save CSV (exclude the dict column best_params for plain CSV)
    results_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "best_params"} for r in results]
    )
    results_path = save_dir / "training_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved : {results_path}")

    # Also save best_params per model as a readable text file
    params_path = save_dir / "best_params.txt"
    with open(params_path, "w") as fh:
        for r in results:
            fh.write(f"[{r['model_name']}]\n")
            for k, v in r["best_params"].items():
                fh.write(f"  {k} = {v}\n")
            fh.write("\n")
    logger.info(f"Best params   : {params_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train EBM, RandomForest, and SVM classifiers using a single "
            "AggregatedDS with 5-fold CV hyperparameter optimisation (AUROC)."
        )
    )
    parser.add_argument(
        "--masking_point",
        type=str,
        default=None,
        help=(
            "Optional time offset for AggregatedDS masking (e.g. '24h', '3D'). "
            "If omitted, all available data is included."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/training",
        help="Directory to save trained model pickles and result files.",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--ebm_n_iter",
        type=int,
        default=20,
        help="RandomizedSearchCV iterations for EBM (default: 20).",
    )
    parser.add_argument(
        "--rf_n_iter",
        type=int,
        default=30,
        help="RandomizedSearchCV iterations for RandomForest (default: 30).",
    )
    parser.add_argument(
        "--svm_n_iter",
        type=int,
        default=20,
        help="RandomizedSearchCV iterations for SVM (default: 20).",
    )

    main(parser.parse_args())
