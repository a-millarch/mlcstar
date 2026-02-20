"""
run_pipeline.py — End-to-end mlcstar pipeline orchestration.

Steps:
  1. Create base_df  (cohort construction)
  2. Create bin_df   (temporal bin grid)
  3. Collect raw concepts from Azure
  4. Filter to in-hospital records
  5. Map/bin concepts to temporal grid
  6. Train EBM models over time
  7. (Optional) Evaluate on test set

Usage:
    # From the mlcstar project root:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip_data_prep
    python scripts/run_pipeline.py --max_days 14 --cut_hours 48
"""

import argparse
import sys
import os

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlcstar.utils import ProjectManager, cfg, logger, is_file_present, get_base_df, get_train_test_split
import mlcstar.data.build_patient_info as bpi
from mlcstar.make_data import proces_raw_concepts, proces_inhospital_concepts, map_data_optimized
from mlcstar.models.ebm.train_ebm_over_time import train_multiple_ebms


def run(args):
    pm = ProjectManager()
    logger.info("=" * 80)
    logger.info("mlcstar pipeline starting")
    logger.info("=" * 80)

    # ============================================================================
    # STEP 1: Create base_df
    # ============================================================================
    if not args.skip_data_prep:
        if is_file_present(cfg['base_df_path']):
            logger.info(f"[SKIP] base_df found at {cfg['base_df_path']}")
        else:
            logger.info("[RUN] Creating base_df...")
            bpi.create_base_df(cfg)

        # ============================================================================
        # STEP 2: Create bin_df
        # ============================================================================
        if is_file_present(cfg['bin_df_path']):
            logger.info(f"[SKIP] bin_df found at {cfg['bin_df_path']}")
        else:
            logger.info("[RUN] Creating bin_df...")
            bpi.create_bin_df(cfg)

        # ============================================================================
        # STEP 3: Collect raw concepts from Azure
        # ============================================================================
        logger.info("[RUN] Collecting raw concepts...")
        base_df = get_base_df()
        proces_raw_concepts(cfg, base=base_df, reset=args.reset_raw)

        # ============================================================================
        # STEP 4: Filter to in-hospital records
        # ============================================================================
        logger.info("[RUN] Filtering to in-hospital records...")
        proces_inhospital_concepts(cfg, reset=args.reset_interim)

        # ============================================================================
        # STEP 5: Map/bin concepts
        # ============================================================================
        logger.info("[RUN] Mapping concepts to temporal bins...")
        map_data_optimized(cfg)

    # ============================================================================
    # STEP 6: Train EBM models over time
    # ============================================================================
    if not args.skip_training:
        logger.info("[RUN] Training EBM models over time...")

        ebm_params = {
            'random_state': 42,
            'interactions': args.interactions,
            'validation_size': 0.2,
            'early_stopping_rounds': args.early_stopping_rounds,
            'max_leaves': args.max_leaves,
            'inner_bags': 0,
        }

        results_df = train_multiple_ebms(
            max_days=args.max_days,
            cut_hours=args.cut_hours,
            step_hours=args.step_hours,
            step_days=args.step_days,
            val_frac=args.val_frac,
            ebm_params=ebm_params,
            save_dir=args.save_dir,
            subsample_steps=args.subsample_steps,
        )

        logger.info(f"Training complete. {len(results_df)} models trained.")

    logger.info("=" * 80)
    logger.info("Pipeline complete.")
    logger.info("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="mlcstar end-to-end pipeline")

    # Pipeline control
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data preparation steps (1-5); go straight to training")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip EBM training step")
    parser.add_argument("--reset_raw", action="store_true",
                        help="Re-collect raw data even if already present")
    parser.add_argument("--reset_interim", action="store_true",
                        help="Re-filter interim data even if already present")

    # Training parameters
    parser.add_argument("--max_days", type=int, default=30)
    parser.add_argument("--cut_hours", type=int, default=72)
    parser.add_argument("--step_hours", type=int, default=1)
    parser.add_argument("--step_days", type=int, default=1)
    parser.add_argument("--val_frac", type=float, default=0.20)
    parser.add_argument("--save_dir", type=str, default="models/ebm")
    parser.add_argument("--subsample_steps", type=int, default=None)

    # EBM hyperparameters
    parser.add_argument("--interactions", type=int, default=3)
    parser.add_argument("--max_leaves", type=int, default=2)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
