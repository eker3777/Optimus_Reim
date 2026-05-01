import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, average_precision_score
from sklearn.model_selection import StratifiedKFold

# =====================================================================
# CONFIGURATION & FEATURE COLUMNS (Mirrored from Transformer)
# =====================================================================
CONTINUOUS_COLS = [
    "score_differential_actor", 
    "score_differential_poss",
    "n_skaters_actor", 
    "n_skaters_poss", 
    "n_skaters_opp", 
    "n_skaters_def",
    "net_empty_actor", 
    "net_empty_opp",
    "period_time_remaining", 
    "time_since_last_event",
    "distance_to_net_event", 
    "angle_to_net_event",
    "distance_from_last_event", 
    "speed_from_last_event", 
    "angle_from_last_event",
    "goalie_angle_change"
]

CATEGORICAL_COLS = [
    "event_type_id", 
    "outcome_id", 
    "period_id"
]

TARGET_COL = "target"
FOLD_COL = "fold"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ridge Regression (L2 Logistic) Baseline for Phase 6")
    parser.add_argument(
        "--data-path", 
        type=Path, 
        required=True, 
        help="Path to tensor_ready_dataset.parquet"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path(r"X:\My Files\Python\Sports Analytics Projects\Hockey\HALO\Results\Ridge_Regression_xT"), 
        help="Directory to save OOF predictions and metrics"
    )
    parser.add_argument("--c-value", type=float, default=1.0, help="Inverse of regularization strength")
    return parser.parse_args()

def build_pipeline(c_value: float, continuous_cols: list, categorical_cols: list) -> Pipeline:
    """Builds a Scikit-Learn pipeline that handles missing values, scales continuous, and OHE categoricals."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    classifier = LogisticRegression(
        penalty='l2',
        C=c_value,
        solver='lbfgs',
        max_iter=2000, 
        n_jobs=-1,
        random_state=42
    )

    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def calculate_baseline_logloss(y_train, y_test):
    """Calculates the logloss of a naive baseline model predicting train-set class frequencies."""
    class_counts = y_train.value_counts(normalize=True).sort_index()
    baseline_probs = np.tile(class_counts.values, (len(y_test), 1))
    return log_loss(y_test, baseline_probs, labels=[0, 1, 2])

def create_stratified_folds(df: pd.DataFrame, n_folds: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Create stratified K-fold splits on unique games, mirroring transformer's fold creation.
    Returns fold assignment array (one fold value per row, indexed by df.index).
    """
    print("=" * 80)
    print("CREATE STRATIFIED K-FOLD SPLITS (GAME-TIMELINE UNITS)")
    print("=" * 80)
    
    unique_games = df['game_id'].dropna().unique()
    print(f'\nTotal unique games in dataset: {len(unique_games):,}')
    
    # Create goal-count stratification (0=low, 1=med, 2=high)
    event_goal_mask = df['event_type'].astype(str).str.strip().str.lower().eq('goal')
    game_goal_counts = df.loc[event_goal_mask].groupby('game_id').size()
    total_goal_tokens = int(event_goal_mask.sum())
    
    print(f'Total goal-type events: {total_goal_tokens:,}')
    
    goal_strata = []
    for game_id in unique_games:
        goals = int(game_goal_counts.get(game_id, 0))
        if goals <= 3:
            stratum = 0
        elif goals <= 6:
            stratum = 1
        else:
            stratum = 2
        goal_strata.append(stratum)
    
    goal_strata = np.array(goal_strata)
    
    print('\nGoal strata distribution across games:')
    for stratum in [0, 1, 2]:
        count = int(np.sum(goal_strata == stratum))
        pct = count / max(1, len(goal_strata)) * 100
        if stratum == 0:
            label = 'Low (0-3 goals)'
        elif stratum == 1:
            label = 'Medium (4-6 goals)'
        else:
            label = 'High (7+ goals)'
        print(f'  {label:20s}: {count:4d} games ({pct:5.1f}%)')
    
    # Apply StratifiedKFold on games
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_assignment = np.zeros(len(df), dtype=int)
    
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(unique_games, goal_strata)):
        val_games = unique_games[va_idx]
        val_mask = df['game_id'].isin(val_games)
        fold_assignment[val_mask] = fold_idx
        
        # Report fold split
        n_val_games = len(val_games)
        n_val_rows = val_mask.sum()
        print(f'\nFold {fold_idx}: {n_val_games:,} validation games ({n_val_rows:,} rows)')
    
    return fold_assignment

def main():
    args = parse_args()
    
    # 1. Ensure Output Directory Exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    
    # Rename legacy column names to match transformer's actor/opp naming
    legacy_to_actor_opp = {
        'net_empty_poss': 'net_empty_actor',
        'net_empty_def': 'net_empty_opp',
        'home_team_poss': 'home_team_actor'
    }
    for old_col, new_col in legacy_to_actor_opp.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Mirror Transformer logic: Drop EOS tokens
    initial_rows = len(df)
    if "is_eos" in df.columns:
        df = df[df["is_eos"] == 0].copy()
    
    # Create fold assignments using stratified K-fold on unique games
    fold_assignment = create_stratified_folds(df, n_folds=5, random_state=42)
    df['fold'] = fold_assignment
    
    # Drop rows with missing target or required features
    df = df.dropna(subset=[TARGET_COL]).copy()
    print(f"Filtered out EOS tokens and NaNs. Rows remaining: {len(df):,} (from {initial_rows:,})")
    
    # Verify feature columns exist
    available_continuous = [col for col in CONTINUOUS_COLS if col in df.columns]
    available_categorical = [col for col in CATEGORICAL_COLS if col in df.columns]
    
    missing_continuous = [col for col in CONTINUOUS_COLS if col not in df.columns]
    missing_categorical = [col for col in CATEGORICAL_COLS if col not in df.columns]
    
    if missing_continuous:
        print(f"Warning: Missing continuous features: {missing_continuous}")
    if missing_categorical:
        print(f"Warning: Missing categorical features: {missing_categorical}")
    
    if not available_continuous or not available_categorical:
        raise ValueError(f"Not enough features available. Continuous: {len(available_continuous)}, Categorical: {len(available_categorical)}")
    
    print(f"\nUsing {len(available_continuous)} continuous + {len(available_categorical)} categorical features")
    
    # Prepare features and target
    feature_cols = available_continuous + available_categorical
    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)
    folds = df[FOLD_COL].astype(int)
    
    unique_folds = np.sort(folds.unique())
    print(f"Detected {len(unique_folds)} folds: {unique_folds}")
    
    # Storage for OOF outputs
    oof_predictions_list = []
    metrics_list = []
    
    print("\n" + "="*60)
    print(f" TRAINING RIDGE CLASSIFIER (L2 Logistic, C={args.c_value})")
    print("="*60)
    
    for fold_idx in unique_folds:
        # Train / Test split
        train_mask = (folds != fold_idx)
        val_mask = (folds == fold_idx)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        # Build and fit pipeline with available columns
        pipeline = build_pipeline(args.c_value, available_continuous, available_categorical)
        pipeline.fit(X_train, y_train)
        
        # Predict Probabilities
        y_pred_proba = pipeline.predict_proba(X_val)
        
        # Calculate Metrics
        fold_logloss = log_loss(y_val, y_pred_proba, labels=[0, 1, 2])
        base_logloss = calculate_baseline_logloss(y_train, y_val)
        skill_impr = 100.0 * (base_logloss - fold_logloss) / base_logloss
        
        y_val_bin_0 = (y_val == 0).astype(int)
        y_val_bin_1 = (y_val == 1).astype(int)
        
        aucpr_0 = average_precision_score(y_val_bin_0, y_pred_proba[:, 0])
        aucpr_1 = average_precision_score(y_val_bin_1, y_pred_proba[:, 1])
        
        # Store predictions for this fold (preserving original dataframe index)
        fold_preds = pd.DataFrame({
            'fold': fold_idx,
            'target': y_val,
            'P_actor_goal': y_pred_proba[:, 0], # Class 0
            'P_opp_goal': y_pred_proba[:, 1],   # Class 1
            'P_no_goal': y_pred_proba[:, 2]     # Class 2
        }, index=X_val.index)
        oof_predictions_list.append(fold_preds)
        
        # Store metrics
        metrics_list.append({
            'Fold': fold_idx,
            'LogLoss': fold_logloss,
            'Skill_Improvement_%': skill_impr,
            'AUCPR_Possession_Goal': aucpr_0,
            'AUCPR_Defending_Goal': aucpr_1
        })
        
        print(f"Fold {fold_idx} | LogLoss: {fold_logloss:.4f} (Skill: {skill_impr:+.2f}%) | "
              f"AUCPR_0: {aucpr_0:.4f} | AUCPR_1: {aucpr_1:.4f}")
        
    # =====================================================================
    # EXPORT RESULTS
    # =====================================================================
    print("\n" + "="*60)
    print(" SAVING OOF PREDICTIONS AND METRICS")
    print("="*60)
    
    # 1. Compile and Save OOF Predictions
    # Concatenate all fold predictions and sort by original index to match input dataset order
    oof_df = pd.concat(oof_predictions_list).sort_index()
    
    # Join the original target and event keys back in if they exist in the original df
    # so you have a complete row just like your transformer output.
    keys_to_keep = [col for col in ['game_id', 'sl_event_id', 'event_type'] if col in df.columns]
    if keys_to_keep:
        oof_df = df[keys_to_keep].join(oof_df)
        
    preds_out_path = args.output_dir / "ridge_oof_predictions.parquet"
    oof_df.to_parquet(preds_out_path, index=False)
    print(f"Saved OOF Predictions to: {preds_out_path}")
    
    # 2. Compile and Save Metrics Summary
    metrics_df = pd.DataFrame(metrics_list)
    
    # Calculate Mean row and append it
    mean_metrics = metrics_df.mean().to_dict()
    mean_metrics['Fold'] = 'Mean'
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_metrics])], ignore_index=True)
    
    metrics_out_path = args.output_dir / "ridge_metrics_summary.csv"
    metrics_df.to_csv(metrics_out_path, index=False)
    print(f"Saved Metrics Summary to: {metrics_out_path}")
    
    print("\nTraining and Export Complete.")

if __name__ == "__main__":
    main()