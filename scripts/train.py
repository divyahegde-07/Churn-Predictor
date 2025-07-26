import argparse
import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def model_fn(model_dir):
    """Load model for SageMaker inference"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Original hyperparameters
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--scale_pos_weight', type=float, default=2.77)
    parser.add_argument('--early_stopping_rounds', type=int, default=10)
    
    # NEW: Additional tunable hyperparameters
    parser.add_argument('--alpha', type=float, default=0)           # L1 regularization
    parser.add_argument('--lambda', type=float, default=1)          # L2 regularization  
    parser.add_argument('--gamma', type=float, default=0)           # Minimum loss reduction
    parser.add_argument('--min_child_weight', type=float, default=1) # Minimum child weight
    parser.add_argument('--colsample_bylevel', type=float, default=1) # Column sampling by level
    
    # Data directories
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    args = parser.parse_args()
    
    # Fix for lambda keyword issue
    lambda_val = getattr(args, 'lambda')
    
    print("Starting XGBoost training with hyperparameters:")
    print(f"   max_depth: {args.max_depth}")
    print(f"   eta: {args.eta}")
    print(f"   subsample: {args.subsample}")
    print(f"   colsample_bytree: {args.colsample_bytree}")
    print(f"   colsample_bylevel: {args.colsample_bylevel}")
    print(f"   alpha (L1): {args.alpha}")
    print(f"   lambda (L2): {lambda_val}")
    print(f"   gamma: {args.gamma}")
    print(f"   min_child_weight: {args.min_child_weight}")
    print(f"   scale_pos_weight: {args.scale_pos_weight}")
    print(f"   num_round: {args.num_round}")
    
    # Load training data
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'), header=None)
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]
    
    # Load validation data  
    val_data = pd.read_csv(os.path.join(args.validation, 'test.csv'), header=None)
    val_y = val_data.iloc[:, 0]
    val_X = val_data.iloc[:, 1:]
    
    print(f"📊 Training data: {train_X.shape[0]} rows, {train_X.shape[1]} features")
    print(f"📊 Validation data: {val_X.shape[0]} rows, {val_X.shape[1]} features")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dval = xgb.DMatrix(val_X, label=val_y)
    
    # Set parameters (including ALL the new ones)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'error'],
        'max_depth': args.max_depth,
        'eta': args.eta,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'colsample_bylevel': args.colsample_bylevel,
        'alpha': args.alpha,                    # L1 regularization
        'lambda': lambda_val,                   # L2 regularization (fixed!)
        'gamma': args.gamma,                    # Minimum loss reduction
        'min_child_weight': args.min_child_weight,  # Minimum child weight
        'scale_pos_weight': args.scale_pos_weight,
        'seed': 42
    }
    
    print("XGBoost Parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Train model
    print("\n Starting training...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=True
    )
    
    # Make predictions
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    
    # Calculate metrics
    train_auc = roc_auc_score(train_y, train_pred)
    val_auc = roc_auc_score(val_y, val_pred)
    
    print(f"\n=== TRAINING RESULTS ===")
    print(f"Training AUC: {train_auc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    # Convert to binary predictions for classification report
    train_pred_binary = (train_pred > 0.5).astype(int)
    val_pred_binary = (val_pred > 0.5).astype(int)
    
    print(f"\n=== VALIDATION CLASSIFICATION REPORT ===")
    print(classification_report(val_y, val_pred_binary))
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print(f"\nModel saved to {args.model_dir}")