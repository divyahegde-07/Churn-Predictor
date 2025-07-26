"""
Configuration file for Churn Predictor ML Pipeline
All constants and configurable parameters should be defined here.
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the churn prediction pipeline"""
    
    # ============================================================================
    # AWS CONFIGURATION
    # ============================================================================
    
    # S3 Configuration
    S3_BUCKET_NAME = "churn-predictor-bucket"  # Change this to your bucket name
    
    # S3 Paths
    S3_RAW_DATA_PATH = "raw-data/"
    S3_PROCESSED_DATA_PATH = "processed/featured_data/"
    S3_TRAINING_DATA_PATH = "training-data/"
    S3_SCRIPTS_PATH = "scripts/"
    S3_MODELS_PATH = "models/"
    S3_MODEL_METRICS_PATH = "model-metrics/"
    S3_ENDPOINT_DATA_CAPTURE_PATH = "endpoint-data-capture/"
    
    # AWS Region
    AWS_REGION = "us-east-1"  # Change if using different region
    
    # IAM Role (will be dynamically retrieved or set)
    SAGEMAKER_EXECUTION_ROLE = None  # Will be set at runtime
    
    # ============================================================================
    # DATA CONFIGURATION
    # ============================================================================
    
    # Data file names
    RAW_DATA_FILENAME = "CustomerChurn.csv"
    
    # Feature engineering
    TARGET_COLUMN = "churn_binary"
    FEATURES_TO_DROP = ["customerID", "Churn"]  # Original columns to drop
    
    # Service columns for feature engineering
    SERVICE_COLUMNS = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Data splitting
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    STRATIFY = True
    
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    
    # XGBoost Default Parameters
    XGBOOST_DEFAULT_PARAMS = {
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_round': 100,
        'early_stopping_rounds': 10,
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'error'],
        'seed': 42
    }
    
    # Class imbalance handling
    SCALE_POS_WEIGHT = 2.77  # Calculated from data: negative_samples / positive_samples
    
    # Hyperparameter Tuning Ranges
    HYPERPARAMETER_RANGES = {
        'max_depth': (3, 10),
        'eta': (0.01, 0.3),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'colsample_bylevel': (0.5, 1.0),
        'num_round': (50, 200),
        'lambda': (0, 10),
        'alpha': (0, 10),
        'min_child_weight': (0.5, 10),
        'gamma': (0, 5)
    }
    
    # Tuning Job Configuration
    MAX_TUNING_JOBS = 10
    MAX_PARALLEL_TUNING_JOBS = 3
    TUNING_OBJECTIVE_METRIC = 'validation:auc'
    
    # ============================================================================
    # SAGEMAKER CONFIGURATION
    # ============================================================================
    
    # Training Instance Configuration
    TRAINING_INSTANCE_TYPE = 'ml.m5.large'
    TRAINING_INSTANCE_COUNT = 1
    
    # Endpoint Configuration
    ENDPOINT_INSTANCE_TYPE = 'ml.m5.large'
    ENDPOINT_INSTANCE_COUNT = 1
    
    # XGBoost Framework Version
    XGBOOST_FRAMEWORK_VERSION = '1.7-1'
    
    # Data Capture Configuration
    DATA_CAPTURE_ENABLED = True
    DATA_CAPTURE_SAMPLING_PERCENTAGE = 100
    
    # ============================================================================
    # MODEL REGISTRY CONFIGURATION
    # ============================================================================
    
    # Model Package Group
    MODEL_PACKAGE_GROUP_NAME = "churn-prediction-models"
    MODEL_PACKAGE_GROUP_DESCRIPTION = "Churn prediction models for customer retention"
    
    # Model Approval
    MODEL_APPROVAL_STATUS = 'PendingManualApproval'  # or 'Approved' for auto-approval
    
    # ============================================================================
    # AWS GLUE CONFIGURATION
    # ============================================================================
    
    # Glue Job Configuration
    GLUE_JOB_NAME = "churn-feature-engineering"
    GLUE_PYTHON_VERSION = "3"
    GLUE_WORKER_TYPE = "G.1X"
    GLUE_NUMBER_OF_WORKERS = 2
    GLUE_MAX_CONCURRENT_RUNS = 1
    GLUE_TIMEOUT = 60  # minutes
    
    # ============================================================================
    # NAMING CONVENTIONS
    # ============================================================================
    
    # Base names for resources (timestamp will be appended)
    BASE_TRAINING_JOB_NAME = "churn-xgboost"
    BASE_TUNING_JOB_NAME = "churn-hyperparameter-tuning"
    BASE_MODEL_NAME = "churn-prediction-model"
    BASE_ENDPOINT_CONFIG_NAME = "churn-prediction-config"
    BASE_ENDPOINT_NAME = "churn-prediction-endpoint"
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @classmethod
    def get_s3_uri(cls, path: str) -> str:
        """Get full S3 URI for a given path"""
        return f"s3://{cls.S3_BUCKET_NAME}/{path}"
    
    @classmethod
    def get_s3_raw_data_uri(cls) -> str:
        """Get S3 URI for raw data"""
        return cls.get_s3_uri(f"{cls.S3_RAW_DATA_PATH}{cls.RAW_DATA_FILENAME}")
    
    @classmethod
    def get_s3_processed_data_uri(cls) -> str:
        """Get S3 URI for processed data"""
        return cls.get_s3_uri(cls.S3_PROCESSED_DATA_PATH)
    
    @classmethod
    def get_s3_training_data_uri(cls) -> str:
        """Get S3 URI for training data"""
        return cls.get_s3_uri(cls.S3_TRAINING_DATA_PATH)
    
    @classmethod
    def get_s3_scripts_uri(cls) -> str:
        """Get S3 URI for scripts"""
        return cls.get_s3_uri(cls.S3_SCRIPTS_PATH)
    
    @classmethod
    def get_timestamp_suffix(cls) -> str:
        """Get timestamp suffix for resource naming"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    
    @classmethod
    def get_resource_name(cls, base_name: str) -> str:
        """Get resource name with timestamp suffix"""
        return f"{base_name}-{cls.get_timestamp_suffix()}"
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check required values
        if not cls.S3_BUCKET_NAME:
            issues.append("S3_BUCKET_NAME must be specified")
        
        if not cls.AWS_REGION:
            issues.append("AWS_REGION must be specified")
        
        # Check hyperparameter ranges
        for param, (min_val, max_val) in cls.HYPERPARAMETER_RANGES.items():
            if min_val >= max_val:
                issues.append(f"Invalid range for {param}: min ({min_val}) >= max ({max_val})")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

# ============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# ============================================================================

def load_config_from_env():
    """Load configuration overrides from environment variables"""
    
    # Override bucket name from environment
    if os.getenv('CHURN_S3_BUCKET'):
        Config.S3_BUCKET_NAME = os.getenv('CHURN_S3_BUCKET')
    
    # Override AWS region from environment
    if os.getenv('AWS_DEFAULT_REGION'):
        Config.AWS_REGION = os.getenv('AWS_DEFAULT_REGION')
    
    # Override SageMaker role from environment
    if os.getenv('SAGEMAKER_EXECUTION_ROLE'):
        Config.SAGEMAKER_EXECUTION_ROLE = os.getenv('SAGEMAKER_EXECUTION_ROLE')

# Auto-load environment overrides when module is imported
load_config_from_env() 