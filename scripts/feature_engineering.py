"""
AWS Glue Job Script for Churn Prediction Feature Engineering

This script processes raw customer churn data and creates features for ML training.
Designed to run as an AWS Glue job with configurable input/output paths.
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql import functions as F
from pyspark.sql.types import *
from awsglue.job import Job

# Configuration (inline for Glue job)
class FeatureConfig:
    """Configuration specific to feature engineering"""
    
    # Service columns for feature engineering
    SERVICE_COLUMNS = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Columns to process
    TARGET_COLUMN = "churn_binary"
    TOTAL_CHARGES_COLUMN = "TotalCharges"
    MONTHLY_CHARGES_COLUMN = "MonthlyCharges"
    TENURE_COLUMN = "tenure"
    CHURN_COLUMN = "Churn"
    
    # Risk indicators
    HIGH_RISK_PAYMENT_METHOD = "Electronic check"
    HIGH_RISK_CONTRACT = "Month-to-month"
    HIGH_RISK_INTERNET_SERVICE = "Fiber optic"
    
    # Thresholds
    HIGH_CHARGES_THRESHOLD = 80
    SHORT_TENURE_THRESHOLD = 12

def clean_data(df):
    """Clean and prepare the raw data"""
    
    print("Starting data cleaning...")
    
    # Handle TotalCharges column - replace " " with 0.0 and convert to double
    df = df.withColumn(
        FeatureConfig.TOTAL_CHARGES_COLUMN,
        F.when(F.col(FeatureConfig.TOTAL_CHARGES_COLUMN) == " ", 0.0)
        .otherwise(F.col(FeatureConfig.TOTAL_CHARGES_COLUMN).cast("double"))
    )
    
    # Create binary churn target
    df = df.withColumn(
        FeatureConfig.TARGET_COLUMN,
        F.when(F.col(FeatureConfig.CHURN_COLUMN) == "Yes", 1).otherwise(0)
    )
    
    print("Data cleaning completed")
    return df

def create_customer_value_features(df):
    """Create customer value and engagement features"""
    
    print("Creating customer value features...")
    
    # Tenure-based features
    df = df.withColumn("tenure_years", F.col(FeatureConfig.TENURE_COLUMN) / 12.0)
    
    # Charges per tenure (avoid division by zero)
    df = df.withColumn(
        "charges_per_tenure", 
        F.col(FeatureConfig.MONTHLY_CHARGES_COLUMN) / (F.col(FeatureConfig.TENURE_COLUMN) + 1)
    )
    
    # Average monthly charges (from total charges)
    df = df.withColumn(
        "avg_monthly_charges",
        F.when(F.col(FeatureConfig.TENURE_COLUMN) > 0, 
               F.col(FeatureConfig.TOTAL_CHARGES_COLUMN) / F.col(FeatureConfig.TENURE_COLUMN))
        .otherwise(F.col(FeatureConfig.MONTHLY_CHARGES_COLUMN))
    )
    
    print("Customer value features created")
    return df

def create_service_adoption_features(df):
    """Create service adoption and usage features"""
    
    print("Creating service adoption features...")
    
    # Count total services used
    total_services = F.lit(0)
    for col in FeatureConfig.SERVICE_COLUMNS:
        total_services = total_services + F.when(F.col(col) == "Yes", 1).otherwise(0)
    
    df = df.withColumn("total_services", total_services)
    
    # Service adoption rate
    df = df.withColumn(
        "service_adoption_rate", 
        F.col("total_services") / len(FeatureConfig.SERVICE_COLUMNS)
    )
    
    print("Service adoption features created")
    return df

def create_risk_indicator_features(df):
    """Create risk indicator features based on business logic"""
    
    print("Creating risk indicator features...")
    
    # High risk payment method
    df = df.withColumn(
        "high_risk_payment",
        F.when(F.col("PaymentMethod") == FeatureConfig.HIGH_RISK_PAYMENT_METHOD, 1).otherwise(0)
    )
    
    # Month-to-month contract risk
    df = df.withColumn(
        "month_to_month_risk",
        F.when(F.col("Contract") == FeatureConfig.HIGH_RISK_CONTRACT, 1).otherwise(0)
    )
    
    # Fiber optic service (often associated with higher churn)
    df = df.withColumn(
        "has_fiber_optic",
        F.when(F.col("InternetService") == FeatureConfig.HIGH_RISK_INTERNET_SERVICE, 1).otherwise(0)
    )
    
    # Senior citizen indicator (already exists but renaming for clarity)
    df = df.withColumn("is_senior_citizen", F.col("SeniorCitizen"))
    
    print("Risk indicator features created")
    return df

def create_interaction_features(df):
    """Create interaction features combining multiple attributes"""
    
    print("Creating interaction features...")
    
    # Fiber optic with high charges
    df = df.withColumn(
        "fiber_high_charges",
        F.when(
            (F.col("has_fiber_optic") == 1) & 
            (F.col(FeatureConfig.MONTHLY_CHARGES_COLUMN) > FeatureConfig.HIGH_CHARGES_THRESHOLD), 
            1
        ).otherwise(0)
    )
    
    # Senior citizen with short tenure
    df = df.withColumn(
        "senior_short_tenure",
        F.when(
            (F.col("is_senior_citizen") == 1) & 
            (F.col(FeatureConfig.TENURE_COLUMN) < FeatureConfig.SHORT_TENURE_THRESHOLD), 
            1
        ).otherwise(0)
    )
    
    print("Interaction features created")
    return df

def feature_engineering_pipeline(df):
    """Run the complete feature engineering pipeline"""
    
    print("Starting feature engineering pipeline...")
    print(f"Input data shape: {df.count()} rows, {len(df.columns)} columns")
    
    # Apply all feature engineering steps
    df = clean_data(df)
    df = create_customer_value_features(df)
    df = create_service_adoption_features(df)
    df = create_risk_indicator_features(df)
    df = create_interaction_features(df)
    
    print(f"Final data shape: {df.count()} rows, {len(df.columns)} columns")
    print("Feature engineering pipeline completed")
    
    return df

def main():
    """Main function to run the Glue job"""
    
    # Get job parameters
    args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_PATH', 'OUTPUT_PATH'])
    
    # Initialize Glue context
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)
    
    print("="*60)
    print("CHURN PREDICTION FEATURE ENGINEERING JOB")
    print("="*60)
    print(f"Job Name: {args['JOB_NAME']}")
    print(f"Input Path: {args['INPUT_PATH']}")
    print(f"Output Path: {args['OUTPUT_PATH']}")
    
    try:
        # Read the input data
        print(f"Reading data from: {args['INPUT_PATH']}")
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(args['INPUT_PATH'])
        
        # Run feature engineering pipeline
        df_featured = feature_engineering_pipeline(df)
        
        # Display sample of processed data
        print("\nSample of processed data:")
        df_featured.show(5, truncate=False)
        
        # Show feature summary
        print("\nFeature summary:")
        feature_cols = [col for col in df_featured.columns if col not in ['customerID']]
        for col in sorted(feature_cols):
            print(f"  - {col}")
        
        # Save processed data
        print(f"\nSaving processed data to: {args['OUTPUT_PATH']}")
        df_featured.write.mode("overwrite").option("header", "true").csv(args['OUTPUT_PATH'])
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"ERROR: Feature engineering failed: {str(e)}")
        raise e
    
    finally:
        job.commit()

if __name__ == "__main__":
    main() 