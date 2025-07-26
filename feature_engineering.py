import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Get job parameters
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_PATH', 'OUTPUT_PATH'])

# Initialize contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read the CSV data
print(f"Reading data from: {args['INPUT_PATH']}")
df = spark.read.option("header", "true").option("inferSchema", "true").csv(args['INPUT_PATH'])

print(f"Original data shape: {df.count()} rows, {len(df.columns)} columns")

# Data cleaning
# Fix TotalCharges column (has spaces instead of proper nulls)
df = df.withColumn("TotalCharges", 
                  F.when(F.col("TotalCharges") == " ", 0.0)
                  .otherwise(F.col("TotalCharges").cast("double")))

# Create binary churn target
df = df.withColumn("churn_binary", 
                  F.when(F.col("Churn") == "Yes", 1).otherwise(0))

# Feature engineering
print("Creating new features...")

# Customer value features
df = df.withColumn("tenure_years", F.col("tenure") / 12.0)
df = df.withColumn("charges_per_tenure", F.col("MonthlyCharges") / (F.col("tenure") + 1))
df = df.withColumn("avg_monthly_charges", 
                  F.when(F.col("tenure") > 0, F.col("TotalCharges") / F.col("tenure"))
                  .otherwise(F.col("MonthlyCharges")))

# Service adoption features
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Count total services (convert Yes to 1, others to 0)
total_services = F.lit(0)
for col in service_cols:
    total_services = total_services + F.when(F.col(col) == "Yes", 1).otherwise(0)

df = df.withColumn("total_services", total_services)
df = df.withColumn("service_adoption_rate", F.col("total_services") / len(service_cols))

# Risk indicators
df = df.withColumn("high_risk_payment", 
                  F.when(F.col("PaymentMethod") == "Electronic check", 1).otherwise(0))

df = df.withColumn("month_to_month_risk",
                  F.when(F.col("Contract") == "Month-to-month", 1).otherwise(0))

df = df.withColumn("has_fiber_optic",
                  F.when(F.col("InternetService") == "Fiber optic", 1).otherwise(0))

df = df.withColumn("is_senior_citizen", F.col("SeniorCitizen"))

# Interaction features
df = df.withColumn("fiber_high_charges",
                  F.when((F.col("has_fiber_optic") == 1) & (F.col("MonthlyCharges") > 80), 1).otherwise(0))

df = df.withColumn("senior_short_tenure",
                  F.when((F.col("is_senior_citizen") == 1) & (F.col("tenure") < 12), 1).otherwise(0))

print(f"Final data shape: {df.count()} rows, {len(df.columns)} columns")

# Save processed data
print(f"Saving processed data to: {args['OUTPUT_PATH']}")
df.write.mode("overwrite").option("header", "true").csv(args['OUTPUT_PATH'])

print("Feature engineering completed successfully!")
job.commit() 