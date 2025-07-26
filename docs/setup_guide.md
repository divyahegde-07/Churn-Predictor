# Churn Prediction Setup Guide

This guide will walk you through setting up the complete churn prediction pipeline from scratch.

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] **AWS Account** with billing enabled
- [ ] **AWS CLI** installed and configured
- [ ] **Python 3.8+** installed locally
- [ ] **Git** for version control
- [ ] **Customer churn dataset** (CSV format)

## 🔧 Step 1: AWS Account Setup

### 1.1 Create AWS Account
1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Follow the registration process
4. Add a payment method (required for SageMaker)

### 1.2 Install AWS CLI
```bash
# On macOS
brew install awscli

# On Windows
choco install awscli

# On Linux
pip install awscli
```

### 1.3 Configure AWS CLI
```bash
aws configure
```
Enter your:
- AWS Access Key ID
- AWS Secret Access Key  
- Default region (e.g., `us-east-1`)
- Default output format (`json`)

## 🛡️ Step 2: IAM Role Creation

### 2.1 Create SageMaker Execution Role

1. **Go to IAM Console**
   - Navigate to AWS Console → IAM → Roles

2. **Create Role**
   - Click "Create role"
   - Select "AWS service" → "SageMaker"
   - Click "Next: Permissions"

3. **Attach Policies**
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
   - `AWSGlueServiceRole`
   - `CloudWatchLogsFullAccess`

4. **Name and Create**
   - Role name: `ChurnPredictorSageMakerRole`
   - Click "Create role"

5. **Copy Role ARN**
   - Copy the Role ARN for later use
   - Format: `arn:aws:iam::123456789012:role/ChurnPredictorSageMakerRole`

### 2.2 Update Trust Relationship

1. **Edit Trust Relationship**
   - Go to your role → Trust relationships → Edit trust relationship

2. **Add Glue Service**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {
           "Service": [
             "sagemaker.amazonaws.com",
             "glue.amazonaws.com"
           ]
         },
         "Action": "sts:AssumeRole"
       }
     ]
   }
   ```

3. **Update Policy**
   - Click "Update Trust Policy"

## 🗄️ Step 3: S3 Bucket Setup

### 3.1 Create S3 Bucket

```bash
# Replace with your unique bucket name
export BUCKET_NAME="your-churn-predictor-bucket"
aws s3 mb s3://$BUCKET_NAME
```

### 3.2 Create Folder Structure

```bash
# Create the required folder structure
aws s3api put-object --bucket $BUCKET_NAME --key raw-data/
aws s3api put-object --bucket $BUCKET_NAME --key processed/
aws s3api put-object --bucket $BUCKET_NAME --key training-data/
aws s3api put-object --bucket $BUCKET_NAME --key scripts/
aws s3api put-object --bucket $BUCKET_NAME --key models/
```

### 3.3 Upload Sample Data

```bash
# Upload your CustomerChurn.csv file
aws s3 cp CustomerChurn.csv s3://$BUCKET_NAME/raw-data/CustomerChurn.csv
```

## 💻 Step 4: Local Environment Setup

### 4.1 Clone Repository

```bash
git clone <your-repository-url>
cd churn-predictor
```

### 4.2 Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 4.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.4 Configure Project

1. **Update Configuration**
   ```python
   # Edit config/config.py
   S3_BUCKET_NAME = "your-churn-predictor-bucket"  # Your bucket name
   AWS_REGION = "us-east-1"  # Your preferred region
   ```

2. **Set Environment Variables (Optional)**
   ```bash
   export CHURN_S3_BUCKET=your-churn-predictor-bucket
   export SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::123456789012:role/ChurnPredictorSageMakerRole
   ```

## 🧪 Step 5: Test Setup

### 5.1 Test AWS Connection

```python
import boto3

# Test S3 connection
s3 = boto3.client('s3')
print(s3.list_buckets())

# Test SageMaker connection
sm = boto3.client('sagemaker')
print(sm.list_training_jobs(MaxResults=1))
```

### 5.2 Test Configuration

```python
from config.config import Config

# Validate configuration
validation = Config.validate_config()
print(f"Configuration valid: {validation['valid']}")

if not validation['valid']:
    for issue in validation['issues']:
        print(f"  - {issue}")
```

### 5.3 Verify Data Upload

```python
import pandas as pd

# Load data from S3
df = pd.read_csv(Config.get_s3_raw_data_uri())
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

## 📚 Step 6: SageMaker Notebook Setup

### 6.1 Create Notebook Instance

1. **Go to SageMaker Console**
   - Navigate to AWS Console → SageMaker → Notebook instances

2. **Create Notebook Instance**
   - Name: `churn-prediction-notebook`
   - Instance type: `ml.t3.medium` (for development)
   - IAM role: Select your `ChurnPredictorSageMakerRole`

3. **Configure Git Repository (Optional)**
   - Add your repository URL for direct access

4. **Start Instance**
   - Click "Create notebook instance"
   - Wait for status to become "InService"

### 6.2 Upload Notebook

1. **Open JupyterLab**
   - Click "Open JupyterLab" when instance is ready

2. **Upload Project Files**
   - Upload the entire project folder OR
   - Clone your git repository directly

3. **Install Dependencies**
   ```bash
   # In terminal within JupyterLab
   pip install -r requirements.txt
   ```

## 🚀 Step 7: Run the Pipeline

### 7.1 Open Main Notebook

1. **Navigate to Notebook**
   - Open `notebooks/churn_prediction_pipeline.ipynb`

2. **Run Configuration Cells**
   - Execute the first few cells to validate setup

### 7.2 Follow Step-by-Step Execution

The notebook is organized into clear sections:

1. **Configuration & Validation** ✓
2. **Data Upload & Verification** ✓  
3. **Feature Engineering** (AWS Glue - Manual step)
4. **Model Training** (Automated)
5. **Hyperparameter Tuning** (Automated)
6. **Model Registry** (Automated)
7. **Model Deployment** (Automated)
8. **Endpoint Testing** (Automated)

### 7.3 Manual Steps Required

**AWS Glue Job Creation:**
1. Upload feature engineering script (automated by notebook)
2. Go to AWS Glue Console
3. Create ETL job with provided configuration
4. Run the job
5. Return to notebook to verify processed data

## 🔍 Step 8: Verification & Testing

### 8.1 Verify Each Step

- [ ] Raw data uploaded to S3
- [ ] Glue job completed successfully
- [ ] Processed data available in S3
- [ ] Training job completed
- [ ] Hyperparameter tuning finished
- [ ] Model registered in Model Registry
- [ ] Endpoint deployed successfully
- [ ] Test predictions working

### 8.2 Check AWS Console

1. **S3 Console**: Verify all data files
2. **SageMaker Console**: Check training jobs and endpoints
3. **Glue Console**: Verify ETL job status
4. **CloudWatch**: Check logs for any errors

## 💰 Step 9: Cost Management

### 9.1 Monitor Costs

1. **Set up Billing Alerts**
   - AWS Console → Billing & Cost Management → Billing preferences
   - Enable "Receive Billing Alerts"
   - Create CloudWatch alarm for cost threshold

2. **Check Cost Explorer**
   - Monitor daily costs during development
   - Review costs by service (SageMaker, S3, Glue)

### 9.2 Clean Up Resources

**After testing, clean up to avoid ongoing charges:**

```python
# In your notebook, run cleanup cells
# This will delete:
# - SageMaker endpoint
# - Endpoint configuration  
# - Model instance

# Keep for future use:
# - S3 data and artifacts
# - Model Registry entries
# - Training job history
```

## 🆘 Troubleshooting

### Common Setup Issues

1. **"Access Denied" Errors**
   - Verify IAM role has correct permissions
   - Check trust relationship includes required services

2. **"Bucket Not Found"**
   - Ensure bucket name is unique globally
   - Verify region matches your configuration

3. **"Invalid Role ARN"**
   - Copy role ARN from IAM console exactly
   - Ensure role exists in same account

4. **Python Import Errors**
   - Verify virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

5. **SageMaker Quota Errors**
   - Request quota increase for ml.m5.large instances
   - Try smaller instance types for testing

### Getting Additional Help

1. **AWS Documentation**
   - [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
   - [AWS Glue User Guide](https://docs.aws.amazon.com/glue/)

2. **AWS Support**
   - Use AWS Support Center for account-specific issues
   - Developer support plan recommended

3. **Community Resources**
   - AWS re:Post community forums
   - Stack Overflow with `amazon-sagemaker` tag

## ✅ Completion Checklist

- [ ] AWS account setup complete
- [ ] IAM roles created with proper permissions
- [ ] S3 bucket created and configured
- [ ] Local environment setup and tested
- [ ] SageMaker notebook instance running
- [ ] Project code uploaded and configured
- [ ] Pipeline executed successfully
- [ ] Endpoint deployed and tested
- [ ] Cost monitoring enabled

## 🎉 Next Steps

Once setup is complete:

1. **Explore the results** in SageMaker Console
2. **Test different configurations** by modifying `config/config.py`
3. **Add your own features** to the feature engineering script
4. **Integrate the endpoint** into your applications
5. **Set up automated retraining** pipelines

**Congratulations! Your churn prediction pipeline is now ready for production use!** 