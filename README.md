# Customer Churn Prediction on AWS

A complete machine learning pipeline for predicting customer churn using AWS services including SageMaker, Glue, and S3.

## 🎯 Overview

This project implements an end-to-end ML pipeline that:
- Processes customer data using **AWS Glue** for feature engineering
- Trains **XGBoost models** with hyperparameter tuning on **SageMaker**
- Registers models in **SageMaker Model Registry**
- Deploys models to **real-time endpoints**
- Provides comprehensive testing and monitoring capabilities

## 📊 Performance

- **Validation AUC**: 0.845+ (improved from 0.840 baseline)
- **Features**: 20+ engineered features from raw customer data
- **Training Time**: ~15-25 minutes with hyperparameter tuning
- **Deployment**: Real-time inference endpoint ready

## 🏗️ Architecture

```
Raw Data (S3) → Feature Engineering (Glue) → Model Training (SageMaker) 
     ↓                                               ↓
Data Storage ← Model Registry ← Hyperparameter Tuning
     ↓                ↓
Data Capture ← Real-time Endpoint
```

## 📁 Project Structure

```
churn-predictor/
├── config/
│   └── config.py              # All configuration constants
├── scripts/
│   ├── feature_engineering.py # AWS Glue job script
│   └── train.py               # SageMaker training script
├── notebooks/
│   └── churn_prediction_pipeline.ipynb # Main orchestration notebook
├── data/
│   └── sample/                # Sample data for testing
├── docs/
│   └── setup_guide.md         # Detailed setup instructions
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured
3. **Python 3.8+** installed
4. **SageMaker Execution Role** created

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd churn-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**
   ```bash
   aws configure
   ```

4. **Update configuration**
   - Edit `config/config.py`
   - Set your S3 bucket name
   - Update AWS region if needed

5. **Upload your data**
   ```bash
   aws s3 cp CustomerChurn.csv s3://your-bucket/raw-data/CustomerChurn.csv
   ```

6. **Run the pipeline**
   - Open `notebooks/churn_prediction_pipeline.ipynb`
   - Execute cells step by step
   - Follow manual steps for AWS Glue job creation

## 📋 Configuration

All configuration is managed in `config/config.py`. Key settings:

```python
# S3 Configuration
S3_BUCKET_NAME = "your-bucket-name"  # UPDATE THIS

# Model Parameters
SCALE_POS_WEIGHT = 2.77  # Handles class imbalance
MAX_TUNING_JOBS = 10     # Hyperparameter tuning jobs

# Instance Types
TRAINING_INSTANCE_TYPE = 'ml.m5.large'
ENDPOINT_INSTANCE_TYPE = 'ml.m5.large'
```

### Environment Variables

You can override configuration using environment variables:

```bash
export CHURN_S3_BUCKET=my-churn-bucket
export AWS_DEFAULT_REGION=us-west-2
export SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::123456789012:role/MySageMakerRole
```

## 🔧 Usage

### 1. Feature Engineering (AWS Glue)

The feature engineering script creates:
- **Customer value features**: tenure_years, charges_per_tenure, avg_monthly_charges
- **Service adoption features**: total_services, service_adoption_rate
- **Risk indicators**: high_risk_payment, month_to_month_risk, has_fiber_optic
- **Interaction features**: fiber_high_charges, senior_short_tenure

### 2. Model Training (SageMaker)

The training process includes:
- **Baseline training** with default parameters
- **Hyperparameter tuning** across 10 different parameter combinations
- **Class imbalance handling** using scale_pos_weight
- **Early stopping** to prevent overfitting

### 3. Model Deployment

Deploy the best model to a real-time endpoint:
- **Instance type**: ml.m5.large (configurable)
- **Data capture**: 100% of requests logged
- **Auto-scaling**: Available (not configured by default)

### 4. Making Predictions

```python
import boto3

# Create runtime client
runtime = boto3.client('sagemaker-runtime')

# Prepare your data (CSV format, no headers)
csv_data = "1,65,0,29.85,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0"

# Make prediction
response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='text/csv',
    Body=csv_data
)

# Get churn probability
result = response['Body'].read().decode('utf-8')
churn_probability = float(result.strip())
print(f"Churn probability: {churn_probability:.4f}")
```

## 📊 Features

The pipeline creates 20+ features from raw customer data:

### Customer Value Features
- `tenure_years`: Customer tenure in years
- `charges_per_tenure`: Monthly charges divided by tenure
- `avg_monthly_charges`: Average monthly charges from total

### Service Adoption Features
- `total_services`: Count of services used
- `service_adoption_rate`: Percentage of available services used

### Risk Indicators
- `high_risk_payment`: Electronic check payment method
- `month_to_month_risk`: Month-to-month contract
- `has_fiber_optic`: Fiber optic internet service
- `is_senior_citizen`: Senior citizen status

### Interaction Features
- `fiber_high_charges`: Fiber optic + high monthly charges
- `senior_short_tenure`: Senior citizen + short tenure

## 💰 Cost Optimization

### Training Costs
- **Baseline training**: ~$0.50-1.00 per job
- **Hyperparameter tuning**: ~$5.00-10.00 for 10 jobs
- **Data processing**: ~$0.10-0.50 for Glue job

### Inference Costs
- **ml.m5.large endpoint**: ~$0.10 per hour
- **Data capture**: S3 storage costs only
- **No additional costs** for predictions

### Cost Reduction Tips
1. **Stop endpoints** when not needed
2. **Use smaller instances** for development
3. **Reduce tuning jobs** for faster iteration
4. **Use spot instances** for training (advanced)

## 🛠️ Development

### Adding New Features

1. **Update feature engineering script**:
   ```python
   # In scripts/feature_engineering.py
   def create_new_feature(df):
       df = df.withColumn("new_feature", F.col("existing_col") * 2)
       return df
   ```

2. **Test locally** with sample data

3. **Upload updated script** to S3

4. **Re-run Glue job**

### Changing Model Parameters

1. **Update configuration**:
   ```python
   # In config/config.py
   XGBOOST_DEFAULT_PARAMS = {
       'max_depth': 8,  # Changed from 6
       'eta': 0.05,     # Changed from 0.1
       # ... other params
   }
   ```

2. **Re-run training** notebook

### Custom Training Script

The training script (`scripts/train.py`) can be customized for:
- **Different algorithms** (Random Forest, Neural Networks)
- **Custom metrics** and evaluation
- **Advanced preprocessing**
- **Model explainability**

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Integration Tests
```bash
python tests/test_pipeline.py
```

### Model Validation
- Cross-validation on training data
- Hold-out test set evaluation
- A/B testing framework ready

## 📈 Monitoring

### Model Performance
- **Data drift detection** (configurable)
- **Performance metrics** tracking
- **CloudWatch integration**

### Endpoint Monitoring
- **Request/response logging** (100% capture)
- **Latency and error metrics**
- **Auto-scaling triggers**

## 🔒 Security

### IAM Permissions
Required permissions for SageMaker execution role:
- S3: Read/Write to your bucket
- SageMaker: Full access to training and endpoints
- Glue: Access to run ETL jobs
- CloudWatch: Logging and metrics

### Data Privacy
- **Data encryption** at rest and in transit
- **VPC deployment** available
- **Access logging** enabled

## 🆘 Troubleshooting

### Common Issues

1. **"Bucket not found"**
   - Update `Config.S3_BUCKET_NAME` in config/config.py
   - Ensure bucket exists and is accessible

2. **"Role not authorized"**
   - Check SageMaker execution role permissions
   - Verify trust relationship includes SageMaker

3. **"Training job failed"**
   - Check CloudWatch logs for detailed error messages
   - Verify data format (CSV, target as first column)

4. **"Endpoint creation failed"**
   - Check instance limits in your AWS account
   - Verify model artifacts exist in S3

### Getting Help

1. **Check AWS CloudWatch Logs** for detailed error messages
2. **Review SageMaker Console** for job status and logs
3. **Verify S3 permissions** and bucket access
4. **Test with sample data** first

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- AWS SageMaker team for excellent ML platform
- XGBoost community for the robust algorithm
- Open source contributors

---

**🎉 Ready to predict churn? Get started with the [setup guide](docs/setup_guide.md)!** 