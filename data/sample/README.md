# Sample Data Directory

This directory contains sample data files for testing and development.

## Files

### CustomerChurn_sample.csv
A small sample of the customer churn dataset (first 100 rows) for:
- Local testing of feature engineering scripts
- Validation of data formats
- Quick pipeline testing without uploading full dataset

### Expected Data Format

The customer churn dataset should have the following columns:

```
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, 
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, 
TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, 
PaymentMethod, MonthlyCharges, TotalCharges, Churn
```

## Usage

1. **For Local Testing**:
   ```python
   import pandas as pd
   df = pd.read_csv('data/sample/CustomerChurn_sample.csv')
   ```

2. **For Pipeline Validation**:
   - Use sample data to test feature engineering logic
   - Validate data preprocessing steps
   - Test model training with small dataset

## Data Privacy

- Sample data should be anonymized
- Remove any personally identifiable information
- Use only for development and testing purposes

## Full Dataset

For production use:
1. Upload your full CustomerChurn.csv to S3
2. Update the S3 path in configuration
3. Run the complete pipeline with full dataset 