# Fraud Risk Scoring App

An end-to-end fraud detection pipeline trained on ~460,000 electricity customer records with a fraud rate of **1.23%**.
Due to severe class imbalance, threshold tuning is applied to optimize recall while maintaining precision.
An interactive Streamlit application allows users to upload new data, adjust the fraud threshold, and dynamically rank high-risk accounts.

## Live Demo

Deployed via Streamlit Community Cloud.
https://fraud-risk-app-bf7avzazvojk3ukwuwkkcf.streamlit.app
You can view trainig data statistics, and upload your own csv at the bottom.



## Problem Context

Fraud detection usually is an imbalanced classification problem, where fraudulent cases represent a very small proportion of total customers.

**Challenges:**
- Fraud rate ≈ 1.23%
- Accuracy is misleading (baseline ≈ 98.7%)
- Requires evaluation using ROC-AUC, precision, and recall
- Operational trade-offs between false positives and missed fraud



## Model Performance

- **ROC-AUC:** 0.816  
- **Fraud Recall (threshold = 0.30):** 44%  
- **Fraud Precision (threshold = 0.30):** 41%  
- **Fraud Rate:** 1.23%

Threshold tuning significantly improves fraud detection compared to the default 0.50 threshold.



## Approach

### Data Preprocessing
- Categorical encoding using `OneHotEncoder`
- Median imputation for numeric features
- Class imbalance handled with `class_weight="balanced"`

### Model
- Random Forest Classifier
- Probability-based fraud scoring
- Threshold optimization for operational tuning

### Deployment
- Model serialized using `joblib`
- Interactive Streamlit web application
- Adjustable fraud risk threshold
- Downloadable scored results
- Feature importance visualization



## Key Features

- Interactive fraud threshold slider  
- Risk distribution visualization  
- Ranked high-risk customer table  
- Downloadable scored dataset  
- Feature importance analysis  



## Tech Stack

- Python  
- Scikit-learn  
- Pandas / NumPy  
- Streamlit  
- Matplotlib  
- Joblib  
- Git LFS  


## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py 
```


## Data Privacy

The original dataset contains anonymized operational customer data
No personally identifiable information is included in this repository


## Key Take-away

This project demonstrates:
* Handling of real-world imbalanced datasets
* End-to-end ML pipeline development
* Threshold tuning for business decision-making
* Model deployment in an application
* Practical fraud risk scoring implementation