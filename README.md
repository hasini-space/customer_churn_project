# 🔮 Telecom Customer Churn Predictor

A machine learning-powered solution for predicting customer churn in telecom industries. This project includes a production-ready FastAPI backend, interactive Streamlit UI, and an optimized predictive model trained with hyperparameter tuning.

## 🎯 Project Overview

Customer churn prediction is critical for telecom businesses to identify at-risk customers and implement retention strategies. This project provides:

- **Real-time Churn Prediction**: Assess customer risk status instantly
- **Production-Grade API**: FastAPI backend for seamless integration
- **Interactive Dashboard**: Streamlit UI for easy exploration and predictions
- **Optimized ML Model**: RandomForest classifier with GridSearchCV optimization
- **Complete Pipeline**: From data generation to deployment

### Live Demo

📱 **[Try the Interactive Demo](https://hasini-space-customer-churn-project-ui-wl3p55.streamlit.app/)**

## 📊 Project Structure

```
customer_churn_project/
├── main.py                      # Model training & hyperparameter optimization
├── app.py                       # FastAPI backend for predictions
├── ui.py                        # Streamlit interactive dashboard
├── requirements.txt             # Python dependencies
├── src/
│   └── pipeline.py             # Data generation & model pipeline
└── models/
    └── churn_pipeline.joblib   # Trained model artifact
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hasini-space/customer_churn_project.git
   cd customer_churn_project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

Generate and train the optimized churn prediction model:

```bash
python main.py
```

This script will:
- Generate synthetic telecom customer data
- Build a preprocessing + classification pipeline
- Perform hyperparameter tuning via GridSearchCV
- Save the optimized model to `models/churn_pipeline.joblib`
- Display performance metrics (ROC-AUC, Classification Report)

### Running the Applications

#### Option 1: Interactive Streamlit Dashboard (Recommended)
```bash
streamlit run ui.py
```
Access at: `http://localhost:8501`

#### Option 2: FastAPI Backend
```bash
uvicorn app:app --reload
```
API Documentation: `http://localhost:8000/docs`

## 📋 Features & Components

### Model Pipeline
- **Preprocessing**: Categorical encoding, numerical scaling
- **Classifier**: RandomForest with optimized hyperparameters
- **Optimization**: GridSearchCV (5-fold CV, ROC-AUC scoring)

### Input Features
| Feature | Type | Description |
|---------|------|-------------|
| **Age** | Numeric | Customer age in years |
| **Tenure** | Numeric | Months with the company |
| **Monthly Charges** | Numeric | Monthly service cost ($) |
| **Contract Type** | Categorical | Month-to-month, One year, Two year |
| **Payment Method** | Categorical | Electronic check, Mailed check, Bank transfer, Credit card |
| **Paperless Billing** | Categorical | Yes/No |

### Output
```json
{
  "churn_prediction": 0 or 1,
  "churn_probability": 0.0 to 1.0,
  "risk_status": "High Risk" or "Low Risk"
}
```

## 🎨 Streamlit Dashboard

The interactive UI provides:
- ✅ Real-time churn risk assessment
- 📊 Visual probability metrics
- 💡 Actionable recommendations for retention
- 🔘 Intuitive form-based input with sliders and dropdowns
- 🎯 Color-coded risk indicators (success/error/warning messages)

**Features:**
- Responsive 2-column layout for better UX
- Cached model loading for fast performance
- Clear visual feedback and recommendations

## 🔌 FastAPI Endpoints

### GET `/`
Returns API health status
```bash
curl http://localhost:8000/
```

### POST `/predict`
Predicts customer churn
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Tenure": 12,
    "MonthlyCharges": 75.0,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes"
  }'
```

**Interactive Docs**: Visit `http://localhost:8000/docs` (Swagger UI)

## 📦 Dependencies

```
# Data & ML
numpy >= 1.24.0
pandas >= 2.0.0
scikit-learn >= 1.3.0
xgboost >= 2.0.0

# Serialization
joblib >= 1.3.0

# Deployment
fastapi >= 0.100.0
uvicorn >= 0.22.0
pydantic >= 2.0.0
streamlit >= 1.25.0
```

## 📈 Model Performance

After training with GridSearchCV optimization:
- **Optimal Hyperparameters**: Best depth, estimators found via CV
- **ROC-AUC Score**: High predictive performance
- **Evaluation Metrics**: Precision, Recall, F1-Score, Confusion Matrix

Sample output:
```
Optimal Parameters Found: {'classifier__max_depth': X, 'classifier__n_estimators': Y}

[Classification Report]
             precision    recall  f1-score   support
       0.0      0.XXX    0.XXX    0.XXX      XXXX
       1.0      0.XXX    0.XXX    0.XXX      XXXX

Optimized ROC-AUC Score: 0.XXXX
```

## 🔄 Workflow

```
1. Data Generation (main.py)
       ↓
2. Train-Test Split
       ↓
3. Pipeline Building (preprocessing + classifier)
       ↓
4. Hyperparameter Tuning (GridSearchCV)
       ↓
5. Model Evaluation & Export
       ↓
6. Deployment (API/UI)
```

## 🛠️ Development

To extend or modify:

1. **Update data generation**: Edit `src/pipeline.py`
2. **Modify model architecture**: Update `build_model_pipeline()` in `src/pipeline.py`
3. **Add new features**: Update `CustomerData` model in `app.py` and input form in `ui.py`
4. **Retrain**: Run `python main.py`

## 💡 Use Cases

- **Proactive Retention**: Identify high-risk customers for targeted loyalty programs
- **Resource Optimization**: Allocate customer service resources efficiently
- **Business Strategy**: Analyze churn drivers and contract impact
- **Risk Assessment**: Real-time churn scoring for new/existing customers

## 📝 Example Predictions

### High-Risk Customer
```
Age: 25 | Tenure: 1 month | Contract: Month-to-month
Monthly Charges: $200 | Payment: Electronic check
→ Churn Probability: 85% | Status: 🚨 High Risk
```

### Low-Risk Customer
```
Age: 55 | Tenure: 48 months | Contract: Two year
Monthly Charges: $65 | Payment: Bank transfer
→ Churn Probability: 12% | Status: ✅ Low Risk
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available for educational and commercial use.

## 👤 Author

**Hasini** - [GitHub Profile](https://github.com/hasini-space)

## 🙏 Acknowledgments

- Built with scikit-learn, FastAPI, and Streamlit
- Designed for production-grade machine learning applications
- Optimized for real-world telecom industry applications

---

## 📞 Support & Feedback

For questions or feedback:
- Open an issue on GitHub
- Check the interactive demo: [Streamlit App](https://hasini-space-customer-churn-project-ui-wl3p55.streamlit.app/)

**Happy predicting! 🚀**
