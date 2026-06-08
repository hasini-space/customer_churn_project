from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from src.pipeline import generate_mock_data, build_model_pipeline

def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    print("Step 1: Generating engine data streams...")
    df = generate_mock_data()
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Step 2: Initializing base pipeline...")
    base_pipeline, _, _ = build_model_pipeline()
    
    # Define hyperparameter grid (use 'classifier__' prefix to target the pipeline step)
    param_grid = {
        'classifier__max_depth': [3, 5, 7],
        'classifier__n_estimators': [50, 100, 150]
    }
    
    print("Step 3: Launching Hyperparameter Optimization via Grid Search CV...")
    grid_search = GridSearchCV(
        estimator=base_pipeline, 
        param_grid=param_grid, 
        cv=5, 
        scoring='roc_auc', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Optimal Parameters Found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    print("\nStep 4: Evaluating optimized model on test matrices...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*20 + " OPTIMIZED EVALUATION METRICS " + "="*20)
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred))
    print(f"Optimized ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("="*60)
    
    # Save the absolute best model pipeline configuration
    model_path = 'models/churn_pipeline.joblib'
    joblib.dump(best_model, model_path)
    print(f"\nSuccess: Optimized pipeline artifact exported to '{model_path}'")

if __name__ == "__main__":
    main()