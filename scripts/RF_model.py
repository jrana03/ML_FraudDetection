import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import os

n_cores = os.cpu_count()

base_dir = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = base_dir / "processed_data" / "fraudTrain_processed.csv"
TESTING_DATA_PATH = base_dir / "processed_data" / "fraudTest_processed.csv"
MODEL_PATH = base_dir / "saved_models" / "rfboost_model.pkl"

def train_model(data_path):
    # Load data
    df = pd.read_csv(data_path)
    X_train = df.drop("is_fraud", axis=1)
    y_train = df["is_fraud"]

    # Initialize and train XGBoost model
    '''model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="aucpr",
        scale_pos_weight=1,
    )'''

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=n_cores-1,          
        verbose=1                
    )

    print("Starting model training...")
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print("Model training completed and saved.")


def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred = (probs >= threshold).astype(int)

    print(classification_report(
        y_test, y_pred,
        target_names=["Not Fraud (0)", "Fraud (1)"],
        digits=4
    ))

    # Raw confusion matrix
    '''cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=["Not Fraud (0)", "Fraud (1)"]
    )
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix (rows=Actual, columns=Predicted)")
    plt.show()'''

    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Fraud)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_pred

def main():
    # Only train the model if it doesn't already exist
    if not MODEL_PATH.exists():
        train_model(TRAINING_DATA_PATH) 
    else:
        print(f"Model already exists at {MODEL_PATH}, skipping training.")

    model = joblib.load(MODEL_PATH)
    print("Model loaded from disk.")

    # Load test data
    df = pd.read_csv(TESTING_DATA_PATH)
    X_test = df.drop("is_fraud", axis=1)
    y_test = df["is_fraud"]

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()