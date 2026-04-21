import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline

from src.config import PROCESSED_TRAIN_PATH, MODELS_DIR, RISK_LABELS
from src.model_pipeline import get_feature_config, build_preprocessor


def load_data():
    return pd.read_csv(PROCESSED_TRAIN_PATH)


def build_models(preprocessor):
    """Returns dict of named pipelines."""
    return {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            ))
        ]),
        "XGBoost": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="multi:softprob",
                num_class=4,
                eval_metric="mlogloss"
            ))
        ]),
    }


def select_best_model_by_cv(models, X, y):
    """
    Runs 5-fold stratified cross-validation on all models.
    Selects the best model based on mean Weighted ROC-AUC (OVR).
    Returns (best_name, best_pipeline, cv_results_df).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring="roc_auc_ovr_weighted",
            return_train_score=False
        )

        mean_auc = np.mean(scores["test_score"])
        std_auc = np.std(scores["test_score"])

        print(f"  {name:<25s}: ROC-AUC = {mean_auc:.4f} +/- {std_auc:.4f}")
        results.append({"model": name, "mean_roc_auc": mean_auc, "std_roc_auc": std_auc})

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["mean_roc_auc"].idxmax()]
    best_name = best_row["model"]

    print(f"\n  >> Best model by CV ROC-AUC: {best_name} ({best_row['mean_roc_auc']:.4f})")

    return best_name, models[best_name], results_df


def evaluate_on_test(name, model, X_test, y_test):
    """Evaluates a trained model on the held-out test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

    print(f"\n{'=' * 70}")
    print(f"TEST SET EVALUATION: {name}")
    print("=" * 70)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=list(RISK_LABELS.values())
    ))
    print("Weighted ROC-AUC (OVR):", round(auc, 4))

    return auc


def main():
    df = load_data()

    feature_config = get_feature_config()
    X = df[feature_config.numeric_cols]
    y = df["risk_class"]

    # Hold out 20% for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(feature_config)
    models = build_models(preprocessor)

    # Step 1: Select best model via 5-fold CV on training set
    print("\n" + "=" * 70)
    print("STEP 1: MODEL SELECTION VIA 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    best_name, best_pipeline, cv_results = select_best_model_by_cv(
        models, X_train, y_train
    )

    # Step 2: Retrain best model on full training set
    print("\n" + "=" * 70)
    print("STEP 2: RETRAIN BEST MODEL ON FULL TRAINING SET")
    print("=" * 70)

    best_pipeline.fit(X_train, y_train)
    test_auc = evaluate_on_test(best_name, best_pipeline, X_test, y_test)

    # Step 3: Save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, save_path)

    # Also save the model name so calibration script knows what was chosen
    joblib.dump({"model_name": best_name}, MODELS_DIR / "best_model_info.joblib")

    print(f"\n{'=' * 70}")
    print("BEST MODEL SAVED")
    print(f"  Model:          {best_name}")
    print(f"  CV ROC-AUC:     {cv_results.loc[cv_results['model'] == best_name, 'mean_roc_auc'].values[0]:.4f}")
    print(f"  Test ROC-AUC:   {test_auc:.4f}")
    print(f"  Saved to:       {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
