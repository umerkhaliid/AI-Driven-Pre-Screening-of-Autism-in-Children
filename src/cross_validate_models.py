import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import PROCESSED_TRAIN_PATH
from src.model_pipeline import get_feature_config, build_preprocessor


def run_cross_validation():
    df = pd.read_csv(PROCESSED_TRAIN_PATH)

    feature_config = get_feature_config()
    X = df[feature_config.numeric_cols]
    y = df["risk_class"]

    preprocessor = build_preprocessor(feature_config)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss"
        )
    }

    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted"
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results_summary = []

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", clf)
        ])

        scores = cross_validate(
            pipeline,
            X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        print("\n" + "=" * 75)
        print(f"5-FOLD CROSS VALIDATION RESULTS: {name}")
        print("=" * 75)

        model_result = {"model": name}

        for metric in scoring.keys():
            metric_key = f"test_{metric}"
            mean_val = np.mean(scores[metric_key])
            std_val = np.std(scores[metric_key])

            display_name = metric.replace("_", " ").upper()
            print(f"{display_name:<30}: {mean_val:.4f} +/- {std_val:.4f}")

            model_result[metric] = mean_val
            model_result[f"{metric}_std"] = std_val

        results_summary.append(model_result)

    results_df = pd.DataFrame(results_summary)
    print("\n\nFINAL SUMMARY TABLE:")
    display_cols = ["model", "accuracy", "precision_weighted", "recall_weighted",
                    "f1_weighted", "roc_auc_ovr_weighted"]
    print(results_df[display_cols].to_string(index=False))

    return results_df


if __name__ == "__main__":
    run_cross_validation()
