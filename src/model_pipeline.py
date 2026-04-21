from dataclasses import dataclass
from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


@dataclass
class FeatureConfig:
    q_cols: List[str]
    numeric_cols: List[str]


def get_feature_config() -> FeatureConfig:
    q_cols = [f"a{i}" for i in range(1, 25)]

    numeric_cols = q_cols + [
        "age_mons",
        "sex",
        "jaundice",
        "family_mem_with_asd"
    ]

    return FeatureConfig(q_cols=q_cols, numeric_cols=numeric_cols)


def build_preprocessor(feature_config: FeatureConfig) -> ColumnTransformer:

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_config.numeric_cols)
        ],
        remainder="drop"
    )

    return preprocessor
