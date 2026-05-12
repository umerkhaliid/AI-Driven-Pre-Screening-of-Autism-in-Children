from dataclasses import dataclass, field
from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.nlp_extractor import get_nlp_feature_names


@dataclass
class FeatureConfig:
    q_cols: List[str]
    nlp_cols: List[str]
    numeric_cols: List[str]  # q_cols + demographic + nlp_cols (all 37)


def get_feature_config() -> FeatureConfig:
    q_cols = [f"a{i}" for i in range(1, 25)]
    nlp_cols = get_nlp_feature_names()  # 10 NLP flag columns

    # Full 38-dimensional feature vector:
    #   24 questionnaire items (a1-a24)
    #   4  demographic fields  (age_mons, sex, jaundice, family_mem_with_asd)
    #   10 NLP symptom flags   (nlp_*)
    numeric_cols = q_cols + [
        "age_mons",
        "sex",
        "jaundice",
        "family_mem_with_asd",
    ] + nlp_cols

    return FeatureConfig(q_cols=q_cols, nlp_cols=nlp_cols, numeric_cols=numeric_cols)


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
