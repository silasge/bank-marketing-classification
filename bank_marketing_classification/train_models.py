import argparse
from typing import Optional

import joblib
import pandas as pd
from loguru import logger
from sklearn import model_selection, pipeline

from bank_marketing_classification.models import models
from bank_marketing_classification.preprocessing import feature_engineering


def train_model(
    train_data: str,
    model: str,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_iter: int = 20,
    random_state: int = 42,
    save_to: Optional[str] = None,
):
    df = pd.read_csv(train_data)
    X = df.drop("y", axis=1)
    y = feature_engineering.transform_label.fit_transform(df[["y"]]).ravel()

    model_ = models.get_model(model, random_state=random_state)

    model_pipe = pipeline.Pipeline(
        [
            ("transf_features", feature_engineering.transform_features),
            ("clf", model_["estimator"]),
        ]
    )

    model_cv = model_selection.RandomizedSearchCV(
        model_pipe,
        model_["grid"],
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring,
        n_iter=n_iter,
    )

    model_cv.fit(X, y)

    if save_to:
        joblib.dump(value=model_cv, filename=save_to)

    return model_cv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("save_to", type=str)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    logger.info(f"Treinando modelo {args.model}.")
    train_model(
        train_data=args.train_data,
        model=args.model,
        cv=args.cv,
        scoring=args.scoring,
        n_iter=args.n_iter,
        random_state=args.random_state,
        save_to=args.save_to,
    )
    logger.info("Treinamento concluído.")
