import argparse
import os

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn import metrics, preprocessing


def get_best_model(models_path: list[str]):
    best_models = [0] * len(models_path)
    model_names = [0] * len(models_path)
    best_scores = [0] * len(models_path)
    for i, m in enumerate(models_path):
        model = joblib.load(m)
        best_model = model.best_estimator_
        best_score = model.best_score_
        model_name = best_model.named_steps["clf"].__class__.__name__
        logger.info(
            f"O melhor modelo de {model_name} tem roc_auc no conjunto de validação de {best_score}"
        )
        best_models[i] = best_model
        model_names[i] = model_name
        best_scores[i] = best_score
    logger.info(
        f"O melhor modelo entre os avaliados é o {model_names[np.argmax(best_scores)]}"
    )
    return best_models[np.argmax(best_scores)]


def predict_on_test_set(test_set, best_model, threshold, save_to=None):
    bank_test = pd.read_csv(test_set)

    bank_X_tt = bank_test.drop("y", axis=1)
    bank_y_tt_pp = (
        preprocessing.LabelBinarizer().fit_transform(bank_test[["y"]]).ravel()
    )

    model_name = best_model.named_steps["clf"].__class__.__name__

    if model_name == "LinearSVC":
        y_scores = best_model.predict(bank_X_tt)
        logger.warning(
            f"Argumento threshold não utilizado para modelo de classe {model_name}"
        )
    else:
        y_preds = best_model.predict_proba(bank_X_tt)
        y_scores = y_preds[:, 1] >= threshold

    roc_auc_test = metrics.roc_auc_score(bank_y_tt_pp, y_scores)
    recall_test = metrics.recall_score(bank_y_tt_pp, y_scores)
    precision_test = metrics.precision_score(bank_y_tt_pp, y_scores)

    logger.info(
        f"Métricas do modelo {model_name} no conjunto de teste usando o threshold {threshold}:"
    )
    logger.info(f"ROC AUC: {roc_auc_test}")
    logger.info(f"Recall: {recall_test}")
    logger.info(f"Precision: {precision_test}")

    bank_test["y_preds"] = y_scores

    if save_to:
        bank_test.to_csv(
            os.path.join(save_to, "bank_test_predictions.csv"), index=False
        )

    return bank_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--save_to", type=str)
    args = parser.parse_args()
    best_model_ = get_best_model(args.models)
    predict_on_test_set(
        test_set=args.test_set,
        best_model=best_model_,
        threshold=args.threshold,
        save_to=args.save_to,
    )
