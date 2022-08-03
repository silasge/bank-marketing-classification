import numpy as np
from scipy.stats import uniform
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn import tree


def get_model(model: str, random_state: int = 42):
    models_dict = {
        "lr": {
            "estimator": linear_model.LogisticRegression(
                random_state=random_state, solver="saga", class_weight="balanced"
            ),
            "grid": {
                "transf_features__te_job__transformer": [
                    "target_encoder",
                    "leave_one_out",
                    "one_hot_encoder",
                ],
                "transf_features__te_job__smoothing": uniform(loc=0, scale=10),
                "transf_features__te_job__sigma": uniform(loc=0, scale=1),
                "transf_features__bin_campaign__n_bins": [3, 4, 5],
                "transf_features__bin_campaign__encode": ["onehot", "ordinal"],
                "transf_features__bin_campaign__strategy": ["uniform", "quantile", "kmeans"],
                "transf_features__bin_previous__n_bins": [3, 4, 5],
                "transf_features__bin_previous__encode": ["onehot", "ordinal"],
                "transf_features__bin_previous__strategy": ["uniform", "quantile", "kmeans"],
                "clf__penalty": ["l1", "l2", "elasticnet", "none"],
                "clf__C": uniform(loc=0, scale=10),
                "clf__l1_ratio": uniform(
                    loc=0, scale=1
                ),  # Usado somente quando penalty == "elasticnet"
            },
        },
        "svc": {
            "estimator": svm.LinearSVC(
                random_state=random_state, class_weight="balanced"
            ),
            "grid": {
                "transf_features__te_job__transformer": [
                    "target_encoder",
                    "leave_one_out",
                    "one_hot_encoder",
                ],
                "transf_features__te_job__smoothing": uniform(loc=0, scale=2),
                "transf_features__te_job__sigma": uniform(loc=0, scale=1),
                "transf_features__bin_campaign__n_bins": [3, 4, 5],
                "transf_features__bin_campaign__encode": ["onehot", "ordinal"],
                "transf_features__bin_campaign__strategy": ["uniform", "quantile", "kmeans"],
                "transf_features__bin_previous__n_bins": [3, 4, 5],
                "transf_features__bin_previous__encode": ["onehot", "ordinal"],
                "transf_features__bin_previous__strategy": ["uniform", "quantile", "kmeans"],
                "clf__C": uniform(loc=0, scale=1),
                "clf__loss": ["hinge", "squared_hinge"],
            },
        },
        "dt": {
            "estimator": tree.DecisionTreeClassifier(
                random_state=random_state, class_weight="balanced"
            ),
            "grid": {
                "transf_features__te_job__transformer": [
                    "target_encoder",
                    "leave_one_out",
                    "one_hot_encoder",
                ],
                "transf_features__te_job__smoothing": uniform(loc=0, scale=2),
                "transf_features__te_job__sigma": uniform(loc=0, scale=1),
                "transf_features__bin_campaign__n_bins": [3, 4, 5],
                "transf_features__bin_campaign__encode": ["onehot", "ordinal"],
                "transf_features__bin_campaign__strategy": ["uniform", "quantile", "kmeans"],
                "transf_features__bin_previous__n_bins": [3, 4, 5],
                "transf_features__bin_previous__encode": ["onehot", "ordinal"],
                "transf_features__bin_previous__strategy": ["uniform", "quantile", "kmeans"],
                "clf__min_samples_split": uniform(loc=0, scale=0.1),
                "clf__min_samples_leaf": uniform(loc=0, scale=0.02),
            },
        },
        "rf": {
            "estimator": ensemble.RandomForestClassifier(
                random_state=random_state, n_jobs=-1
            ),
            "grid": {
                "transf_features__te_job__transformer": [
                    "target_encoder",
                    "leave_one_out",
                    "one_hot_encoder",
                ],
                "transf_features__te_job__smoothing": uniform(loc=0, scale=2),
                "transf_features__te_job__sigma": uniform(loc=0, scale=1),
                "transf_features__bin_campaign__n_bins": [3, 4, 5],
                "transf_features__bin_campaign__encode": ["onehot", "ordinal"],
                "transf_features__bin_campaign__strategy": ["uniform", "quantile", "kmeans"],
                "transf_features__bin_previous__n_bins": [3, 4, 5],
                "transf_features__bin_previous__encode": ["onehot", "ordinal"],
                "transf_features__bin_previous__strategy": ["uniform", "quantile", "kmeans"],
                "clf__n_estimators": [50, 100, 200, 300],
                "clf__min_samples_split": uniform(loc=0, scale=0.1),
                "clf__min_samples_leaf": uniform(loc=0, scale=0.02),
                "clf__max_depth": np.arange(5, 11),
                "clf__max_features": uniform(loc=0.5, scale=1),
                "clf__class_weight": ["balanced", "balanced_subsample"]
            },               
        },
    }

    return models_dict[model]
