import argparse
from scipy.stats import uniform
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from bank_marketing_classification.features.make_features import transform_features, transform_label
from sklearn.pipeline import Pipeline
import numpy as np
import joblib


_MODELS = {
    "lr": {
        "estimator": LogisticRegression(random_state=42, solver="saga", class_weight="balanced"),
        "grid": {
            "transf_features__te_job__transformer": ["target_encoder", "leave_one_out", "one_hot_encoder"], 
            "transf_features__te_job__smoothing": uniform(loc=0, scale=10), 
            "transf_features__te_job__sigma": uniform(loc=0, scale=1),  
            "clf__penalty": ["l1", "l2", "elasticnet", "none"],
            "clf__C": uniform(loc=0, scale=10),
            "clf__l1_ratio": uniform(loc=0, scale=1) # Usado somente quando penalty == "elasticnet"
        }
    },
    "svc": {
        "estimator": LinearSVC(random_state=42, class_weight="balanced"),
        "grid": {
            "transf_features__te_job__transformer": ["target_encoder", "leave_one_out", "one_hot_encoder"], 
            "transf_features__te_job__smoothing": uniform(loc=0, scale=2), 
            "transf_features__te_job__sigma": uniform(loc=0, scale=1), 
            "clf__C": uniform(loc=0, scale=10),
            "clf__penalty": ["l1", "l2"]
        }
    },
    "dt": {
        "estimator": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "grid": {
            "transf_features__te_job__transformer": ["target_encoder", "leave_one_out", "one_hot_encoder"], 
            "transf_features__te_job__smoothing": uniform(loc=0, scale=2), 
            "transf_features__te_job__sigma": uniform(loc=0, scale=1), 
            "clf__min_samples_split": uniform(loc=0, scale=1),
            "clf__min_samples_leaf": uniform(loc=0, scale=1)
        }
    },
    "rf": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "grid": {
            "transf_features__te_job__transformer": ["target_encoder", "leave_one_out", "one_hot_encoder"], 
            "transf_features__te_job__smoothing": uniform(loc=0, scale=2), 
            "transf_features__te_job__sigma": uniform(loc=0, scale=1), 
            "clf__n_estimators": [50, 100, 200, 300],
            "clf__min_samples_split": uniform(loc=0, scale=1),
            "clf__min_samples_leaf": uniform(loc=0, scale=1),
            "clf__max_depth": np.arange(1, 11),
            "clf__max_features": uniform(loc=0, scale=1),
            "clf__class_weight": ["balanced", "balanced_subsample"]      
        }
    }
}

def train_model(train_data: str, model: str, save_to: str, cv: int=10, scoring: str="roc_auc", n_iter: int=20):
    df = pd.read_csv(train_data)
    X = df.drop("y", axis=1)
    y = transform_label.fit_transform(df[["y"]]).ravel()
    
    model_pipe = Pipeline([
        ("transf_features", transform_features),
        ("clf", _MODELS[model]["estimator"])
    ])
    
    model_cv = RandomizedSearchCV(
        model_pipe,
        _MODELS[model]["grid"],
        cv=cv,
        random_state=42,
        n_jobs=-1,
        scoring=scoring,
        n_iter=n_iter
    )
    
    model_cv.fit(X, y)
    joblib.dump(value=model_cv, filename=save_to)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, dest="train_data")
    parser.add_argument("--model", type=str, dest="model")
    parser.add_argument("--save_to", type=str, dest="save_to")
    parser.add_argument("--cv", type=int, default=10, dest="cv")
    parser.add_argument("--scoring", type=str, default="roc_auc", dest="scoring")
    parser.add_argument("--n_iter", type=int, default=20, dest="n_iter")
    args = parser.parse_args()
    train_model(train_data=args.train_data, 
                model=args.model, 
                save_to=args.save_to,
                cv=args.cv,
                scoring=args.scoring,
                n_iter=args.n_iter)