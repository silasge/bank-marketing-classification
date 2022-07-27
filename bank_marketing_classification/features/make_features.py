import numpy as np
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

ordinalencoder_mapping = [
    {"col": "education",
     "mapping": {
         "unknown": 0,
         "illiterate": 0,
         "basic.4y": 1,
         "basic.6y": 1,
         "basic.9y": 1,
         "high.school": 2,
         "professional.course": 3,
         "university.degree": 4
     }},
    {"col": "month",
     "mapping": {
         "jan": 1,
         "feb": 2,
         "mar": 3,
         "apr": 4, 
         "may": 5, 
         "jun": 6, 
         "jul": 7, 
         "aug": 8, 
         "sep": 9, 
         "oct": 10, 
         "nov": 11, 
         "dec": 12
     }},
    {"col": "day_of_week",
     "mapping": {
         "mon": 1, 
         "tue": 2, 
         "wed": 3, 
         "thu": 4, 
         "fri": 5 
     }}
]

pipe_pca = Pipeline([
    ("std", StandardScaler()),
    ("pca", PCA(n_components=1))
])

def _bin_age(x):
    return np.select(
        [(x < 30), (x >= 30) & (x < 60), (x >= 60)],
        [1, 2, 3]
    )
    
def _bin_campaign(x):
    return np.select(
        [(x == 1), (x == 2), (x >= 3)],
        [1, 2, 3]
    )
    
def _bin_previous(x):
    return np.select(
        [(x == 0), (x == 1), (x >= 2)],
        [0, 1, 2]
    )
    
class JobTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer="one_hot_encoder", smoothing=1, sigma=0.05):
        self.transformer = transformer
        self.smoothing = smoothing
        self.sigma = sigma
        
    def fit(self, X, y=None):
        if self.transformer == "target_encoder":
            self.encoder = ce.TargetEncoder(smoothing=self.smoothing)
        elif self.transformer == "leave_one_out":
            self.encoder = ce.LeaveOneOutEncoder(random_state=42, sigma=self.sigma)
        elif self.transformer == "one_hot_encoder":
            self.encoder = ce.OneHotEncoder(use_cat_names=True)
        self.encoder.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.encoder.transform(X)

transform_features = ColumnTransformer([
    ("bin_age", FunctionTransformer(func=_bin_age), ["age"]),
    ("te_job", JobTransformer(), ["job"]),
    ("oh_marital", OneHotEncoder(drop=["unknown"]), ["marital"]),
    ("oe_vars", ce.OrdinalEncoder(mapping=ordinalencoder_mapping), ["education", "month", "day_of_week"]),
    ("oh_default", OneHotEncoder(drop=["yes"]), ["default"]),
    ("oh_housing_loan_contact", OneHotEncoder(), ["housing", "loan", "contact", "poutcome"]),
    ("bin_campaign", FunctionTransformer(func=_bin_campaign), ["campaign"]),
    ("bin_previous", FunctionTransformer(func=_bin_previous), ["previous"]),
    ("pca_econ_vars", pipe_pca, ["emp.var.rate", "nr.employed", "euribor3m"]),
    ("std_idx", StandardScaler(), ["cons.price.idx", "cons.conf.idx"])
])

transform_label = LabelBinarizer()
