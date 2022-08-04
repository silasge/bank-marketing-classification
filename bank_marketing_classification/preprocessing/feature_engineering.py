import category_encoders as ce
import numpy as np
from bank_marketing_classification.preprocessing import custom_preproc
from sklearn import compose, decomposition, pipeline, preprocessing

ordinalencoder_mapping = [
    {
        "col": "education",
        "mapping": {
            "unknown": 0,
            "illiterate": 0,
            "basic.4y": 1,
            "basic.6y": 1,
            "basic.9y": 1,
            "high.school": 2,
            "professional.course": 3,
            "university.degree": 4,
        },
    },
    {
        "col": "month",
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
            "dec": 12,
        },
    },
    {
        "col": "day_of_week",
        "mapping": {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5},
    },
]

pipe_pca = pipeline.Pipeline(
    [
        ("std", preprocessing.StandardScaler()),
        ("pca", decomposition.PCA(n_components=1)),
    ]
)


def _bin_age(x):
    return np.select([(x < 30), (x >= 30) & (x < 60), (x >= 60)], [1, 2, 3])


transform_features = compose.ColumnTransformer(
    [
        ("bin_age", preprocessing.FunctionTransformer(func=_bin_age), ["age"]),
        ("te_job", custom_preproc.GetEncoders(), ["job"]),
        ("oh_marital", preprocessing.OneHotEncoder(drop=["unknown"]), ["marital"]),
        (
            "oe_vars",
            ce.OrdinalEncoder(mapping=ordinalencoder_mapping),
            ["education", "month", "day_of_week"],
        ),
        ("oh_default", preprocessing.OneHotEncoder(drop=["yes"]), ["default"]),
        (
            "oh_housing_loan_contact",
            preprocessing.OneHotEncoder(),
            ["housing", "loan", "contact", "poutcome"],
        ),
        ("bin_campaign", preprocessing.KBinsDiscretizer(), ["campaign"]),
        ("bin_previous", preprocessing.KBinsDiscretizer(), ["previous"]),
        ("pca_econ_vars", pipe_pca, ["emp.var.rate", "nr.employed", "euribor3m"]),
        (
            "std_idx",
            preprocessing.StandardScaler(),
            ["cons.price.idx", "cons.conf.idx"],
        ),
    ]
)

transform_label = preprocessing.LabelBinarizer()
