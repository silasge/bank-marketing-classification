import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin

class GetEncoders(BaseEstimator, TransformerMixin):
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