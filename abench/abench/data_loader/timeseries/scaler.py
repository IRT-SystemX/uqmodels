import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

class TemporalFeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_cls=StandardScaler, **scaler_kwargs):
        """
        Parameters
        ----------
        scaler_cls : class
            A scikit-learn scaler class (e.g., StandardScaler, MinMaxScaler)
        scaler_kwargs : dict
            Keyword arguments to pass to the scaler constructor
        """
        self.scaler_cls = scaler_cls
        self.scaler_kwargs = scaler_kwargs
        self.scalers_ = []

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_features = X.shape[2]
        self.scalers_ = []

        for i in range(n_features):
            scaler = self.scaler_cls(**self.scaler_kwargs)
            feature_data = X[:, :, i].reshape(-1, 1)
            scaler.fit(feature_data)
            self.scalers_.append(scaler)
        return self

    def transform(self, X):
        if not self.is_fitted():
            raise RuntimeError("Scaler is not fitted.")
        X = np.asarray(X)
        X_scaled = np.empty_like(X)

        for i, scaler in enumerate(self.scalers_):
            feature_data = X[:, :, i].reshape(-1, 1)
            scaled = scaler.transform(feature_data).reshape(X.shape[0], X.shape[1])
            X_scaled[:, :, i] = scaled

        return X_scaled

    def inverse_transform(self, X_scaled):
        if not self.is_fitted():
            raise RuntimeError("Scaler is not fitted.")
        X_scaled = np.asarray(X_scaled)
        X_orig = np.empty_like(X_scaled)

        for i, scaler in enumerate(self.scalers_):
            scaled_feature = X_scaled[:, :, i].reshape(-1, 1)
            inv = scaler.inverse_transform(scaled_feature).reshape(X_scaled.shape[0], X_scaled.shape[1])
            X_orig[:, :, i] = inv

        return X_orig

    def is_fitted(self):
        """
        Returns True if all internal scalers are fitted, False otherwise.
        """
        if not isinstance(self.scalers_, list) or len(self.scalers_) == 0:
            return False

        try:
            for scaler in self.scalers_:
                check_is_fitted(scaler)
            return True
        except NotFittedError:
            return False

    def save(self, filepath):
        """
        Saves the fitted TemporalFeatureScaler object to a file using pickle.

        Parameters
        ----------
        filepath : str
            Path to the file to save the object.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Loads a TemporalFeatureScaler object from a file.

        Parameters
        ----------
        filepath : str
            Path to the file to load the object from.

        Returns
        -------
        TemporalFeatureScaler
            The loaded object.
        """
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj

class SampleScaler(TemporalFeatureScaler):
    """
    Scaler vectorisé héritant de TemporalFeatureScaler,
    qui normalise chaque sample indépendamment sur N_step.

    mode : {"standard", "robust"}
        - "standard" : (x - mean) / std
        - "robust"   : (x - median) / (q2 - q1)
    
    q1, q2 : float
        Quantiles utilisés pour l'IQR si mode="robust".
    """
    def __init__(self, mode="standard", q1=0.05, q2=0.95):
        super().__init__(scaler_cls=None)
        if mode not in ["standard", "robust"]:
            raise ValueError("mode doit être 'standard' ou 'robust'")
        if not (0 <= q1 < q2 <= 1):
            raise ValueError("q1 et q2 doivent être entre 0 et 1 et q1 < q2")
        self.mode = mode
        self.q1 = q1
        self.q2 = q2

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("X doit être un tableau 3D [N_sample, N_step, N_features]")
        return self  # rien à stocker car calcul au transform

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("X doit être un tableau 3D [N_sample, N_step, N_features]")

        if self.mode == "standard":
            center = np.mean(X, axis=1, keepdims=True)
            scale = np.std(X, axis=1, keepdims=True)
        elif self.mode == "robust":
            center = np.median(X, axis=1, keepdims=True)
            q_high = np.quantile(X, self.q2, axis=1, keepdims=True)
            q_low = np.quantile(X, self.q1, axis=1, keepdims=True)
            scale = q_high - q_low

        scale = np.where(scale == 0, 1, scale)
        return (X - center) / scale

    def inverse_transform(self, X_scaled, X_ref):
        """
        Reconstitue les données originales à partir de X_scaled.
        Nécessite X_ref pour recalculer les statistiques.
        """
        X_scaled = np.asarray(X_scaled)
        X_ref = np.asarray(X_ref)

        if self.mode == "standard":
            center = np.mean(X_ref, axis=1, keepdims=True)
            scale = np.std(X_ref, axis=1, keepdims=True)
        elif self.mode == "robust":
            center = np.median(X_ref, axis=1, keepdims=True)
            q_high = np.quantile(X_ref, self.q2, axis=1, keepdims=True)
            q_low = np.quantile(X_ref, self.q1, axis=1, keepdims=True)
            scale = q_high - q_low

        scale = np.where(scale == 0, 1, scale)
        return X_scaled * scale + center