"""Models: a tiny registry plus the xgb (default) and logreg wrappers.

All models implement `fit(X, y, X_val, y_val)`, `predict_proba`, and
`feature_importances`. Adding one is a small wrapper + `@register`.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type["ModelProtocol"]] = {}


@runtime_checkable
class ModelProtocol(Protocol):
    name: ClassVar[str]

    def fit(self, X, y, *, sample_weight=None, X_val=None, y_val=None) -> None: ...
    def predict_proba(self, X) -> np.ndarray: ...
    def feature_importances(self) -> Optional[np.ndarray]: ...


def register(cls: type[ModelProtocol]) -> type[ModelProtocol]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} missing class-level `name`")
    if cls.name in MODEL_REGISTRY:
        raise ValueError(f"model {cls.name!r} already registered")
    MODEL_REGISTRY[cls.name] = cls
    return cls


def build_model(name: str, **kwargs) -> ModelProtocol:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


@register
class XGBClassifierModel:
    """XGBoost binary classifier. Scales positive-class weight for imbalance and
    supports early stopping via an inner-validation slice of the train set.
    """

    name = "xgb"

    DEFAULTS = dict(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
    )

    def __init__(
        self,
        *,
        scale_pos_weight: Optional[float] = None,
        early_stopping_rounds: int = 50,
        random_state: int = 0,
        **kwargs,
    ) -> None:
        from xgboost import XGBClassifier

        params = {**self.DEFAULTS, **kwargs}
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        params["random_state"] = random_state
        self._early_stopping_rounds = early_stopping_rounds
        self._clf = XGBClassifier(**params)

    def fit(self, X, y, *, sample_weight=None, X_val=None, y_val=None) -> None:
        eval_set = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            eval_set = [(X_val, y_val)]
            try:
                self._clf.set_params(early_stopping_rounds=self._early_stopping_rounds)
            except Exception:
                pass  # older xgboost takes it as a fit-time arg instead
        self._clf.fit(X, y, sample_weight=sample_weight, eval_set=eval_set, verbose=False)

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(X)

    def feature_importances(self) -> Optional[np.ndarray]:
        return getattr(self._clf, "feature_importances_", None)


@register
class LogRegModel:
    """Logistic-regression baseline. Standardizes (sparse-safe) so coefficients
    are comparable.
    """

    name = "logreg"

    def __init__(
        self,
        *,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 0,
        C: float = 1.0,
        max_iter: int = 2000,
        **kwargs,
    ) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        class_weight = (
            "balanced" if scale_pos_weight is None else {0: 1.0, 1: float(scale_pos_weight)}
        )
        self._pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),  # sparse-safe
            ("clf", LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                solver="liblinear",
                **kwargs,
            )),
        ])

    def fit(self, X, y, *, sample_weight=None, X_val=None, y_val=None) -> None:
        self._pipe.fit(X, y, clf__sample_weight=sample_weight)

    def predict_proba(self, X) -> np.ndarray:
        return self._pipe.predict_proba(X)

    def feature_importances(self) -> Optional[np.ndarray]:
        coef = self._pipe.named_steps["clf"].coef_
        return np.abs(coef[0]) if coef is not None else None
