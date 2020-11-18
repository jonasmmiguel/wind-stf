import warnings

import pandas as pd
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed

from sklearn.utils import _print_elapsed_time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import _name_estimators, Pipeline


class MakeStrictlyPositive(TransformerMixin, BaseEstimator):
    """Add constant to variable so that it only assumes positive values."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.offset_ = X.min(axis=0)
        return self

    def transform(self, X, y=None):
        return X + abs(self.offset_) + 1e-08

    def inverse_transform(self, X, y=None):
        return X - abs(self.offset_) - 1e-08


registered_transformers = {
    'get_quantile_equivalent_normal_dist': QuantileTransformer(
                                                output_distribution='normal',
                                                random_state=0,
                                            ),
    'make_strictly_positive': MakeStrictlyPositive(),
}


class Scaler(Pipeline):
    """
     Sklearn Pipeline inverse_transform returns a np.array with unnammed columns.
     This class stores at fit time the dataframe colnames as attribute and
    """

    def __init__(self, steps):
        # Initialize the same attributes as Superclass
        super().__init__(steps)

        # Initialize class-specific attributes
        self.columns = []

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.values

        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    @property
    def transform(self):
        """Apply transforms, and transform with the final estimator

        This also works where final estimator is ``None``: all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
        """
        # _final_estimator is None or has transform, otherwise attribute error
        # XXX: Handling the None case means we can't use if_delegate_has_method
        if self._final_estimator != 'passthrough':
            self._final_estimator.transform
        return self._transform

    def _transform(self, X):
        Xt = X
        for _, _, transform in self._iter():
            Xt = transform.transform(Xt)

        if isinstance(X, pd.DataFrame):
            Xt = pd.DataFrame(Xt, index=X.index, columns=self.columns)

        return Xt

    @property
    def inverse_transform(self):
        """Apply inverse transformations in reverse order

        All estimators in the pipeline must support ``inverse_transform``.

        Parameters
        ----------
        Xt : array-like of shape  (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
        """
        # raise AttributeError if necessary for hasattr behaviour
        # XXX: Handling the None case means we can't use if_delegate_has_method
        for _, _, transform in self._iter():
            transform.inverse_transform
        return self._inverse_transform

    def _inverse_transform(self, X):
        num_cols_X = X.shape[1]
        num_cols_dfskeleton = len(self.columns)
        is_X_partial = (num_cols_X < num_cols_dfskeleton)
        if isinstance(X, pd.DataFrame) and is_X_partial:
            dfskeleton = pd.DataFrame(data=np.nan, index=X.index, columns=self.columns)
            dfskeleton.update(X)
            Xt = dfskeleton.values
        else:
            Xt = X

        reverse_iter = reversed(list(self._iter()))
        for _, _, transform in reverse_iter:
            Xt = transform.inverse_transform(Xt)

        if self.columns != []:
            Xt = pd.DataFrame(data=Xt, index=X.index, columns=self.columns)

        return Xt


def make_pipeline(*steps):
    """Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators.

    See Also
    --------
    sklearn.pipeline.Pipeline : Class for creating a pipeline of
        transforms with a final estimator.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('gaussiannb', GaussianNB())])

    Returns
    -------
    p : Pipeline
    """
    return Scaler(_name_estimators(steps))
