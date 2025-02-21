from abc import abstractmethod, ABCMeta
from typing import Any, Literal, Optional
import numpy as np

type FloatMatrix = np.ndarray[Any, np.ndarray[Any, np.dtype[np.float64]]]
type FloatVector = np.ndarray[Any, np.dtype[np.float64]]

class LinearModel(metaclass=ABCMeta):
    def __init__(self, b: Optional[FloatVector] = None, intercept: float = 0, fit_intercept: bool = True):
        self.intercept_ = intercept
        self.fit_intercept_ = fit_intercept
        self.b_: Optional[FloatVector] = b
        self.x_mean: Optional[FloatVector] = None
        self.x_std: Optional[FloatVector] = None
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None

        

    @abstractmethod
    def fit():
        pass
    
    def _r2_score(
        self, 
        X: FloatMatrix, 
        y: FloatVector
    ) -> float:
        """
        From sklearn's documentation: Returns the coefficient R^2, defined as (1 - \\frac{u}{v}), where u is the residual sum of squares ((y_true - y_pred)** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum()
        """
        y_pred = self.predict(X)
        return 1 - (((y - y_pred) ** 2).sum() / ((y - np.mean(y)) ** 2).sum())
    
    def score(
        self, 
        X: FloatMatrix, 
        y: FloatVector,
        metric: Literal["r2"] = "r2"
    ):
        if metric == "r2":
            return self._r2_score(X, y)
        

    def rmse(
            self, 
            y_true: FloatVector, 
            y_pred: FloatVector
            ) -> float:
        """
        Calculate the root mean squared error between the true and predicted values. RMSE = \\sqrt{\\frac{u}{n} Where u is the residual sum of squares ((y_true - y_pred)** 2).sum() and n is the number of rows in y_true.
        """

        if y_true.shape[0] != y_pred.shape[0]:
            raise Exception("y_true and y_pred must have the same number of rows")
        
        return np.sqrt(np.mean(np.subtract(y_true, y_pred) ** 2))

    def u_rmse(
            self, 
            X: FloatMatrix, 
            y: FloatVector
        ) -> float:
        """
        Calculate the root mean squared error between the true and predicted values, with the features and true values. RMSE = \\sqrt{\\frac{u}{n} Where u is the residual sum of squares ((y_true - y_pred)** 2).sum() and n is the number of rows in y_true.
        """

        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same number of rows")
        
        y_pred = self.predict(X)
        return self.rmse(y, y_pred)
    
    def predict(
            self, 
            X: FloatMatrix
        ) -> FloatVector:
        """
            Predict a label from a given set of feature values.
        """

        if self.b_ is None:
            raise Exception("Model has not been fitted")

        if X.shape[1] != self.b_.shape[0]:
            raise Exception("The number of columns in X must match the number coefficients of the model")
        
        # This matrix multiplication is the same as the dot product between every set of feature values and the coefficients.
        return X @ self.b_ + self.intercept_
    

    def _center(
            self,
            X: FloatMatrix,
            y: FloatVector,
            skip_calculation: bool = False
        ) -> FloatVector:

        if not skip_calculation:
            self._save_metrics(X, y)

        X_normalized = (X - self.x_mean) 
        y_normalized = (y - self.y_mean) 
        return  X_normalized, y_normalized

    def _normalize(
        self, 
        X: FloatMatrix,
        y: FloatVector,
        skip_calculation: bool = False
    ) -> FloatMatrix:
        """
        Normalize the features using the mean and standard deviation of the features.
        """
        
        if not skip_calculation:
            self._save_metrics(X, y)

        return (X - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std


    
    def _pre_fit(
            self,
            X: FloatMatrix,
            y: FloatVector
        ) -> tuple[FloatMatrix, FloatVector]:

        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same number of rows")
        
        X_centered = X
        y_centered = y
        self._save_metrics(X, y)

        if self.fit_intercept_:
            X_centered, y_centered = self._center(X, y)


        return X_centered, y_centered
        
    
    def _save_metrics(
            self,
            X: FloatMatrix,
            y: FloatVector
        ) -> tuple[float, float]:
        """
        Save the mean and standard deviation of the features and labels to normalize the data later.
        """

        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)

    def _get_intercept(self):
        if not self.fit_intercept_ or self.b_ is None or self.x_mean is None or not self.y_mean:
            return 0
        
        return self.y_mean - (self.x_mean @ self.b_)