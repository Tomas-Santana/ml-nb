import math
import numpy as np
from typing import Any, Self 


type FloatMatrix = np.ndarray[Any, np.ndarray[Any, np.dtype[np.float64]]]
type FloatVector = np.ndarray[Any, np.dtype[np.float64]]

class LinearRegression:
    def __init__(
                self, 
                b: FloatVector
            ) -> None:
        self.b = b
        pass
    
    def rmse(
            self, 
            y_true: FloatVector, 
            y_pred: FloatVector
            ) -> float:

        if y_true.shape[0] != y_pred.shape[0]:
            raise Exception("y_true and y_pred must have the same number of rows")
        
        return math.sqrt(np.mean(np.subtract(y_true, y_pred) ** 2))
    
    @classmethod
    def fit(
            cls, 
            X: FloatMatrix, 
            y: FloatVector
        ) -> Self:
        """
        Get the coefficients of the linear regression model.
        """
        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same number of rows")

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        b = np.linalg.inv(X.T @ X) @ X.T @ y
        return cls(b)
    
    def predict(
            self, 
            X: FloatMatrix
        ) -> FloatVector:
        """
            Predict a label from a given set of feature values
        """

        if X.shape[1] != self.b.shape[0] - 1:
            raise Exception("The number of columns in X must match the number coefficients of the model")

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        return X @ self.b
    
    def score(
            self, 
            X: FloatMatrix, 
            y: FloatVector
        ) -> float:
        """
        From sklearn's documentation: Returns the coefficient R^2, defined as (1 - \\frac{u}{v}), where u is the residual sum of squares ((y_true - y_pred)** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum()
        """
        y_pred = self.predict(X)
        return 1 - (((y - y_pred) ** 2).sum() / ((y - np.mean(y)) ** 2)).sum()

        
    


