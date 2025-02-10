import math
import numpy as np
from typing import Any, Self 


type FloatMatrix = np.ndarray[Any, np.ndarray[Any, np.dtype[np.float64]]]
type FloatVector = np.ndarray[Any, np.dtype[np.float64]]

class LinearRegression:
    def __init__(
                self, 
                b: FloatVector,
                intersect: float = 0.0
            ) -> None:
        self.b_ = b
        self.intersect_ = intersect
        pass
    
    @classmethod
    def fit(
            cls, 
            X: FloatMatrix, 
            y: FloatVector
        ) -> Self:
        """
        Get the coefficients of the linear regression model. We use OLS, b = (X^T X)^{-1} X^T y to calculate the coefficients.
        """
        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same number of rows")

        # Add a column of ones to account for the intercept term
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        b = np.linalg.inv(X.T @ X) @ X.T @ y
        # Separate the intercept from the coefficients. Makes it easier to get predictions since adding a column of ones to the feature matrix is not necessary.
        return cls(b[1:], b[0])
    
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
        
        return math.sqrt(np.mean(np.subtract(y_true, y_pred) ** 2))
    
    def predict(
            self, 
            X: FloatMatrix
        ) -> FloatVector:
        """
            Predict a label from a given set of feature values.
        """

        if X.shape[1] != self.b_.shape[0]:
            raise Exception("The number of columns in X must match the number coefficients of the model")
        
        return X @ self.b_ + self.intersect_
    
    def score(
            self, 
            X: FloatMatrix, 
            y: FloatVector
        ) -> float:
        """
        From sklearn's documentation: Returns the coefficient R^2, defined as (1 - \\frac{u}{v}), where u is the residual sum of squares ((y_true - y_pred)** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum()
        """
        y_pred = self.predict(X)
        return 1 - (((y - y_pred) ** 2).sum() / ((y - np.mean(y)) ** 2).sum())

        
    


