import math
from typing import Optional
import numpy as np
from typing import Any, Self 
from .LinearModel import LinearModel, FloatMatrix, FloatVector

class LinearRegression(LinearModel):
    def __init__(
                self, 
                b: Optional[FloatVector] = None,
                intercept: float = 0.0,
                fit_intercept: bool = True
            ) -> None:
        super().__init__(b, intercept, fit_intercept)
        pass
    
    def fit(
            self, 
            X: FloatMatrix, 
            y: FloatVector
        ) -> Self:
        """
        Get the coefficients of the linear regression model. It uses OLS, b = (X^T X)^{-1} X^T y to calculate the coefficients.
        """
        try:
            X_centered, y_centered = self._pre_fit(X, y)
        except Exception as e:
            raise e
        
        self.b_ = np.linalg.lstsq(X_centered.T @ X_centered, X_centered.T @ y_centered, rcond=None)[0]
        self.intercept_ = self._get_intercept()
        return self

    


