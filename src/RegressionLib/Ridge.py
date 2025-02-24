import math
import numpy as np
from typing import Any, Optional, Self
from .LinearModel import LinearModel, FloatMatrix, FloatVector

class Ridge(LinearModel):
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.alpha_ = alpha
        super().__init__(None, 0, fit_intercept)
    
    def fit(
            self, 
            X: FloatMatrix, 
            y: FloatVector
        ) -> Self:
        """
        Get the coefficients of the linear regression model. It uses Ridge, b = (X^T X + alpha * I)^-1 X^T y
        """
        try:
            X_centered, y_centered = self._pre_fit(X, y)
        except Exception as e:
            raise e

        self.b_ = np.linalg.lstsq(X_centered.T @ X_centered + self.alpha_ * np.identity(X_centered.shape[1]), X_centered.T @ y_centered)[0]
        self.intercept_ = self._get_intercept()

        return self
    
        