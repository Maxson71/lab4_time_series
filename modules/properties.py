"""
Модуль виявлення властивостей часових рядів
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from typing import Dict, Tuple


class TimeSeriesProperties:
    """Клас для виявлення властивостей часових рядів"""

    def __init__(self, data: pd.Series):
        self.data = data

    def compute_statistics(self) -> Dict:
        """Базові статистичні показники"""
        return {
            'count': int(len(self.data)),
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data, ddof=1)),
            'var': float(np.var(self.data, ddof=1)),
            'min': float(np.min(self.data)),
            'q25': float(np.quantile(self.data, 0.25)),
            'median': float(np.median(self.data)),
            'q75': float(np.quantile(self.data, 0.75)),
            'max': float(np.max(self.data)),
            'skew': float(stats.skew(self.data, bias=False)),
            'kurtosis': float(stats.kurtosis(self.data, fisher=True, bias=False)),
            'cv': float(np.std(self.data, ddof=1) / np.mean(self.data))  # Коефіцієнт варіації
        }

    def test_stationarity(self) -> Dict:
        """
        Тестування стаціонарності
        - ADF тест (Augmented Dickey-Fuller)
        - KPSS тест
        """
        # ADF тест
        adf_result = adfuller(self.data.dropna(), autolag='AIC')

        # KPSS тест
        kpss_result = kpss(self.data.dropna(), regression='c', nlags='auto')

        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_is_stationary': adf_result[1] < 0.05,  # p < 0.05 => стаціонарний
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'kpss_is_stationary': kpss_result[1] > 0.05  # p > 0.05 => стаціонарний
        }

    def compute_autocorrelation(self, nlags: int = 40) -> Dict:
        """Обчислення автокореляції (ACF) та часткової автокореляції (PACF)"""
        acf_vals = acf(self.data.dropna(), nlags=nlags, fft=False)
        pacf_vals = pacf(self.data.dropna(), nlags=nlags)

        return {
            'acf': acf_vals,
            'pacf': pacf_vals,
            'acf_lag1': acf_vals[1] if len(acf_vals) > 1 else None
        }

    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict:
        """
        Виявлення викидів

        Parameters:
        -----------
        method : str
            'iqr' (міжквартильний розмах) або 'zscore'
        threshold : float
            Порогове значення (1.5 для IQR, 3 для z-score)
        """
        if method == 'iqr':
            Q1 = np.quantile(self.data, 0.25)
            Q3 = np.quantile(self.data, 0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (self.data < lower_bound) | (self.data > upper_bound)
        else:  # zscore
            z_scores = np.abs(stats.zscore(self.data))
            outliers = z_scores > threshold

        return {
            'outlier_count': int(np.sum(outliers)),
            'outlier_percentage': float(100 * np.sum(outliers) / len(self.data)),
            'outlier_indices': np.where(outliers)[0].tolist()
        }

    def analyze_trend(self, data_with_trend: pd.DataFrame) -> Dict:
        """Аналіз трендових властивостей"""
        if 'trend_kind' not in data_with_trend.columns:
            return {}

        residuals = data_with_trend['residual']

        # Тест на нормальність залишків (Shapiro-Wilk)
        shapiro_stat, shapiro_p = stats.shapiro(residuals.dropna()[:5000])  # Обмеження через розмір

        # Тест на гомоскедастичність (рівність дисперсії)
        # Розділимо на дві частини та порівняємо дисперсії
        mid = len(residuals) // 2
        var1 = np.var(residuals[:mid], ddof=1)
        var2 = np.var(residuals[mid:], ddof=1)

        return {
            'trend_type': data_with_trend['trend_kind'].iloc[0],
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals, ddof=1)),
            'shapiro_statistic': float(shapiro_stat),
            'shapiro_pvalue': float(shapiro_p),
            'is_normal_residuals': shapiro_p > 0.05,
            'variance_ratio': float(max(var1, var2) / min(var1, var2)),
            'is_homoscedastic': (max(var1, var2) / min(var1, var2)) < 2.0
        }

    def full_analysis(self) -> Dict:
        """Повний аналіз властивостей"""
        results = {}
        results['statistics'] = self.compute_statistics()
        results['stationarity'] = self.test_stationarity()
        results['autocorrelation'] = self.compute_autocorrelation()
        results['outliers'] = self.detect_outliers()

        return results
