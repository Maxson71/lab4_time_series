"""
Модуль генерації синтетичних часових рядів
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


class SyntheticTimeSeriesGenerator:
    """Клас для генерації синтетичних часових рядів"""

    def __init__(self, reference_data: pd.DataFrame, seed: int = 42):
        """
        Parameters:
        -----------
        reference_data : pd.DataFrame
            Реальні дані для копіювання властивостей
        seed : int
            Seed для відтворюваності
        """
        self.reference_data = reference_data
        self.seed = seed
        np.random.seed(seed)

    def generate_with_trend(self,
                            n: int,
                            trend_type: str = 'linear',
                            trend_params: Dict = None,
                            noise_params: Dict = None) -> pd.DataFrame:
        """
        Генерація TS з трендом

        Parameters:
        -----------
        n : int
            Кількість точок
        trend_type : str
            'linear', 'quadratic', 'exponential'
        trend_params : Dict
            Параметри тренду (коефіцієнти)
        noise_params : Dict
            Параметри шуму (mean, std)
        """
        t = np.arange(n)

        # Тренд
        if trend_type == 'linear':
            a = trend_params.get('a', 1.0)
            b = trend_params.get('b', 0.0)
            trend = a * t + b
        elif trend_type == 'quadratic':
            a = trend_params.get('a', 0.001)
            b = trend_params.get('b', 1.0)
            c = trend_params.get('c', 0.0)
            trend = a * t ** 2 + b * t + c
        elif trend_type == 'exponential':
            a = trend_params.get('a', 1.0)
            b = trend_params.get('b', 0.001)
            trend = a * np.exp(b * t)
        else:
            trend = np.zeros(n)

        # Шум
        mu = noise_params.get('mean', 0.0)
        sigma = noise_params.get('std', 1.0)
        noise = np.random.normal(mu, sigma, n)

        # Комбінація
        y = trend + noise

        return pd.DataFrame({
            't': t,
            'value': y,
            'trend': trend,
            'noise': noise
        })

    def generate_with_seasonality(self,
                                  n: int,
                                  seasonal_periods: list = [7, 30, 365],
                                  seasonal_amplitudes: list = None,
                                  noise_std: float = 1.0) -> pd.DataFrame:
        """
        Генерація TS з сезонністю

        Parameters:
        -----------
        n : int
            Кількість точок
        seasonal_periods : list
            Періоди сезонності
        seasonal_amplitudes : list
            Амплітуди для кожного періоду
        noise_std : float
            Стандартне відхилення шуму
        """
        t = np.arange(n)

        if seasonal_amplitudes is None:
            seasonal_amplitudes = [1.0] * len(seasonal_periods)

        # Сезонні компоненти
        seasonal = np.zeros(n)
        for period, amplitude in zip(seasonal_periods, seasonal_amplitudes):
            seasonal += amplitude * np.sin(2 * np.pi * t / period)

        # Шум
        noise = np.random.normal(0, noise_std, n)

        # Комбінація
        y = seasonal + noise

        return pd.DataFrame({
            't': t,
            'value': y,
            'seasonal': seasonal,
            'noise': noise
        })

    def generate_arma(self, n: int, ar_params: list = [0.5], ma_params: list = [0.3],
                      noise_std: float = 1.0) -> pd.DataFrame:
        """
        Генерація ARMA процесу

        Parameters:
        -----------
        n : int
            Кількість точок
        ar_params : list
            Параметри AR (autoregressive)
        ma_params : list
            Параметри MA (moving average)
        noise_std : float
            Стандартне відхилення білого шуму
        """
        from statsmodels.tsa.arima_process import ArmaProcess

        # ARMA процес
        ar = np.r_[1, -np.array(ar_params)]
        ma = np.r_[1, np.array(ma_params)]

        arma_process = ArmaProcess(ar, ma)
        y = arma_process.generate_sample(n, scale=noise_std)

        return pd.DataFrame({
            't': np.arange(n),
            'value': y
        })

    def generate_from_properties(self, properties: Dict, n: int = None) -> pd.DataFrame:
        """
        Генерація на основі виявлених властивостей реальних даних

        Parameters:
        -----------
        properties : Dict
            Словник з властивостями (mean, std, skew, kurtosis, trend, etc.)
        n : int
            Кількість точок (якщо None - як у reference_data)
        """
        if n is None:
            n = len(self.reference_data)

        t = np.arange(n).reshape(-1, 1)

        # Тренд
        trend_type = properties.get('trend_type', 'linear')
        if trend_type == 'quadratic' and 'trend_coeffs' in properties:
            coeffs = properties['trend_coeffs']
            trend = coeffs[0] + coeffs[1] * t.ravel() + coeffs[2] * (t.ravel() ** 2)
        elif 'trend_coeffs' in properties:
            coeffs = properties['trend_coeffs']
            trend = coeffs[0] + coeffs[1] * t.ravel()
        else:
            trend = np.zeros(n)

        # Шум з заданими властивостями
        mu = properties.get('noise_mean', 0.0)
        sigma = properties.get('noise_std', 1.0)

        # Генерація з урахуванням skewness та kurtosis
        if 'skew' in properties and abs(properties['skew']) > 0.5:
            # Використання skewnorm для асиметричного розподілу
            a = properties['skew']
            noise = stats.skewnorm.rvs(a, loc=mu, scale=sigma, size=n)
        else:
            noise = np.random.normal(mu, sigma, n)

        y = trend + noise

        return pd.DataFrame({
            't': np.arange(n),
            'value': y,
            'trend': trend,
            'noise': noise
        })

    def verify_similarity(self, synthetic_data: pd.DataFrame,
                          real_column: str = 'rate') -> Dict:
        """
        Верифікація схожості синтетичних та реальних даних

        Returns:
        --------
        Dict з метриками схожості
        """
        real = self.reference_data[real_column].values
        synthetic = synthetic_data['value'].values

        # Статистичне порівняння
        real_mean = np.mean(real)
        synth_mean = np.mean(synthetic)
        real_std = np.std(real, ddof=1)
        synth_std = np.std(synthetic, ddof=1)

        # KS-тест (Kolmogorov-Smirnov)
        ks_stat, ks_pval = stats.ks_2samp(real, synthetic)

        # T-тест для середніх
        t_stat, t_pval = stats.ttest_ind(real, synthetic)

        return {
            'mean_difference': abs(real_mean - synth_mean),
            'std_difference': abs(real_std - synth_std),
            'mean_relative_error': abs(real_mean - synth_mean) / real_mean,
            'std_relative_error': abs(real_std - synth_std) / real_std,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'ks_similar': ks_pval > 0.05,  # Не відхиляємо гіпотезу про схожість
            't_statistic': t_stat,
            't_pvalue': t_pval,
            't_similar': t_pval > 0.05
        }
