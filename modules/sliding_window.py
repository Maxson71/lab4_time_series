"""
Модуль алгоритмів ковзного вікна для обробки часових рядів
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class SlidingWindowAlgorithms:
    """Клас для реалізації алгоритмів ковзного вікна"""

    def __init__(self, data: pd.Series):
        """
        Parameters:
        -----------
        data : pd.Series
            Часовий ряд для обробки
        """
        self.data = data
        self.fitted_values = None
        self.model = None

    def simple_moving_average(self, window: int) -> pd.Series:
        """
        Проста ковзна середня (SMA)

        Parameters:
        -----------
        window : int
            Розмір вікна
        """
        return self.data.rolling(window=window, min_periods=1).mean()

    def weighted_moving_average(self, window: int) -> pd.Series:
        """
        Зважена ковзна середня (WMA)
        Більша вага надається останнім значенням

        Parameters:
        -----------
        window : int
            Розмір вікна
        """
        weights = np.arange(1, window + 1)

        def weighted_mean(x):
            if len(x) < window:
                w = np.arange(1, len(x) + 1)
                return np.sum(x * w) / np.sum(w)
            return np.sum(x * weights) / np.sum(weights)

        return self.data.rolling(window=window, min_periods=1).apply(weighted_mean, raw=True)

    def exponential_moving_average(self, span: int) -> pd.Series:
        """
        Експоненціальна ковзна середня (EMA)

        Parameters:
        -----------
        span : int
            Період згладжування
        """
        return self.data.ewm(span=span, adjust=False).mean()

    def double_exponential_smoothing(self, alpha: float = 0.3, beta: float = 0.1) -> pd.Series:
        """
        Подвійне експоненціальне згладжування (метод Хольта)

        Parameters:
        -----------
        alpha : float
            Параметр згладжування рівня
        beta : float
            Параметр згладжування тренду
        """
        result = np.zeros(len(self.data))
        level = self.data.iloc[0]
        trend = self.data.iloc[1] - self.data.iloc[0]

        result[0] = level

        for i in range(1, len(self.data)):
            prev_level = level
            level = alpha * self.data.iloc[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            result[i] = level

        return pd.Series(result, index=self.data.index)

    def fit_arma(self, order: Tuple[int, int] = (1, 1)) -> Dict:
        """
        Підгонка ARMA(p,q) моделі

        Parameters:
        -----------
        order : tuple
            (p, q) - порядки AR та MA компонент
        """
        try:
            # Видалення тренду для стаціонарності
            diff_data = self.data.diff().dropna()

            model = ARIMA(diff_data, order=(order[0], 0, order[1]))
            fitted = model.fit()

            self.model = fitted

            # Відновлення оригінального масштабу
            predictions = fitted.fittedvalues
            self.fitted_values = self.data.iloc[0] + predictions.cumsum()

            return {
                'model': fitted,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'params': fitted.params,
                'fitted_values': self.fitted_values,
                'residuals': self.data.iloc[1:] - self.fitted_values
            }
        except Exception as e:
            print(f"Error fitting ARMA: {e}")
            return None

    def fit_arima(self, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict:
        """
        Підгонка ARIMA(p,d,q) моделі

        Parameters:
        -----------
        order : tuple
            (p, d, q) - порядки AR, інтегрування та MA
        """
        try:
            model = ARIMA(self.data, order=order)
            fitted = model.fit()

            self.model = fitted
            self.fitted_values = fitted.fittedvalues

            return {
                'model': fitted,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'params': fitted.params,
                'fitted_values': self.fitted_values,
                'residuals': fitted.resid
            }
        except Exception as e:
            print(f"Error fitting ARIMA: {e}")
            return None

    def fit_sarima(self, order: Tuple[int, int, int] = (1, 1, 1),
                   seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7)) -> Dict:
        """
        Підгонка SARIMA моделі (з сезонністю)

        Parameters:
        -----------
        order : tuple
            (p, d, q) - несезонні порядки
        seasonal_order : tuple
            (P, D, Q, s) - сезонні порядки та період
        """
        try:
            model = SARIMAX(self.data, order=order, seasonal_order=seasonal_order)
            fitted = model.fit(disp=False)

            self.model = fitted
            self.fitted_values = fitted.fittedvalues

            return {
                'model': fitted,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'params': fitted.params,
                'fitted_values': self.fitted_values,
                'residuals': fitted.resid
            }
        except Exception as e:
            print(f"Error fitting SARIMA: {e}")
            return None

    def auto_arima_selection(self, max_p: int = 5, max_q: int = 5,
                             max_d: int = 2) -> Dict:
        """
        Автоматичний вибір найкращої ARIMA моделі за критерієм AIC

        Parameters:
        -----------
        max_p, max_q, max_d : int
            Максимальні значення параметрів для перебору
        """
        best_aic = np.inf
        best_order = None
        best_model = None

        results = []

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue

                    try:
                        model = ARIMA(self.data, order=(p, d, q))
                        fitted = model.fit()

                        results.append({
                            'order': (p, d, q),
                            'aic': fitted.aic,
                            'bic': fitted.bic
                        })

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_model = fitted

                    except:
                        continue

        self.model = best_model
        self.fitted_values = best_model.fittedvalues if best_model else None

        return {
            'best_order': best_order,
            'best_aic': best_aic,
            'best_model': best_model,
            'all_results': pd.DataFrame(results).sort_values('aic')
        }

    def forecast(self, steps: int) -> pd.Series:
        """
        Прогнозування на steps кроків вперед

        Parameters:
        -----------
        steps : int
            Кількість кроків для прогнозування
        """
        if self.model is None:
            raise ValueError("Спочатку потрібно підігнати модель!")

        forecast = self.model.forecast(steps=steps)
        return forecast

    def get_forecast_with_intervals(self, steps: int, alpha: float = 0.05) -> Dict:
        """
        Прогнозування з довірчими інтервалами

        Parameters:
        -----------
        steps : int
            Кількість кроків
        alpha : float
            Рівень значущості (0.05 для 95% інтервалу)
        """
        if self.model is None:
            raise ValueError("Спочатку потрібно підігнати модель!")

        forecast_obj = self.model.get_forecast(steps=steps)
        forecast_mean = forecast_obj.predicted_mean
        forecast_ci = forecast_obj.conf_int(alpha=alpha)

        return {
            'forecast': forecast_mean,
            'lower_bound': forecast_ci.iloc[:, 0],
            'upper_bound': forecast_ci.iloc[:, 1]
        }


class ModelEvaluator:
    """Клас для оцінювання якості моделей"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Розрахунок метрик якості прогнозування

        Returns:
        --------
        Dict з метриками: MAE, MSE, RMSE, MAPE, R²
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # R² score
        r2 = r2_score(y_true, y_pred)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }

    @staticmethod
    def plot_forecast(train_data: pd.Series, test_data: pd.Series,
                      forecast: pd.Series, title: str = "Прогноз",
                      save_path: str = None):
        """Візуалізація прогнозування"""
        plt.figure(figsize=(14, 6))

        plt.plot(train_data.index, train_data.values,
                 label='Тренувальні дані', color='blue', linewidth=1.5)
        plt.plot(test_data.index, test_data.values,
                 label='Тестові дані', color='green', linewidth=1.5)
        plt.plot(forecast.index, forecast.values,
                 label='Прогноз', color='red', linewidth=2, linestyle='--')

        plt.axvline(x=train_data.index[-1], color='gray',
                    linestyle=':', alpha=0.7, label='Початок прогнозу')

        plt.xlabel('Дата', fontsize=11)
        plt.ylabel('Курс EUR/UAH (грн)', fontsize=11)
        plt.title(title, fontsize=13, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_residuals(residuals: pd.Series, save_path: str = None):
        """Аналіз залишків"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Графік залишків у часі
        axes[0, 0].plot(residuals, linewidth=1)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Залишки у часі')
        axes[0, 0].set_xlabel('Час')
        axes[0, 0].set_ylabel('Залишки')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Гістограма залишків
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Розподіл залишків')
        axes[0, 1].set_xlabel('Залишки')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q plot
        from scipy import stats
        stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. ACF залишків
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals.dropna(), lags=40, ax=axes[1, 1])
        axes[1, 1].set_title('Автокореляція залишків')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
