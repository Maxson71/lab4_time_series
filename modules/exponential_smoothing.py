"""
Модуль експоненціального згладжування для часових рядів
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class ExponentialSmoothingModels:
    """Клас для реалізації моделей експоненціального згладжування"""

    def __init__(self, data: pd.Series):
        """
        Parameters:
        -----------
        data : pd.Series
            Часовий ряд для обробки
        """
        self.data = data
        self.model = None
        self.fitted_values = None
        self.forecast_values = None

    def simple_exponential_smoothing(self, alpha: float = None,
                                     optimized: bool = True) -> Dict:
        """
        Просте експоненціальне згладжування (SES)
        Для рядів без тренду та сезонності

        Parameters:
        -----------
        alpha : float
            Параметр згладжування (0 < alpha < 1)
        optimized : bool
            Якщо True, alpha підбирається автоматично
        """
        try:
            model = ExponentialSmoothing(
                self.data,
                trend=None,
                seasonal=None
            )

            if optimized:
                fitted = model.fit(optimized=True)
                alpha_opt = fitted.params['smoothing_level']
            else:
                fitted = model.fit(smoothing_level=alpha, optimized=False)
                alpha_opt = alpha

            self.model = fitted
            self.fitted_values = fitted.fittedvalues

            return {
                'model': fitted,
                'alpha': alpha_opt,
                'fitted_values': self.fitted_values,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'sse': fitted.sse
            }
        except Exception as e:
            print(f"Error in SES: {e}")
            return None

    def double_exponential_smoothing(self, alpha: float = None,
                                     beta: float = None,
                                     trend: str = 'add',
                                     optimized: bool = True) -> Dict:
        """
        Подвійне експоненціальне згладжування (метод Хольта)
        Для рядів з трендом без сезонності

        Parameters:
        -----------
        alpha : float
            Параметр згладжування рівня
        beta : float
            Параметр згладжування тренду
        trend : str
            'add' (адитивний) або 'mul' (мультиплікативний)
        optimized : bool
            Автоматичний підбір параметрів
        """
        try:
            model = ExponentialSmoothing(
                self.data,
                trend=trend,
                seasonal=None
            )

            if optimized:
                fitted = model.fit(optimized=True)
            else:
                fitted = model.fit(
                    smoothing_level=alpha,
                    smoothing_trend=beta,
                    optimized=False
                )

            self.model = fitted
            self.fitted_values = fitted.fittedvalues

            return {
                'model': fitted,
                'alpha': fitted.params['smoothing_level'],
                'beta': fitted.params['smoothing_trend'],
                'fitted_values': self.fitted_values,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'sse': fitted.sse
            }
        except Exception as e:
            print(f"Error in DES: {e}")
            return None

    def triple_exponential_smoothing(self, trend: str = 'add',
                                     seasonal: str = 'add',
                                     seasonal_periods: int = 7,
                                     optimized: bool = True) -> Dict:
        """
        Потрійне експоненціальне згладжування (метод Хольта-Вінтерса)
        Для рядів з трендом та сезонністю

        Parameters:
        -----------
        trend : str
            'add' або 'mul'
        seasonal : str
            'add' або 'mul'
        seasonal_periods : int
            Період сезонності
        optimized : bool
            Автоматичний підбір параметрів
        """
        try:
            model = ExponentialSmoothing(
                self.data,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            )

            fitted = model.fit(optimized=optimized)

            self.model = fitted
            self.fitted_values = fitted.fittedvalues

            return {
                'model': fitted,
                'alpha': fitted.params['smoothing_level'],
                'beta': fitted.params.get('smoothing_trend', None),
                'gamma': fitted.params.get('smoothing_seasonal', None),
                'fitted_values': self.fitted_values,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'sse': fitted.sse
            }
        except Exception as e:
            print(f"Error in TES: {e}")
            return None

    def damped_trend_smoothing(self, trend: str = 'add',
                               damped_trend: bool = True,
                               optimized: bool = True) -> Dict:
        """
        Експоненціальне згладжування з затухаючим трендом

        Parameters:
        -----------
        trend : str
            'add' або 'mul'
        damped_trend : bool
            Увімкнути затухання тренду
        """
        try:
            model = ExponentialSmoothing(
                self.data,
                trend=trend,
                seasonal=None,
                damped_trend=damped_trend
            )

            fitted = model.fit(optimized=optimized)

            self.model = fitted
            self.fitted_values = fitted.fittedvalues

            return {
                'model': fitted,
                'alpha': fitted.params['smoothing_level'],
                'beta': fitted.params.get('smoothing_trend', None),
                'phi': fitted.params.get('damping_trend', None),
                'fitted_values': self.fitted_values,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'sse': fitted.sse
            }
        except Exception as e:
            print(f"Error in Damped: {e}")
            return None

    def auto_select_model(self) -> Dict:
        """
        Автоматичний вибір найкращої моделі за AIC
        Перевіряє всі комбінації параметрів
        """
        models_to_test = [
            ('SES', {'trend': None, 'seasonal': None, 'damped_trend': False}),
            ('Holt Linear', {'trend': 'add', 'seasonal': None, 'damped_trend': False}),
            ('Holt Damped', {'trend': 'add', 'seasonal': None, 'damped_trend': True}),
            ('Holt-Winters Add', {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7}),
            ('Holt-Winters Mul', {'trend': 'add', 'seasonal': 'mul', 'seasonal_periods': 7}),
            ('Holt-Winters Add-30', {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 30}),
        ]

        results = []
        best_aic = np.inf
        best_model = None
        best_name = None

        for name, params in models_to_test:
            try:
                model = ExponentialSmoothing(self.data, **params)
                fitted = model.fit(optimized=True)

                results.append({
                    'name': name,
                    'aic': fitted.aic,
                    'bic': fitted.bic,
                    'sse': fitted.sse
                })

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
                    best_name = name

            except Exception as e:
                print(f"Could not fit {name}: {e}")
                continue

        self.model = best_model
        self.fitted_values = best_model.fittedvalues if best_model else None

        return {
            'best_model': best_model,
            'best_name': best_name,
            'best_aic': best_aic,
            'all_results': pd.DataFrame(results).sort_values('aic') if results else None
        }

    def make_forecast(self, steps: int) -> pd.Series:
        """
        Прогнозування на steps кроків

        Parameters:
        -----------
        steps : int
            Кількість кроків для прогнозу
        """
        if self.model is None:
            raise ValueError("Спочатку потрібно підігнати модель!")

        forecast = self.model.forecast(steps=steps)
        self.forecast_values = forecast
        return forecast

    def plot_components(self, save_path: str = None):
        """Візуалізація компонентів моделі"""
        if self.model is None:
            raise ValueError("Спочатку потрібно підігнати модель!")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Оригінальні дані та згладжені
        axes[0].plot(self.data.index, self.data.values,
                     label='Оригінальні дані', alpha=0.6)
        axes[0].plot(self.fitted_values.index, self.fitted_values.values,
                     label='Згладжені значення', linewidth=2)
        axes[0].set_title('Оригінальні дані vs Згладжені', fontweight='bold')
        axes[0].set_ylabel('Значення')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Рівень (level)
        if hasattr(self.model, 'level'):
            axes[1].plot(self.model.level, label='Рівень (Level)', color='blue')
            axes[1].set_title('Компонента рівня', fontweight='bold')
            axes[1].set_ylabel('Рівень')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Тренд або залишки
        if hasattr(self.model, 'trend') and self.model.trend is not None:
            axes[2].plot(self.model.trend, label='Тренд', color='red')
            axes[2].set_title('Компонента тренду', fontweight='bold')
            axes[2].set_ylabel('Тренд')
        else:
            residuals = self.data - self.fitted_values
            axes[2].plot(residuals, label='Залишки', color='green', alpha=0.7)
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[2].set_title('Залишки', fontweight='bold')
            axes[2].set_ylabel('Залишки')

        axes[2].legend()
        axes[2].set_xlabel('Час')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class RegressionModels:
    """Клас для регресійних моделей"""

    def __init__(self, data: pd.Series):
        self.data = data
        self.X = None
        self.y = None
        self.model = None
        self.fitted_values = None

    def prepare_features(self, lags: int = 5,
                         include_time: bool = True,
                         include_ma: bool = True,
                         ma_windows: list = [7, 14, 30]):
        """
        Підготовка ознак для регресії

        Parameters:
        -----------
        lags : int
            Кількість лагових ознак
        include_time : bool
            Включити часові ознаки
        include_ma : bool
            Включити ковзні середні
        ma_windows : list
            Вікна для ковзних середніх
        """
        df = pd.DataFrame({'y': self.data})

        # Лагові ознаки
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['y'].shift(i)

        # Часові ознаки
        if include_time:
            df['time'] = np.arange(len(df))
            df['time_squared'] = df['time'] ** 2

        # Ковзні середні
        if include_ma:
            for window in ma_windows:
                df[f'ma_{window}'] = df['y'].rolling(window=window).mean()

        # Прирости
        df['diff_1'] = df['y'].diff()

        # Видалити NaN
        df = df.dropna()

        self.y = df['y']
        self.X = df.drop('y', axis=1)

        return self.X, self.y

    def linear_regression(self) -> Dict:
        """Лінійна регресія"""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(self.X, self.y)

        self.model = model
        self.fitted_values = pd.Series(
            model.predict(self.X),
            index=self.y.index
        )

        return {
            'model': model,
            'coefficients': dict(zip(self.X.columns, model.coef_)),
            'intercept': model.intercept_,
            'fitted_values': self.fitted_values,
            'r2': model.score(self.X, self.y)
        }

    def ridge_regression(self, alpha: float = 1.0) -> Dict:
        """Ridge регресія (L2 регуляризація)"""
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=alpha)
        model.fit(self.X, self.y)

        self.model = model
        self.fitted_values = pd.Series(
            model.predict(self.X),
            index=self.y.index
        )

        return {
            'model': model,
            'alpha': alpha,
            'coefficients': dict(zip(self.X.columns, model.coef_)),
            'intercept': model.intercept_,
            'fitted_values': self.fitted_values,
            'r2': model.score(self.X, self.y)
        }

    def lasso_regression(self, alpha: float = 1.0) -> Dict:
        """Lasso регресія (L1 регуляризація)"""
        from sklearn.linear_model import Lasso

        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(self.X, self.y)

        self.model = model
        self.fitted_values = pd.Series(
            model.predict(self.X),
            index=self.y.index
        )

        # Знайти ненульові коефіцієнти
        non_zero_features = [
            feature for feature, coef in zip(self.X.columns, model.coef_)
            if abs(coef) > 1e-10
        ]

        return {
            'model': model,
            'alpha': alpha,
            'coefficients': dict(zip(self.X.columns, model.coef_)),
            'intercept': model.intercept_,
            'fitted_values': self.fitted_values,
            'r2': model.score(self.X, self.y),
            'selected_features': non_zero_features,
            'n_features': len(non_zero_features)
        }

    def polynomial_regression(self, degree: int = 2) -> Dict:
        """Поліноміальна регресія"""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])

        model.fit(self.X, self.y)

        self.model = model
        self.fitted_values = pd.Series(
            model.predict(self.X),
            index=self.y.index
        )

        return {
            'model': model,
            'degree': degree,
            'fitted_values': self.fitted_values,
            'r2': model.score(self.X, self.y)
        }

    def forecast(self, steps: int) -> pd.Series:
        """Рекурсивне прогнозування"""
        if self.model is None:
            raise ValueError("Спочатку підігніть модель!")

        # Останні значення для ініціалізації
        last_values = self.data.tail(max(self.X.columns.str.extract(r'lag_(\d+)')[0].max(), 50))

        forecast_values = []
        current_data = list(last_values.values)

        for _ in range(steps):
            next_features = self._create_next_features(current_data)
            next_pred = self.model.predict([next_features])[0]

            forecast_values.append(next_pred)
            current_data.append(next_pred)

        return pd.Series(forecast_values)

    def _create_next_features(self, data):
        """Допоміжний метод для створення ознак"""
        features = []
        for col in self.X.columns:
            if 'lag' in col:
                lag_num = int(col.split('_')[1])
                features.append(data[-lag_num])
            elif 'time' in col:
                features.append(len(data))
            elif 'ma' in col:
                window = int(col.split('_')[1])
                features.append(np.mean(data[-window:]))
            else:
                features.append(0)
        return features
