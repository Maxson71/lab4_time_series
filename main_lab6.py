import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from modules.data_loader import TimeSeriesLoader
from modules.exponential_smoothing import ExponentialSmoothingModels, RegressionModels
from modules.sliding_window import ModelEvaluator


def main():
    valcode = "EUR"
    output_dir = Path(valcode) / "lab6"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ЛАБОРАТОРНА РОБОТА №6")
    print("Експоненціальне згладжування та регресійні моделі")
    print("=" * 70)

    # 1. Завантаження даних
    print("\n" + "=" * 70)
    print("ЕТАП 1: Завантаження та дослідження даних")
    print("=" * 70)

    loader = TimeSeriesLoader(valcode=valcode)
    df = loader.load_data()
    ts_data = df.set_index('date')['rate']

    print(f"✓ Завантажено {len(ts_data)} спостережень")
    print(f"  Період: {ts_data.index[0]} - {ts_data.index[-1]}")

    # Розділення даних
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]

    print(f"\n✓ Розділення: train={len(train_data)}, test={len(test_data)}")

    # 2. ГРУПА ВИМОГ 1: Експоненціальне згладжування
    print("\n" + "=" * 70)
    print("ГРУПА ВИМОГ 1: ЕКСПОНЕНЦІАЛЬНЕ ЗГЛАДЖУВАННЯ")
    print("=" * 70)

    es_models = ExponentialSmoothingModels(train_data)

    # Автоматичний вибір моделі
    print("\n[1] Автоматичний вибір моделі...")
    auto_result = es_models.auto_select_model()

    print(f"  Найкраща модель: {auto_result['best_name']}")
    print(f"  AIC: {auto_result['best_aic']:.2f}")
    print("\n  Порівняння моделей:")
    print(auto_result['all_results'])

    # Тестування різних моделей
    models_results = {}

    print("\n[2] Просте експоненціальне згладжування (SES)...")
    ses_result = es_models.simple_exponential_smoothing(optimized=True)
    if ses_result:
        print(f"  α = {ses_result['alpha']:.4f}, AIC = {ses_result['aic']:.2f}")
        models_results['SES'] = ses_result

    print("\n[3] Подвійне згладжування (Holt)...")
    des_result = es_models.double_exponential_smoothing(trend='add', optimized=True)
    if des_result:
        print(f"  α = {des_result['alpha']:.4f}, β = {des_result['beta']:.4f}")
        print(f"  AIC = {des_result['aic']:.2f}")
        models_results['Holt'] = des_result

    print("\n[4] Затухаючий тренд...")
    damped_result = es_models.damped_trend_smoothing(optimized=True)
    if damped_result:
        print(f"  α = {damped_result['alpha']:.4f}, β = {damped_result['beta']:.4f}")
        print(f"  φ = {damped_result['phi']:.4f}")
        print(f"  AIC = {damped_result['aic']:.2f}")
        models_results['Damped'] = damped_result

    print("\n[5] Хольт-Вінтерс (адитивна сезонність, період=7)...")
    hw_result = es_models.triple_exponential_smoothing(
        trend='add', seasonal='add', seasonal_periods=7, optimized=True
    )
    if hw_result:
        print(f"  α = {hw_result['alpha']:.4f}, β = {hw_result['beta']:.4f}, γ = {hw_result['gamma']:.4f}")
        print(f"  AIC = {hw_result['aic']:.2f}")
        models_results['HW-7'] = hw_result

        # Прогнозування
        print("\n" + "=" * 70)
        print("ПРОГНОЗУВАННЯ")
        print("=" * 70)

        test_length = len(test_data)
        forecast_results = {}

        for name, result in models_results.items():
            print(f"\n[{name}] Прогноз на {test_length} кроків...")

            try:
                # Створити нову модель з тими самими параметрами
                es_temp = ExponentialSmoothingModels(train_data)
                es_temp.model = result['model']
                es_temp.fitted_values = result['fitted_values']

                forecast = es_temp.make_forecast(steps=test_length)

                # Створити правильний індекс для прогнозу
                last_date = train_data.index[-1]
                forecast_index = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=test_length,
                    freq='D'
                )
                forecast = pd.Series(forecast.values, index=forecast_index)

                # Метрики
                metrics = ModelEvaluator.calculate_metrics(
                    test_data.values,
                    forecast.values
                )

                forecast_results[name] = {
                    'forecast': forecast,
                    'metrics': metrics
                }

                print(f"  RMSE: {metrics['RMSE']:.4f}")
                print(f"  MAPE: {metrics['MAPE']:.2f}%")
                print(f"  R²: {metrics['R2']:.4f}")

            except Exception as e:
                print(f"  ✗ Помилка при прогнозуванні: {e}")
                continue

    # Візуалізація експоненціального згладжування
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(list(forecast_results.items())[:4]):
        ax = axes[idx]

        ax.plot(train_data.index, train_data.values,
                label='Тренувальні', color='blue', alpha=0.6)
        ax.plot(test_data.index, test_data.values,
                label='Тестові', color='green', alpha=0.6)
        ax.plot(result['forecast'].index, result['forecast'].values,
                label='Прогноз', color='red', linewidth=2, linestyle='--')

        ax.axvline(x=train_data.index[-1], color='gray', linestyle=':', alpha=0.7)

        metrics_text = f"RMSE: {result['metrics']['RMSE']:.3f}\nMAPE: {result['metrics']['MAPE']:.2f}%"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Курс EUR/UAH')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exponential_smoothing_forecasts.png", dpi=300)
    plt.close()
    print("\n✓ Візуалізація експоненціального згладжування збережена")

    # 3. ГРУПА ВИМОГ 2: Регресійні моделі
    print("\n" + "=" * 70)
    print("ГРУПА ВИМОГ 2: РЕГРЕСІЙНІ МОДЕЛІ")
    print("=" * 70)

    reg_models = RegressionModels(train_data)

    print("\n[1] Підготовка ознак...")
    X_train, y_train = reg_models.prepare_features(
        lags=5,
        include_time=True,
        include_ma=True,
        ma_windows=[7, 14, 30]
    )
    print(f"  Кількість ознак: {X_train.shape[1]}")
    print(f"  Ознаки: {list(X_train.columns)}")

    regression_results = {}

    print("\n[2] Лінійна регресія...")
    lr_result = reg_models.linear_regression()
    metrics_lr = ModelEvaluator.calculate_metrics(y_train.values, lr_result['fitted_values'].values)
    regression_results['Linear'] = {'result': lr_result, 'metrics': metrics_lr}
    print(f"  R²: {lr_result['r2']:.4f}")
    print(f"  RMSE: {metrics_lr['RMSE']:.4f}")

    print("\n[3] Ridge регресія (α=1.0)...")
    ridge_result = reg_models.ridge_regression(alpha=1.0)
    metrics_ridge = ModelEvaluator.calculate_metrics(y_train.values, ridge_result['fitted_values'].values)
    regression_results['Ridge'] = {'result': ridge_result, 'metrics': metrics_ridge}
    print(f"  R²: {ridge_result['r2']:.4f}")
    print(f"  RMSE: {metrics_ridge['RMSE']:.4f}")

    print("\n[4] Lasso регресія (α=0.1)...")
    lasso_result = reg_models.lasso_regression(alpha=0.1)
    metrics_lasso = ModelEvaluator.calculate_metrics(y_train.values, lasso_result['fitted_values'].values)
    regression_results['Lasso'] = {'result': lasso_result, 'metrics': metrics_lasso}
    print(f"  R²: {lasso_result['r2']:.4f}")
    print(f"  RMSE: {metrics_lasso['RMSE']:.4f}")
    print(f"  Відібрано ознак: {lasso_result['n_features']}/{X_train.shape[1]}")
    print(f"  Ознаки: {lasso_result['selected_features']}")

    # Візуалізація регресій
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(list(regression_results.items())[:4]):
        ax = axes[idx]

        ax.plot(y_train.index, y_train.values,
                label='Фактичні', color='blue', alpha=0.6)
        ax.plot(data['result']['fitted_values'].index,
                data['result']['fitted_values'].values,
                label='Прогноз моделі', color='red', alpha=0.8)

        metrics_text = f"R²: {data['result']['r2']:.4f}\nRMSE: {data['metrics']['RMSE']:.3f}"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        ax.set_title(f'{name} Regression', fontsize=12, fontweight='bold')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Курс EUR/UAH')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "regression_models.png", dpi=300)
    plt.close()
    print("\n✓ Візуалізація регресійних моделей збережена")

    # Порівняння всіх моделей
    print("\n" + "=" * 70)
    print("ПІДСУМКОВЕ ПОРІВНЯННЯ МОДЕЛЕЙ")
    print("=" * 70)

    comparison = []

    for name, data in forecast_results.items():
        comparison.append({
            'Model': f'ES: {name}',
            'Type': 'Exponential Smoothing',
            'RMSE': data['metrics']['RMSE'],
            'MAE': data['metrics']['MAE'],
            'MAPE': data['metrics']['MAPE'],
            'R2': data['metrics']['R2']
        })

    for name, data in regression_results.items():
        comparison.append({
            'Model': f'Reg: {name}',
            'Type': 'Regression',
            'RMSE': data['metrics']['RMSE'],
            'MAE': data['metrics']['MAE'],
            'MAPE': data['metrics']['MAPE'],
            'R2': data['result']['r2']
        })

    comparison_df = pd.DataFrame(comparison).sort_values('RMSE')
    print("\n", comparison_df.to_string(index=False))

    # Збереження результатів
    comparison_df.to_csv(output_dir / "models_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("АНАЛІЗ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"\nРезультати збережено у: {output_dir}/")


if __name__ == "__main__":
    main()
