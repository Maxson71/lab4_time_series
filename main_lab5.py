"""
Лабораторна робота №5: Алгоритми ковзного вікна
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from modules.data_loader import TimeSeriesLoader
from modules.sliding_window import SlidingWindowAlgorithms, ModelEvaluator


def main():
    # Конфігурація
    valcode = "EUR"
    output_dir = Path(valcode) / "lab5"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ЛАБОРАТОРНА РОБОТА №5: АЛГОРИТМИ КОВЗНОГО ВІКНА")
    print("=" * 70)

    # 1. Завантаження даних
    print("\n" + "=" * 70)
    print("ЕТАП 1: Завантаження та підготовка даних")
    print("=" * 70)

    loader = TimeSeriesLoader(valcode=valcode)
    df = loader.load_data()

    # Використовуємо тільки колонку rate
    ts_data = df.set_index('date')['rate']

    print(f"✓ Завантажено {len(ts_data)} спостережень")
    print(f"  Період: {ts_data.index[0]} - {ts_data.index[-1]}")
    print(f"  Середнє: {ts_data.mean():.4f} грн")
    print(f"  Стд. відхилення: {ts_data.std():.4f} грн")

    # 2. Розділення на train/test
    # Використаємо 80% для навчання, 20% для тестування
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]

    print(f"\n✓ Розділення даних:")
    print(f"  Тренувальна вибірка: {len(train_data)} спостережень ({len(train_data) / len(ts_data) * 100:.1f}%)")
    print(f"  Тестова вибірка: {len(test_data)} спостережень ({len(test_data) / len(ts_data) * 100:.1f}%)")

    # 3. Ковзні середні
    print("\n" + "=" * 70)
    print("ЕТАП 2: Алгоритми ковзного вікна")
    print("=" * 70)

    swa = SlidingWindowAlgorithms(train_data)

    # SMA з різними вікнами
    windows = [7, 14, 30, 60]
    ma_results = {}

    print("\n[1] Проста ковзна середня (SMA):")
    for window in windows:
        sma = swa.simple_moving_average(window)

        # Оцінка якості на тренувальних даних
        valid_idx = train_data.index[window:]
        metrics = ModelEvaluator.calculate_metrics(
            train_data[valid_idx].values,
            sma[valid_idx].values
        )

        ma_results[f'SMA_{window}'] = metrics
        print(f"  SMA-{window}: RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%")

    # WMA
    print("\n[2] Зважена ковзна середня (WMA):")
    wma_30 = swa.weighted_moving_average(30)
    valid_idx = train_data.index[30:]
    metrics_wma = ModelEvaluator.calculate_metrics(
        train_data[valid_idx].values,
        wma_30[valid_idx].values
    )
    print(f"  WMA-30: RMSE={metrics_wma['RMSE']:.4f}, MAPE={metrics_wma['MAPE']:.2f}%")

    # EMA
    print("\n[3] Експоненціальна ковзна середня (EMA):")
    ema_spans = [7, 14, 30]
    for span in ema_spans:
        ema = swa.exponential_moving_average(span)
        metrics_ema = ModelEvaluator.calculate_metrics(
            train_data.values,
            ema.values
        )
        print(f"  EMA-{span}: RMSE={metrics_ema['RMSE']:.4f}, MAPE={metrics_ema['MAPE']:.2f}%")

    # Подвійне експоненціальне згладжування
    print("\n[4] Подвійне експоненціальне згладжування (Holt):")
    des = swa.double_exponential_smoothing(alpha=0.3, beta=0.1)
    metrics_des = ModelEvaluator.calculate_metrics(
        train_data.values,
        des.values
    )
    print(f"  DES: RMSE={metrics_des['RMSE']:.4f}, MAPE={metrics_des['MAPE']:.2f}%")

    # Візуалізація ковзних середніх
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # SMA
    axes[0, 0].plot(train_data.index, train_data.values,
                    label='Оригінальні дані', alpha=0.5, linewidth=1)
    for window in [7, 14, 30]:
        sma = swa.simple_moving_average(window)
        axes[0, 0].plot(sma.index, sma.values, label=f'SMA-{window}', linewidth=1.5)
    axes[0, 0].set_title('Проста ковзна середня (SMA)', fontweight='bold')
    axes[0, 0].set_xlabel('Дата')
    axes[0, 0].set_ylabel('Курс (грн)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # WMA
    axes[0, 1].plot(train_data.index, train_data.values,
                    label='Оригінальні дані', alpha=0.5, linewidth=1)
    axes[0, 1].plot(wma_30.index, wma_30.values,
                    label='WMA-30', color='red', linewidth=1.5)
    sma_30 = swa.simple_moving_average(30)
    axes[0, 1].plot(sma_30.index, sma_30.values,
                    label='SMA-30', color='blue', linewidth=1.5, alpha=0.7)
    axes[0, 1].set_title('Зважена vs Проста КС', fontweight='bold')
    axes[0, 1].set_xlabel('Дата')
    axes[0, 1].set_ylabel('Курс (грн)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # EMA
    axes[1, 0].plot(train_data.index, train_data.values,
                    label='Оригінальні дані', alpha=0.5, linewidth=1)
    for span in [7, 14, 30]:
        ema = swa.exponential_moving_average(span)
        axes[1, 0].plot(ema.index, ema.values, label=f'EMA-{span}', linewidth=1.5)
    axes[1, 0].set_title('Експоненціальна ковзна середня (EMA)', fontweight='bold')
    axes[1, 0].set_xlabel('Дата')
    axes[1, 0].set_ylabel('Курс (грн)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # DES
    axes[1, 1].plot(train_data.index, train_data.values,
                    label='Оригінальні дані', alpha=0.5, linewidth=1)
    axes[1, 1].plot(des.index, des.values,
                    label='Holt DES', color='purple', linewidth=1.5)
    axes[1, 1].set_title('Подвійне експоненціальне згладжування', fontweight='bold')
    axes[1, 1].set_xlabel('Дата')
    axes[1, 1].set_ylabel('Курс (грн)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "moving_averages.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Візуалізація ковзних середніх збережена")

    # 4. ARIMA моделі
    print("\n" + "=" * 70)
    print("ЕТАП 3: ARIMA моделювання")
    print("=" * 70)

    swa_arima = SlidingWindowAlgorithms(train_data)

    # Автоматичний вибір ARIMA
    print("\n[1] Автоматичний вибір ARIMA моделі...")
    auto_result = swa_arima.auto_arima_selection(max_p=3, max_q=3, max_d=2)

    print(f"  Найкраща модель: ARIMA{auto_result['best_order']}")
    print(f"  AIC: {auto_result['best_aic']:.2f}")
    print("\n  Топ-5 моделей за AIC:")
    print(auto_result['all_results'].head())

    # Підгонка найкращої моделі
    best_order = auto_result['best_order']
    arima_result = swa_arima.fit_arima(order=best_order)

    print(f"\n✓ ARIMA{best_order} підігнана успішно")
    print(f"  Параметри моделі:")
    print(arima_result['params'])

    # Оцінка якості на тренувальних даних
    metrics_arima = ModelEvaluator.calculate_metrics(
        train_data.values,
        arima_result['fitted_values'].values
    )
    print(f"\n  Метрики на тренувальних даних:")
    print(f"    RMSE: {metrics_arima['RMSE']:.4f}")
    print(f"    MAE:  {metrics_arima['MAE']:.4f}")
    print(f"    MAPE: {metrics_arima['MAPE']:.2f}%")
    print(f"    R²:   {metrics_arima['R2']:.4f}")

    # Аналіз залишків
    ModelEvaluator.plot_residuals(
        arima_result['residuals'],
        save_path=output_dir / "arima_residuals.png"
    )
    print("\n✓ Аналіз залишків ARIMA збережено")

    # 5. Прогнозування на різні горизонти
    print("\n" + "=" * 70)
    print("ЕТАП 4: Екстраполяція даних")
    print("=" * 70)

    # Різні горизонти прогнозування
    test_length = len(test_data)
    horizons = {
        '0.5x': int(test_length * 0.5),
        '1.0x': test_length,
        '1.5x': int(test_length * 1.5),
        '2.0x': test_length * 2
    }

    forecast_results = {}

    for name, steps in horizons.items():
        print(f"\n[{name}] Прогноз на {steps} кроків:")

        # Прогноз з довірчими інтервалами
        forecast_data = swa_arima.get_forecast_with_intervals(steps=steps, alpha=0.05)

        # Створення індексу для прогнозу
        last_date = train_data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        forecast_series = pd.Series(
            forecast_data['forecast'].values,
            index=forecast_index
        )

        # Оцінка якості (тільки для доступних тестових даних)
        if steps <= len(test_data):
            actual = test_data[:steps]
            metrics = ModelEvaluator.calculate_metrics(
                actual.values,
                forecast_series[:len(actual)].values
            )

            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE:  {metrics['MAE']:.4f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  R²:   {metrics['R2']:.4f}")

            forecast_results[name] = {
                'steps': steps,
                'forecast': forecast_series,
                'metrics': metrics,
                'lower_ci': pd.Series(forecast_data['lower_bound'].values, index=forecast_index),
                'upper_ci': pd.Series(forecast_data['upper_bound'].values, index=forecast_index)
            }
        else:
            print(f"  Прогноз виходить за межі тестових даних")
            forecast_results[name] = {
                'steps': steps,
                'forecast': forecast_series,
                'metrics': None,
                'lower_ci': pd.Series(forecast_data['lower_bound'].values, index=forecast_index),
                'upper_ci': pd.Series(forecast_data['upper_bound'].values, index=forecast_index)
            }

    # Візуалізація прогнозів
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(forecast_results.items()):
        ax = axes[idx]

        # Тренувальні дані
        ax.plot(train_data.index, train_data.values,
                label='Тренувальні дані', color='blue', linewidth=1.5)

        # Тестові дані (якщо є)
        if result['steps'] <= len(test_data):
            ax.plot(test_data[:result['steps']].index,
                    test_data[:result['steps']].values,
                    label='Фактичні дані', color='green', linewidth=1.5)

        # Прогноз
        ax.plot(result['forecast'].index, result['forecast'].values,
                label='Прогноз', color='red', linewidth=2, linestyle='--')

        # Довірчий інтервал
        ax.fill_between(result['forecast'].index,
                        result['lower_ci'].values,
                        result['upper_ci'].values,
                        alpha=0.2, color='red', label='95% Довірчий інтервал')

        # Вертикальна лінія початку прогнозу
        ax.axvline(x=train_data.index[-1], color='gray',
                   linestyle=':', alpha=0.7)

        # Метрики (якщо є)
        if result['metrics']:
            textstr = f"RMSE: {result['metrics']['RMSE']:.3f}\nMAPE: {result['metrics']['MAPE']:.2f}%"
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(f'Прогноз на {name} ({result["steps"]} днів)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Дата', fontsize=10)
        ax.set_ylabel('Курс EUR/UAH (грн)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "forecasts_all_horizons.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Візуалізація прогнозів збережена")

    # 6. Порівняння моделей
    print("\n" + "=" * 70)
    print("ЕТАП 5: Порівняння моделей")
    print("=" * 70)

    # Зберігаємо результати
    comparison_data = {
        'ARIMA': {
            'order': best_order,
            'aic': float(arima_result['model'].aic),
            'bic': float(arima_result['model'].bic),
            'train_metrics': {k: float(v) for k, v in metrics_arima.items()}
        },
        'forecasts': {}
    }

    for name, result in forecast_results.items():
        if result['metrics']:
            comparison_data['forecasts'][name] = {
                'steps': result['steps'],
                'metrics': {k: float(v) for k, v in result['metrics'].items()}
            }

    with open(output_dir / "model_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print("\n✓ Порівняння моделей збережено")

    # Підсумкова таблиця
    print("\n" + "=" * 70)
    print("ПІДСУМКОВА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
    print("=" * 70)

    print(f"\nНайкраща модель: ARIMA{best_order}")
    print(f"AIC: {arima_result['model'].aic:.2f}, BIC: {arima_result['model'].bic:.2f}")

    print("\nМетрики прогнозування:")
    print("-" * 70)
    print(f"{'Горизонт':<15} {'Кроків':<10} {'RMSE':<12} {'MAE':<12} {'MAPE (%)':<12}")
    print("-" * 70)
    for name, result in forecast_results.items():
        if result['metrics']:
            m = result['metrics']
            print(f"{name:<15} {result['steps']:<10} {m['RMSE']:<12.4f} "
                  f"{m['MAE']:<12.4f} {m['MAPE']:<12.2f}")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("АНАЛІЗ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"\nРезультати збережено у папці: {output_dir}/")


if __name__ == "__main__":
    main()
