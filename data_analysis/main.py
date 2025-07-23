import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

show_graphs = True



def load_data(file_path):
    df = pd.read_excel(file_path, header=7)
    return df

filename = input("Введите название файла с датасетом: ")

print("=" * 50)
print(" Загрузка данных:")

try:
    df = load_data(filename)
except:
    print()
    print(f" ERROR: не удалось открыть файл '{filename}'")
    sys.exit(1)
else:
    print()
    print(f" SUCCESS: загружено {len(df)} записей")



def preprocess_data(df, target='Буст с позиции'):

    df_clean = df.copy()

    # Удаление
    cols_to_drop = [
        'Дата создания', 'Название', 'Поставщик', 'Бренд', 'Предмет',
        'Тип рекламы', 'Буст на позицию', 'Позиция в выдаче',
        'Стоимость за 1000 показов'
    ]
    cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    initial_len = len(df_clean)
    df_clean = df_clean.dropna(subset=[target])
    print(f"   Удалено {initial_len - len(df_clean)} строк с пропусками")

    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"   Удалено {initial_len - len(df_clean)} дубликатов")

    # Обработка выбросов в целевой переменной
    Q1 = df_clean[target].quantile(0.25)
    Q3 = df_clean[target].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    initial_len = len(df_clean)
    df_clean = df_clean[(df_clean[target] >= lower_bound) & (df_clean[target] <= upper_bound)]
    print(f"   Удалено {initial_len - len(df_clean)} выбросов в целевой переменной")

    # Удаление сильно коррелирующих признаков
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    corr_matrix = df_clean[numeric_cols].corr()

    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    # Удаляем один из коррелирующих признаков
    cols_to_remove = set()
    for col1, col2 in high_corr_pairs:
        if col1 != target and col2 != target:
            corr1 = abs(df_clean[col1].corr(df_clean[target]))
            corr2 = abs(df_clean[col2].corr(df_clean[target]))
            if corr1 < corr2:
                cols_to_remove.add(col1)
            else:
                cols_to_remove.add(col2)

    df_clean = df_clean.drop(columns=list(cols_to_remove))
    print(f"   Удалено {len(cols_to_remove)} сильно коррелирующих признаков")

    print(f"   Осталось {len(df_clean)} записей и {len(df_clean.columns)} признаков")
    return df_clean

print("=" * 50)
print(" Предобработка данных:")

try:
    df_processed = preprocess_data(df)
except:
    print()
    print(f" ERROR: ошибка в предобработке данных")
    sys.exit(1)
else:
    print()
    print(f" SUCCESS: данные предобработаны")

# Показываем доступные столбцы
print()
print(" Доступные столбцы после предобработки:")
for i, col in enumerate(df_processed.columns, 1):
    print(f"   {i}. {col}")

# Анализ целевой переменной
print("=" * 50)
print(" Анализ целевой переменной 'Буст с позиции':")
print(f"   Среднее значение: {df_processed['Буст с позиции'].mean():.2f}")
print(f"   Медиана: {df_processed['Буст с позиции'].median():.2f}")
print(f"   Стандартное отклонение: {df_processed['Буст с позиции'].std():.2f}")
print(f"   Минимум: {df_processed['Буст с позиции'].min():.2f}")
print(f"   Максимум: {df_processed['Буст с позиции'].max():.2f}")

if show_graphs:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df_processed['Буст с позиции'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Распределение "Буст с позиции"')
    plt.xlabel('Буст с позиции')
    plt.ylabel('Частота')

    plt.subplot(1, 2, 2)
    plt.boxplot(df_processed['Буст с позиции'])
    plt.title('Boxplot "Буст с позиции"')
    plt.ylabel('Буст с позиции')
    plt.tight_layout()
    plt.show()



print("=" * 50)
print(" Построение и обучение модели:")

X = df_processed.drop(columns=['Буст с позиции'])
y = df_processed['Буст с позиции']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

print(f"   Размер обучающей выборки: {len(X_train)}")
print(f"   Размер тестовой выборки: {len(X_test)}")

print()
print("   Используются модели XGBoost и Random Forest")

models = { 'XGBoost': XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
}

print()
try:
    for name, model in models.items():
        print(f"   Обучение модели: {name}")
        model.fit(X_train, y_train)
except:
    print()
    print(f" ERROR: ошибка в обучении модели")
    sys.exit(1)
else:
    print()
    print(f" SUCCESS: модели обучены")



print("=" * 50)
print(" Результаты обучения моделей:")

results = {}
for name, model in models.items():
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    try:
        cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except:
        cv_mean = r2_test
        cv_std = 0

    results[name] = {
        'model': model,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }

    print()
    print(f" Результаты обучения модели {name}")
    print(f"   MAE (train): {mae_train:.2f}")
    print(f"   MAE (test): {mae_test:.2f}")
    print(f"   R² (train): {r2_train:.4f}")
    print(f"   R² (test): {r2_test:.4f}")
    print(f"   RMSE (test): {rmse_test:.2f}")
    print(f"   CV R²: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")

best_model_name = max(results.keys(), key=lambda x: results[x]['r2_test'])
best_model = results[best_model_name]['model']

print()
print(f" Лучшая модель: {best_model_name}")



print("=" * 50)
print(" Анализ важности признаков:")

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    if show_graphs:
        plt.figure(figsize=(10, 6))
        importances.plot(kind='barh')
        plt.title(f'Важность признаков ({best_model_name})')
        plt.xlabel('Важность')
        plt.tight_layout()
        plt.show()

    print()
    print(" Топ-5 важных признаков:")
    for i, (feature, importance) in enumerate(importances.head().items(), 1):
        print(f"   {i}. {feature}: {importance:.4f}")



print("=" * 50)
print(" Оптимизация параметров для улучшения позиции:")

while True:
    print()
    article = int(input("Введите артикул товара: "))

    if article not in df_processed['Артикул'].values:
        print()
        print(" Артикул не найден, доступные артикулы:")

        available_columns = ['Артикул', 'Буст с позиции']
        if 'Средняя цена без СПП' in df_processed.columns:
            available_columns.append('Средняя цена без СПП')
        if 'Рейтинг' in df_processed.columns:
            available_columns.append('Рейтинг')
        
        top_products = df_processed.sort_values('Буст с позиции')[available_columns]
        for i, (idx, row) in enumerate(top_products.iterrows(), 1):
            price_info = f", Цена: {row['Средняя цена без СПП']:.0f}" if 'Средняя цена без СПП' in available_columns else ""
            rating_info = f", Рейтинг: {row['Рейтинг']:.1f}" if 'Рейтинг' in available_columns else ""
            print(f"   {i}. Артикул: {row['Артикул']:.0f}, Позиция: {row['Буст с позиции']:.0f}{price_info}{rating_info}")
    else:
        example_row = df_processed[df_processed['Артикул'] == article].iloc[0]
        break



def predict_position(model, scaler, feature_names, input_values):
    """Прогнозирует позицию товара на основе параметров"""
    input_df = pd.DataFrame([input_values], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]

def optimize_product_position(model, scaler, feature_names, current_values,
                            controllable_features, bounds, current_actual_rank, optimization_method='differential_evolution'):
    """
    Оптимизирует параметры товара для улучшения позиции
    Args:
        model: обученная модель
        scaler: нормализатор
        feature_names: названия признаков
        current_values: текущие значения параметров
        controllable_features: список признаков, которые можно изменять
        bounds: границы для каждого изменяемого признака
        current_actual_rank: фактическая текущая позиция товара
        optimization_method: метод оптимизации
    """

    print(f" Начинаем оптимизацию позиции товара")
    current_predicted_position = predict_position(model, scaler, feature_names, current_values)
    print(f" Фактическая позиция: {current_actual_rank:.0f}")
    print(f" Предсказанная позиция: {current_predicted_position:.0f}")
    def objective_function(params):
        # Создаем новый вектор параметров
        new_values = current_values.copy()
        for i, feature in enumerate(controllable_features):
            new_values[feature] = params[i]

        # Прогнозируем новую позицию
        predicted_position = predict_position(model, scaler, feature_names, new_values)
        return predicted_position

    # Подготовка границ для оптимизации
    optimization_bounds = [bounds[feature] for feature in controllable_features]

    # Запуск оптимизации
    if optimization_method == 'differential_evolution':
        result = differential_evolution(
            objective_function,
            optimization_bounds,
            maxiter=500,
            popsize=10,
            seed=42
        )
    else:
        result = minimize(
            objective_function,
            x0=[current_values[feature] for feature in controllable_features],
            bounds=optimization_bounds,
            method='L-BFGS-B'
        )

    # Получаем оптимальные параметры
    optimal_params = current_values.copy()
    for i, feature in enumerate(controllable_features):
        optimal_params[feature] = result.x[i]

    # Прогнозируем оптимальную позицию
    optimal_predicted_position = predict_position(model, scaler, feature_names, optimal_params)

    # Рассчитываем улучшение относительно фактической позиции
    improvement_from_actual = current_actual_rank - optimal_predicted_position
    improvement_from_predicted = current_predicted_position - optimal_predicted_position
    improvement_percent = (improvement_from_actual/current_actual_rank)*100 if current_actual_rank > 0 else 0

    print()
    print(f" Результаты оптимизации:")
    print(f"   Фактическая позиция: {current_actual_rank:.0f}")
    print(f"   Предсказанная позиция (текущая): {current_predicted_position:.0f}")
    print(f"   Предсказанная позиция (оптимальная): {optimal_predicted_position:.0f}")
    print(f"   Улучшение от фактической позиции: {improvement_from_actual:.0f} позиций")
    print(f"   Улучшение от предсказанной позиции: {improvement_from_predicted:.0f} позиций")
    print(f"   Процент улучшения от фактической: {improvement_percent:.1f}%")

    # Проверяем качество улучшения
    if improvement_from_actual > 0 and improvement_percent >= 2.0:
        print()
        print(f" Оптимизация успешна!")
        print(" Рекомендуемые изменения параметров:")
        for feature in controllable_features:
            old_val = current_values[feature]
            new_val = optimal_params[feature]
            change = new_val - old_val
            if old_val != 0:
                change_pct = (change / old_val) * 100
                print(f"   {feature}: {old_val:.2f} → {new_val:.2f} ({change:+.2f}, {change_pct:+.1f}%)")
            else:
                print(f"   {feature}: {old_val:.2f} → {new_val:.2f} ({change:+.2f})")

        print()
        print(f" Ожидаемый результат:")
        print(f"   Текущая фактическая позиция: {current_actual_rank:.0f}")
        print(f"   Ожидаемая позиция после оптимизации: {optimal_predicted_position:.0f}")
        print(f"   Ожидаемое улучшение: {improvement_from_actual:.0f} позиций")

    elif improvement_from_actual > 0 and improvement_percent < 2.0:
        print()
        print(f" Небольшое улучшение найдено ({improvement_percent:.1f}%)")
        print("Пробуем альтернативные стратегии оптимизации...")

        # Альтернативная стратегия 1: Более агрессивные изменения
        print()
        print(f" Альтернативная стратегия 1: Более агрессивные изменения")
        aggressive_bounds = {}
        for feature in controllable_features:
            if feature == 'Средняя цена без СПП':
                current_price = current_values[feature]
                aggressive_bounds[feature] = (current_price * 0.5, current_price * 0.9)  # Снижение на 10-50%
            elif feature == 'Общая скидка без СПП':
                current_discount = current_values[feature]
                aggressive_bounds[feature] = (current_discount, min(current_discount + 30, 95))  # Увеличение до 30%
            elif feature == 'Коэффициент демпинга':
                aggressive_bounds[feature] = (0, 15)  # От 0 до 15
            elif feature == 'Рейтинг':
                current_rating = current_values[feature]
                aggressive_bounds[feature] = (current_rating, min(current_rating + 1.0, 5.0))  # Улучшение до 1.0 балла
            elif feature == 'Остатки на конец периода':
                current_stock = current_values[feature]
                aggressive_bounds[feature] = (current_stock, current_stock * 5)  # Увеличение в 5 раз
            else:
                aggressive_bounds[feature] = bounds[feature]

        # Запускаем агрессивную оптимизацию
        aggressive_optimization_bounds = [aggressive_bounds[feature] for feature in controllable_features]
        aggressive_result = differential_evolution(
            objective_function,
            aggressive_optimization_bounds,
            maxiter=1000,
            popsize=20,
            seed=42
        )

        aggressive_optimal_params = current_values.copy()
        for i, feature in enumerate(controllable_features):
            aggressive_optimal_params[feature] = aggressive_result.x[i]

        aggressive_optimal_position = predict_position(model, scaler, feature_names, aggressive_optimal_params)
        aggressive_improvement = current_actual_rank - aggressive_optimal_position
        aggressive_improvement_percent = (aggressive_improvement/current_actual_rank)*100 if current_actual_rank > 0 else 0

        print(f"   Агрессивная оптимизация:")
        print(f"   - Ожидаемая позиция: {aggressive_optimal_position:.0f}")
        print(f"   - Улучшение: {aggressive_improvement:.0f} позиций ({aggressive_improvement_percent:.1f}%)")

        # Выбираем лучший результат
        if aggressive_improvement_percent > improvement_percent:
            print("Агрессивная стратегия дала лучший результат")
            optimal_params = aggressive_optimal_params
            optimal_predicted_position = aggressive_optimal_position
            improvement_from_actual = aggressive_improvement
            improvement_percent = aggressive_improvement_percent

            print("Рекомендуемые изменения параметров (агрессивная стратегия):")
            for feature in controllable_features:
                old_val = current_values[feature]
                new_val = optimal_params[feature]
                change = new_val - old_val
                if old_val != 0:
                    change_pct = (change / old_val) * 100
                    print(f"   {feature}: {old_val:.2f} → {new_val:.2f} ({change:+.2f}, {change_pct:+.1f}%)")
                else:
                    print(f"   {feature}: {old_val:.2f} → {new_val:.2f} ({change:+.2f})")
        else:
            print("Агрессивная стратегия не дала лучшего результата")
            print("Используем исходные рекомендации:")
            for feature in controllable_features:
                old_val = current_values[feature]
                new_val = optimal_params[feature]
                change = new_val - old_val
                if old_val != 0:
                    change_pct = (change / old_val) * 100
                    print(f"   {feature}: {old_val:.2f} → {new_val:.2f} ({change:+.2f}, {change_pct:+.1f}%)")
                else:
                    print(f"   {feature}: {old_val:.2f} → {new_val:.2f} ({change:+.2f})")

    else:
        print()
        print(f" Улучшение не найдено")
        print("Пробуем альтернативные стратегии...")

        # Альтернативная стратегия 2: Поиск в противоположном направлении
        print()
        print(f" Альтернативная стратегия 2: Поиск в противоположном направлении")

        # Инвертируем границы для некоторых параметров
        inverted_bounds = {}
        for feature in controllable_features:
            if feature == 'Средняя цена без СПП':
                current_price = current_values[feature]
                inverted_bounds[feature] = (current_price * 1.1, current_price * 1.5)  # Увеличение цены
            elif feature == 'Общая скидка без СПП':
                current_discount = current_values[feature]
                inverted_bounds[feature] = (max(current_discount - 20, 0), current_discount)  # Уменьшение скидки
            elif feature == 'Коэффициент демпинга':
                inverted_bounds[feature] = (10, 20)  # Высокий демпинг
            else:
                inverted_bounds[feature] = bounds[feature]

        # Запускаем инвертированную оптимизацию
        inverted_optimization_bounds = [inverted_bounds[feature] for feature in controllable_features]
        inverted_result = differential_evolution(
            objective_function,
            inverted_optimization_bounds,
            maxiter=500,
            popsize=15,
            seed=42
        )

        inverted_optimal_params = current_values.copy()
        for i, feature in enumerate(controllable_features):
            inverted_optimal_params[feature] = inverted_result.x[i]

        inverted_optimal_position = predict_position(model, scaler, feature_names, inverted_optimal_params)
        inverted_improvement = current_actual_rank - inverted_optimal_position

        print(f"   Инвертированная оптимизация:")
        print(f"   - Ожидаемая позиция: {inverted_optimal_position:.0f}")
        print(f"   - Улучшение: {inverted_improvement:.0f} позиций")

        if inverted_improvement > 0:
            print(f"   Инвертированная стратегия дала положительный результат")
            optimal_params = inverted_optimal_params
            optimal_predicted_position = inverted_optimal_position
            improvement_from_actual = inverted_improvement

            print("Рекомендуемые изменения параметров (инвертированная стратегия):")
            for feature in controllable_features:
                old_val = current_values[feature]
                new_val = optimal_params[feature]
                change = new_val - old_val
                if old_val != 0:
                    change_pct = (change / old_val) * 100
                    print(f"   {feature}: {old_val:.2f} → {new_val:.2f} ({change:+.2f}, {change_pct:+.1f}%)")
                else:
                    print(f"   {feature}: {old_val:.2f} → {new_val:.2f} ({change:+.2f})")
        else:
            print(f"    Инвертированная стратегия не дала положительного результата")
            print("Возможные причины:")
            print("   - Модель не обучена достаточно хорошо")
            print("   - Товар уже находится в оптимальной позиции")
            print("   - Нужны другие параметры для оптимизации")
            print("   - Требуется переобучение модели на новых данных")

    return optimal_params, optimal_predicted_position, improvement_from_actual, current_actual_rank



current_values = example_row.drop('Буст с позиции').to_dict()
current_actual_rank = example_row['Буст с позиции']

print()
print(f" Выбранный товар:")
print(f"   Артикул: {example_row['Артикул']:.0f}")
print(f"   Фактическая позиция: {current_actual_rank:.0f}")
print(f"   Предсказанная позиция: {predict_position(best_model, scaler, X.columns, current_values):.0f}")

print()
print(" Текущие параметры товара:")
for k, v in current_values.items():
    print(f"   {k}: {v:.2f}")

# Определение управляемых параметров и их границ
controllable_features = []
bounds = {}

# Анализируем каждый признак и определяем, можно ли его изменять
for feature in X.columns:
    if feature == 'Артикул':
        continue  # Артикул нельзя изменять

    if feature in ['Заказы', 'Упущенные заказы', 'Количество отзывов на конец периода']:
        # Эти параметры зависят от времени и спроса, их сложно контролировать напрямую
        continue

    if feature == 'Средняя цена без СПП':
        # Цену можно снижать для улучшения позиции
        current_price = current_values[feature]
        bounds[feature] = (current_price * 0.7, current_price * 0.95)  # Снижение на 5-30%
        controllable_features.append(feature)

    elif feature == 'Общая скидка без СПП':
        # Скидку можно увеличивать
        current_discount = current_values[feature]
        bounds[feature] = (current_discount, min(current_discount + 20, 90))  # Увеличение до 20%
        controllable_features.append(feature)

    elif feature == 'Коэффициент демпинга':
        # Демпинг можно контролировать
        bounds[feature] = (0, 10)  # От 0 до 10
        controllable_features.append(feature)

    elif feature == 'Рейтинг':
        # Рейтинг можно улучшать через качество товара
        current_rating = current_values[feature]
        bounds[feature] = (current_rating, min(current_rating + 0.5, 5.0))  # Улучшение до 0.5 балла
        controllable_features.append(feature)

    elif feature == 'Остатки на конец периода':
        # Остатки можно увеличивать
        current_stock = current_values[feature]
        bounds[feature] = (current_stock, current_stock * 3)  # Увеличение в 3 раза
        controllable_features.append(feature)

print()
print(f" Управляемые параметры ({len(controllable_features)}):")
for feature in controllable_features:
    print(f"   {feature}: {bounds[feature]}")



print("=" * 50)
if controllable_features:
    optimal_params, optimal_predicted_position, improvement_from_actual, current_actual_rank = optimize_product_position(
        model=best_model,
        scaler=scaler,
        feature_names=X.columns,
        current_values=current_values,
        controllable_features=controllable_features,
        bounds=bounds,
        current_actual_rank=current_actual_rank,
        optimization_method='differential_evolution'
    )

    # Визуализация результатов
    if improvement_from_actual > 0:
        if show_graphs:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # График изменения параметров
            features_to_plot = [f for f in controllable_features if f in optimal_params]
            current_vals = [current_values[f] for f in features_to_plot]
            optimal_vals = [optimal_params[f] for f in features_to_plot]

            x = np.arange(len(features_to_plot))
            width = 0.35

            axes[0].bar(x - width/2, current_vals, width, label='Текущие', alpha=0.8)
            axes[0].bar(x + width/2, optimal_vals, width, label='Оптимальные', alpha=0.8)
            axes[0].set_xlabel('Параметры')
            axes[0].set_ylabel('Значения')
            axes[0].set_title('Сравнение текущих и оптимальных параметров')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(features_to_plot, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # График позиций
            positions = [current_actual_rank, optimal_predicted_position]
            labels = ['Фактическая', 'Ожидаемая']
            colors = ['red', 'green']

            axes[1].bar(labels, positions, color=colors, alpha=0.7)
            axes[1].set_ylabel('Позиция (меньше = лучше)')
            axes[1].set_title('Сравнение позиций')
            axes[1].grid(True, alpha=0.3)

            # Добавляем значения на столбцы
            for i, pos in enumerate(positions):
                axes[1].text(i, pos + max(positions) * 0.01, f'{pos:.0f}',
                            ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.show()

        print()
        print(f" Оптимизация завершена")
        print(f" Товар может улучшить свою позицию на {improvement_from_actual:.0f} пунктов")
        if current_actual_rank > 0:
            print(f" Это улучшение на {(improvement_from_actual/current_actual_rank)*100:.1f}%")

else:
    print()
    print("Нет управляемых параметров для оптимизации")
    print("Попробуйте изменить логику определения управляемых параметров")
