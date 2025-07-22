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