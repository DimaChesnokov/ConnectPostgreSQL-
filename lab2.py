import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def load_mpg_dataset():
    """Загрузка набора данных 'mpg' из библиотеки Seaborn."""
    try:
        df = sns.load_dataset('mpg')
        print("Набор данных 'mpg' успешно загружен")
        return df
    except Exception as e:
        print("Ошибка загрузки набора данных 'mpg':", e)
        exit()

def analyze_dataset(df):
    """Анализ набора данных: подсчёт строк и столбцов."""
    rows, columns = df.shape
    print(f"Количество строк: {rows}, Количество столбцов: {columns}")

def normalize(data):
    """Нормализация данных (Min-Max Scaling)"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def exploratory_analysis_numeric(df):
    """Разведочный анализ числовых переменных."""
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        print(f"\nАнализ переменной: {col}")
        print(f"Доля пропусков: {df[col].isnull().mean():.5%}")
        print(f"Максимальное значение: {df[col].max()}")
        print(f"Минимальное значение: {df[col].min()}")
        print(f"Среднее значение: {df[col].mean()}")
        print(f"Медиана: {df[col].median()}")
        print(f"Дисперсия: {df[col].var()}")
        print(f"Квантиль 0.1: {df[col].quantile(0.1)}")
        print(f"Квантиль 0.9: {df[col].quantile(0.9)}")
        print(f"Квартиль 1: {df[col].quantile(0.25)}")
        print(f"Квартиль 3: {df[col].quantile(0.75)}")

def exploratory_analysis_categorical(df):
    """Разведочный анализ категориальных переменных."""
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        print(f"\nАнализ переменной: {col}")
        print(f"Доля пропусков: {df[col].isnull().mean():.2%}")
        print(f"Количество уникальных значений: {df[col].nunique()}")
        print(f"Мода: {df[col].mode().iloc[0] if not df[col].mode().empty else 'Нет моды'}")

def encode_categorical_variables(df):
    """Кодирование категориальных переменных."""
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    for col in ['origin']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Столбец '{col}' закодирован методом LabelEncoding.")
    one_hot_cols = ['name']
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
    print("Закодированы OneHotEncoding: 'name'")
    return df

def gradient_descent(x, y, lr=0.001, epochs=1000):
    """Исправленный градиентный спуск."""
    m, b = 0.0, 0.0
    n = len(x)
    x = np.array(x)
    y = np.array(y)
    for epoch in range(epochs):
        y_pred = m * x + b
        error = y - y_pred
        dm = -2 / n * np.sum(x * error)
        db = -2 / n * np.sum(error)
        m -= lr * dm
        b -= lr * db
        if epoch % 100 == 0:
            print(f"Эпоха {epoch}: m = {m:.4f}, b = {b:.4f}")
    return m, b

def stochastic_gradient_descent(x, y, lr=0.001, epochs=100):
    """Исправленный стохастический градиентный спуск."""
    m, b = 0.0, 0.0
    n = len(x)
    x = np.array(x)
    y = np.array(y)
    for epoch in range(epochs):
        for i in range(n):
            xi, yi = x[i], y[i]
            y_pred = m * xi + b
            error = yi - y_pred
            m += lr * error * xi
            b += lr * error
        if epoch % 10 == 0:
            print(f"Эпоха {epoch}: m = {m:.4f}, b = {b:.4f}")
    return m, b

def hypothesis_testing(df):
    """Проверка статистических гипотез."""
    print("\nГипотеза 1: Различается ли расход топлива (mpg) автомобилей с разным числом цилиндров?")
    groups = [df[df['cylinders'] == cyl]['mpg'].dropna() for cyl in df['cylinders'].unique() if not pd.isnull(cyl)]
    stat, p_value = f_oneway(*groups)
    print(f"ANOVA: статистика = {stat:.2f}, p-значение = {p_value:.2e}")

    print("\nГипотеза 2: Различается ли вес автомобилей (weight) в зависимости от страны происхождения (origin)?")
    usa_weight = df[df['origin'] == 0]['weight'].dropna()
    japan_weight = df[df['origin'] == 2]['weight'].dropna()
    stat, p_value = ttest_ind(usa_weight, japan_weight, equal_var=False)
    print(f"t-test: статистика = {stat:.2f}, p-значение = {p_value:.2e}")

def correlation_analysis(df, target_column):
    """Построение таблицы корреляции признаков и целевого столбца."""
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_table = df[numeric_columns].corr()[target_column].sort_values(ascending=False)
    print("\nТаблица корреляции с целевым столбцом:")
    print(correlation_table)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Корреляционная матрица')
    plt.show()

if __name__ == "__main__":
    df = load_mpg_dataset()
    analyze_dataset(df)
    exploratory_analysis_numeric(df)
    exploratory_analysis_categorical(df)
    df = encode_categorical_variables(df)
    df = df.dropna(subset=['mpg', 'weight'])
    x = normalize(df['weight'].values)
    y = normalize(df['mpg'].values)
    hypothesis_testing(df)
    print("\nОбычный градиентный спуск:")
    gradient_descent(x, y)
    print("\nСтохастический градиентный спуск:")
    stochastic_gradient_descent(x, y)
    correlation_analysis(df, target_column='mpg')
