# Homework 10 — Где дешевле жить? Предсказание цен в Airbnb (NYC)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from IPython.display import display

sns.set(style='whitegrid')

# ===============================
# 1) Загрузка данных
# ===============================

path = r"C:\Users\22829\PycharmProjects\pythpn_tasks\homework_10\AB_NYC_2019.csv"
df = pd.read_csv(path)
print('shape:', df.shape)
display(df.head())

print(df.info())
display(df.describe().T)

# ===============================
# Простое EDA
# ===============================

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.histplot(df['price'], bins=100, kde=True)
plt.xlim(0,1000)

plt.subplot(1,2,2)
sns.boxplot(x='neighbourhood_group', y='price', data=df)
plt.ylim(0,1000)
plt.tight_layout()

print("Missing values:")
print(df.isnull().sum())

# ===============================
# Корреляции
# ===============================

num_cols_all = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols_all].corr().abs(), annot=True, fmt='.2f')

# ===============================
# Предобработка
# ===============================

df.drop(columns=['id','name','host_id','host_name','last_review'], errors='ignore', inplace=True)
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

df['log_price'] = np.log1p(df['price'])

# ===============================
# Фичи
# ===============================

def haversine_array(lat1, lon1, lat2, lon2):
    R = 6371
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

man_lat, man_lon = 40.7831, -73.9712
df['dist_to_manhattan_km'] = haversine_array(df['latitude'], df['longitude'], man_lat, man_lon)

df['high_demand'] = ((df['availability_365'] < 60) &
                     (df['number_of_reviews'] > df['number_of_reviews'].median())).astype(int)

# ===============================
# Формируем X,y
# ===============================

numeric_features = [
    'latitude','longitude','dist_to_manhattan_km',
    'minimum_nights','number_of_reviews','reviews_per_month',
    'calculated_host_listings_count','availability_365'
]

numeric_features = [c for c in numeric_features if c in df.columns]

categorical_features = ['neighbourhood_group','neighbourhood','room_type']
categorical_features = [c for c in categorical_features if c in df.columns]

X = df[numeric_features + categorical_features].copy()
y = df['log_price']

# ===============================
# Train / Test split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# ===============================
# Снижение кардинальности neighbourhood
# ===============================

def reduce_cardinality(series, top_n=20, other="OTHER"):
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), other)

# ВСЕГДА добавляем колонку neighbourhood_reduced
if 'neighbourhood' in X_train.columns:

    top20 = X_train['neighbourhood'].value_counts().nlargest(20).index

    X_train['neighbourhood_reduced'] = X_train['neighbourhood'].where(
        X_train['neighbourhood'].isin(top20), 'OTHER'
    )
    X_test['neighbourhood_reduced'] = X_test['neighbourhood'].where(
        X_test['neighbourhood'].isin(top20), 'OTHER'
    )

    X_train.drop(columns=['neighbourhood'], inplace=True)
    X_test.drop(columns=['neighbourhood'], inplace=True)

# финальные списки колонок
cat_cols = ['neighbourhood_reduced','neighbourhood_group','room_type']
cat_cols = [c for c in cat_cols if c in X_train.columns]

num_cols = numeric_features

print("Final numeric cols:", num_cols)
print("Final categorical cols:", cat_cols)

# ===============================
# Preprocessor
# ===============================

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# ===============================
# Функция оценки модели
# ===============================

def evaluate(model, Xtr, Xte, ytr, yte, preproc):
    pipe = Pipeline([('prep', preproc), ('model', model)])
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)

    return pipe, {
        "r2": r2_score(yte, pred),
        "mae": mean_absolute_error(yte, pred),
        "rmse": mean_squared_error(yte, pred) ** 0.5,
        "y_pred": pred
    }

# ===============================
# Модели
# ===============================

print("\nLinear Regression:")
pipe_lr, m_lr = evaluate(LinearRegression(), X_train, X_test, y_train, y_test, preprocessor)
print(m_lr)

print("\nRidgeCV:")
alphas = np.logspace(-3,3,20)
pipe_ridge, m_ridge = evaluate(RidgeCV(alphas=alphas), X_train, X_test, y_train, y_test, preprocessor)
print(m_ridge)

print("\nLassoCV:")
pipe_lasso, m_lasso = evaluate(LassoCV(cv=5, n_alphas=40, max_iter=5000), X_train, X_test, y_train, y_test, preprocessor)
print(m_lasso)

print("\nElasticNetCV:")
pipe_en, m_en = evaluate(
    ElasticNetCV(cv=5, n_alphas=40, l1_ratio=[.1,.3,.5,.7,.9,.95,1], max_iter=5000),
    X_train, X_test, y_train, y_test, preprocessor
)
print(m_en)

# ===============================
# Сравнение результатов
# ===============================

results = pd.DataFrame({
    "Linear": m_lr,
    "Ridge": m_ridge,
    "Lasso": m_lasso,
    "ElasticNet": m_en
}).T[['r2','mae','rmse']]

print("\nModel comparison:")
display(results)

# ===============================
# Извлекаем важность признаков (Ridge)
# ===============================

ohe = pipe_ridge.named_steps['prep'].named_transformers_['cat'].named_steps['ohe']
cat_feature_names = list(ohe.get_feature_names_out(cat_cols))

all_features = num_cols + cat_feature_names
coef = pipe_ridge.named_steps['model'].coef_

importance = pd.Series(coef, index=all_features).sort_values(key=abs, ascending=False)

print("\nTop 30 features:")
display(importance.head(30))

# ===============================
# Конвертация предсказаний в цену
# ===============================

y_test_price = np.expm1(y_test)
y_pred_price = np.expm1(m_ridge["y_pred"])

plt.figure(figsize=(6,6))
plt.scatter(y_test_price, y_pred_price, alpha=0.3)
plt.plot([0,1000],[0,1000],'--')
plt.xlim(0,1000); plt.ylim(0,1000)
plt.xlabel("Actual")
plt.ylabel("Predicted (Ridge)")

# ===============================
# Выводы
# ===============================

print("""
- Логарифмирование улучшает стабилизацию распределения.
- Ridge показывает лучшую устойчивость.
- Самые важные признаки: расстояние до Манхэттена, тип комнаты, borough, доступность и отзывы.
- Категоризация neighbourhood сильно помогает модели.
""")
