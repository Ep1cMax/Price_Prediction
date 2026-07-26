import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from GetDataset import datasetpath

INR_TO_RUB = 0.8083

# 1. Загрузка и очистка
def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    print("Исходные размеры:", df.shape)

    cols_to_drop = ['Unnamed: 0', 'image', 'link']
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_drop, inplace=True, errors='ignore')

    target = 'actual_price'
    if target not in df.columns:
        raise KeyError(f"Колонка '{target}' не найдена.")

    def clean_currency(price_str):
        if isinstance(price_str, str):
            cleaned = re.sub(r'[^\d.\-]', '', price_str)
            cleaned = cleaned.replace(',', '')
            try:
                return float(cleaned)
            except ValueError:
                return np.nan
        return float(price_str) if pd.notnull(price_str) else np.nan

    # Очистка цен
    df[target] = df[target].apply(clean_currency)
    if 'discount_price' in df.columns:
        df['discount_price'] = df['discount_price'].apply(clean_currency)
        df['discount_price'] = df['discount_price'].fillna(df[target])
        mask_invalid = (df['discount_price'].isna()) | (df['discount_price'] <= 0)
        df.loc[mask_invalid, 'discount_price'] = df.loc[mask_invalid, target]
        df['discount_pct'] = (df[target] - df['discount_price']) / df[target]
        df['discount_pct'] = df['discount_pct'].fillna(0).clip(0, 1)
        df['discount_price'] = df['discount_price'] * INR_TO_RUB
    else:
        df['discount_price'] = df[target]
        df['discount_pct'] = 0.0

    df[target] = df[target] * INR_TO_RUB

    # Удаляем только некорректные цены
    df = df.dropna(subset=[target])
    df = df[df[target] > 0]

    # Обрезка выбросов
    upper_limit = df[target].quantile(0.999)
    if upper_limit > df[target].median() * 100:
        upper_limit = df[target].quantile(0.995)
    df = df[df[target] <= upper_limit]
    print(f"Цены выше {upper_limit:,.2f} руб. удалены. Размер: {df.shape}")

    df['log_price'] = np.log1p(df[target])

    # Очистка числовых колонок
    if 'no_of_ratings' in df.columns:
        df['no_of_ratings'] = df['no_of_ratings'].astype(str).str.replace(',', '', regex=False)
        df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors='coerce')
    if 'ratings' in df.columns:
        df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')

    # Инженерия из названия
    def extract_name_features(dataframe):
        df = dataframe.copy()
        df['brand_from_name'] = df['name'].astype(str).str.split().str[:2].str.join(' ')
        df['name_length'] = df['name'].astype(str).str.len()
        df['name_word_count'] = df['name'].astype(str).str.split().str.len()
        premium_words = ['pro', 'plus', 'premium', 'smart', 'inverter',
                         'stainless', 'copper', 'turbo', '5g', 'air', 'conditioner']
        for word in premium_words:
            df[f'has_{word}'] = df['name'].astype(str).str.lower().str.contains(word).astype(int)
        ton_match = df['name'].astype(str).str.extract(r'(\d+\.?\d*)\s*ton', flags=re.IGNORECASE)
        df['tonnage'] = pd.to_numeric(ton_match[0], errors='coerce').fillna(0)
        df.drop(columns=['name'], inplace=True, errors='ignore')
        return df

    df = extract_name_features(df)

    # Объединённая категория
    df['category_full'] = df['main_category'].astype(str) + ' | ' + df['sub_category'].astype(str)
    df.drop(columns=['main_category', 'sub_category'], inplace=True, errors='ignore')

    # Логарифмируем количество отзывов
    if 'no_of_ratings' in df.columns:
        df['log_no_of_ratings'] = np.log1p(df['no_of_ratings'])
        df.drop(columns=['no_of_ratings'], inplace=True, errors='ignore')

    # Заполняем пропуски в категориальных колонках
    df['brand_from_name'] = df['brand_from_name'].fillna('unknown')
    df['category_full'] = df['category_full'].fillna('unknown')

    print(f"Финальный размер после очистки и инженерии: {df.shape}")
    return df, target

# 2. Подготовка признаков
def prepare_data(df):
    cat_cols = ['brand_from_name', 'category_full']
    num_cols = ['ratings', 'discount_price', 'discount_pct',
                'name_length', 'name_word_count', 'log_no_of_ratings',
                'tonnage']
    flag_cols = [col for col in df.columns if col.startswith('has_')]
    num_cols.extend(flag_cols)

    X = df[cat_cols + num_cols].copy()
    y = df['log_price']

    return X, y, cat_cols, num_cols

# 3. Трансформер для чисел
def build_num_transformer():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

# 4. Обучение с унификацией категорий
def train_evaluate(X, y, cat_cols, num_cols):
    # Разделение
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
    print(f"Размеры: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Унификация категорий
    cat_categories = {}
    for col in cat_cols:
        train_cats = X_train[col].astype(str).unique().tolist()
        all_cats = train_cats + ['unknown']
        all_cats = list(dict.fromkeys(all_cats))
        cat_categories[col] = all_cats

        X_train[col] = pd.Categorical(
            X_train[col].astype(str).fillna('unknown'),
            categories=all_cats
        )

        for df_split in (X_val, X_test):
            col_values = df_split[col].astype(str).fillna('unknown')
            mask = ~col_values.isin(train_cats)
            col_values[mask] = 'unknown'
            df_split[col] = pd.Categorical(col_values, categories=all_cats)

    # Обработка чисел
    num_transformer = build_num_transformer()
    X_train_num = num_transformer.fit_transform(X_train[num_cols])
    X_val_num = num_transformer.transform(X_val[num_cols])
    X_test_num = num_transformer.transform(X_test[num_cols])

    X_train_df = pd.DataFrame(X_train_num, columns=num_cols, index=X_train.index)
    X_val_df   = pd.DataFrame(X_val_num, columns=num_cols, index=X_val.index)
    X_test_df  = pd.DataFrame(X_test_num, columns=num_cols, index=X_test.index)
    for col in cat_cols:
        X_train_df[col] = X_train[col]
        X_val_df[col] = X_val[col]
        X_test_df[col] = X_test[col]

    model = xgb.XGBRegressor(
        n_estimators=3000,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        enable_categorical=True,
        max_cat_to_onehot=1
    )
    model.fit(
        X_train_df, y_train,
        eval_set=[(X_train_df, y_train), (X_val_df, y_val)],
        verbose=False
    )
    print(f"Лучшая итерация: {model.best_iteration}")

    y_pred_log_train = model.predict(X_train_df)
    y_pred_log_val = model.predict(X_val_df)
    y_pred_log_test = model.predict(X_test_df)

    y_pred_train = np.expm1(y_pred_log_train)
    y_pred_val = np.expm1(y_pred_log_val)
    y_pred_test = np.expm1(y_pred_log_test)
    y_true_train = np.expm1(y_train)
    y_true_val = np.expm1(y_val)
    y_true_test = np.expm1(y_test)

    def print_metrics(name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{name:5s} | RMSE: {rmse:>12,.2f} | MAE: {mae:>12,.2f} | R²: {r2:.4f}")

    print("\n--- Метрики (рубли) ---")
    print_metrics("Train", y_true_train, y_pred_train)
    print_metrics("Val",   y_true_val,   y_pred_val)
    print_metrics("Test",  y_true_test,  y_pred_test)

    importance = model.feature_importances_
    feature_names = num_cols + cat_cols
    idx = np.argsort(importance)[::-1][:10]
    print("\nТоп-10 важных признаков:")
    for i in idx:
        print(f"{feature_names[i]}: {importance[i]:.4f}")

    artifacts = {
        'model': model,
        'num_transformer': num_transformer,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'cat_categories': cat_categories
    }
    joblib.dump(artifacts, 'joblib/xgboost_price_model.pkl')
    print("\nМодель и артефакты сохранены в 'joblib/xgboost_price_model.pkl'")
    return model

# Главная функция
def main():
    file_path = datasetpath
    df, _ = load_and_clean(file_path)

    # --- Сэмплирование 10 000 случайных строк ---
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
        print(f"Отобрано случайных строк: {len(df)}")
    else:
        print(f"Данных меньше 10000, используется полный набор: {len(df)}")

    X, y, cat_cols, num_cols = prepare_data(df)
    train_evaluate(X, y, cat_cols, num_cols)

if __name__ == "__main__":
    main()