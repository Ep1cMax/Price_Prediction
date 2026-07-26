import joblib
import pandas as pd
import numpy as np
import re
import os

class PricePredictor:
    """Загружает модель и артефакты из единого файла, предсказывает цену в рублях."""

    def __init__(self, model_path='joblib/xgboost_price_model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели {model_path} не найден.")
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.num_transformer = artifacts['num_transformer']
        self.cat_cols = artifacts['cat_cols']
        self.num_cols = artifacts['num_cols']
        self.cat_categories = artifacts['cat_categories']  # dict: col -> список категорий
        self.premium_words = ['pro', 'plus', 'premium', 'smart', 'inverter',
                              'stainless', 'copper', 'turbo', '5g', 'air', 'conditioner']

    def _engineer_features(self, df):
        df = df.copy()
        if 'name' in df.columns:
            df['brand_from_name'] = df['name'].astype(str).str.split().str[:2].str.join(' ')
            df['name_length'] = df['name'].astype(str).str.len()
            df['name_word_count'] = df['name'].astype(str).str.split().str.len()
            for word in self.premium_words:
                df[f'has_{word}'] = df['name'].astype(str).str.lower().str.contains(word).astype(int)
            ton_match = df['name'].astype(str).str.extract(r'(\d+\.?\d*)\s*ton', flags=re.IGNORECASE)
            df['tonnage'] = pd.to_numeric(ton_match[0], errors='coerce').fillna(0)
        if 'main_category' in df.columns and 'sub_category' in df.columns:
            df['category_full'] = df['main_category'].astype(str) + ' | ' + df['sub_category'].astype(str)
        if 'no_of_ratings' in df.columns:
            cleaned = df['no_of_ratings'].astype(str).str.replace(',', '', regex=False)
            no_of_ratings = pd.to_numeric(cleaned, errors='coerce')
            df['log_no_of_ratings'] = np.log1p(no_of_ratings)
        if 'discount_price' in df.columns:
            df['discount_price'] = pd.to_numeric(df['discount_price'], errors='coerce')
            # Если есть actual_price – вычислим discount_pct, иначе 0
            if 'actual_price' in df.columns:
                actual = pd.to_numeric(df['actual_price'], errors='coerce')
                df['discount_pct'] = (actual - df['discount_price']) / actual
                df['discount_pct'] = df['discount_pct'].fillna(0).clip(0, 1)
            else:
                df['discount_pct'] = 0.0
        else:
            # Если колонки discount_price нет, создадим заглушку
            df['discount_price'] = 0.0
            df['discount_pct'] = 0.0
        if 'ratings' in df.columns:
            df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
        # Заполнение категориальных полей
        if 'brand_from_name' in df.columns:
            df['brand_from_name'] = df['brand_from_name'].fillna('unknown')
        else:
            df['brand_from_name'] = 'unknown'
        if 'category_full' in df.columns:
            df['category_full'] = df['category_full'].fillna('unknown')
        else:
            df['category_full'] = 'unknown'
        return df

    def predict(self, data, in_rub=True):
        data = self._engineer_features(data)

        # Обработка категориальных колонок
        for col in self.cat_cols:
            if col not in data:
                data[col] = 'unknown'
            col_values = data[col].astype(str).fillna('unknown')
            known_cats = self.cat_categories.get(col, [])
            # Известные категории без 'unknown'
            known_no_unknown = [c for c in known_cats if c != 'unknown']
            mask = ~col_values.isin(known_no_unknown)
            col_values[mask] = 'unknown'
            data[col] = pd.Categorical(col_values, categories=known_cats if known_cats else None)

        # Числовые признаки
        X_num = data[self.num_cols].copy()
        X_num = pd.DataFrame(self.num_transformer.transform(X_num),
                             columns=self.num_cols, index=data.index)
        X_df = X_num.copy()
        for col in self.cat_cols:
            X_df[col] = data[col]

        # Предсказание (модель обучена на log_price)
        y_pred_log = self.model.predict(X_df)
        y_pred_rub = np.expm1(y_pred_log)   # обратное логарифмирование
        if in_rub:
            return y_pred_rub
        else:
            # перевод в рупии (используем курс из обучения)
            return y_pred_rub / 0.8083

    def get_feature_importance(self):
        """Возвращает DataFrame с именами признаков и их важностью."""
        importance = self.model.feature_importances_
        feature_names = self.num_cols + self.cat_cols
        return pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)