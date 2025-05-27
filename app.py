import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

df = pd.read_excel('C:\Repoz2\Kolchak\Sprav.xlsx')

group_Товарная_категория = sorted(df["Товарная категория"].dropna().unique())
group_Товарная_группа = sorted(df["Товарная группа"].dropna().unique())
group_Целевая_группа = sorted(df["Целевая группа"].dropna().unique())
group_Ассортимент = sorted(df["Ассортимент"].dropna().unique())
group_Производство_обобщенное = sorted(df["Производство обобщенное"].dropna().unique())
group_атрибут1 = sorted(df["атрибут1"].dropna().unique())
group_атрибут2 = sorted(df["атрибут2"].dropna().unique())
group_атрибут4 = sorted(df["атрибут4"].dropna().unique())
group_Страна_оригинала = sorted(df["Страна оригинала"].dropna().unique())
group_Страна_производства = sorted(df["Страна производства"].dropna().unique())
group_Тип_ткани = sorted(df["Тип ткани"].dropna().unique())
group_Цвет = sorted(df["Цвет"].dropna().unique())
group_Однотонность = sorted(df["Однотонность"].dropna().unique())
group_Элементы_дизайна = sorted(df["Элементы дизайна"].dropna().unique())
group_Посадка = sorted(df["Посадка"].dropna().unique())
group_Модность = sorted(df["Модность"].dropna().unique())
group_Тип_продукта = sorted(df["Тип продукта"].dropna().unique())
group_Коллекция = sorted(df["Коллекция"].dropna().unique())
group_Атрибут_цены = sorted(df["Атрибут цены"].dropna().unique())
group_МЕСЯЦ_PMM = sorted(df["МЕСЯЦ PMM"].dropna().unique())



class MixedCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1):
        self.threshold = threshold
        self.one_hot_cols = []
        self.target_encoding_maps = {}

    def fit(self, X, y):
        self.target = y.name if hasattr(y, 'name') else 'target'
        for col in X.select_dtypes(include='object').columns:
            if X[col].nunique() < self.threshold:
                self.one_hot_cols.append(col)
            else:
                self.target_encoding_maps[col] = (
                    X[[col]].join(y).groupby(col)[self.target].mean().to_dict()
                )
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.one_hot_cols:
            dummies = pd.get_dummies(X_new[col], prefix=col, drop_first=True, dtype=int)
            X_new = X_new.drop(col, axis=1)
            X_new = pd.concat([X_new, dummies], axis=1)

        for col, mapping in self.target_encoding_maps.items():
            X_new[col] = X_new[col].map(mapping).fillna(0)

        return X_new

# === Загрузка модели ===
model = joblib.load("C:\\Repoz\\Kolchak\\et.pkl")
scaler = joblib.load("C:\\Repoz\\Kolchak\\scaler.pkl")
encoder = joblib.load("C:\\Repoz\\Kolchak\\encoder.pkl")

# === Интерфейс ===
st.set_page_config(page_title="ExtraTrees Прогноз", layout="centered")
st.title("🌲 ExtraTreesClassifier — прогноз лидерства товара")

with st.form("input_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        category = st.selectbox("Товарная категория",group_Товарная_категория)
        group = st.selectbox("Товарная группа", group_Товарная_группа)
        target_group = st.selectbox("Целевая группа", group_Целевая_группа)
        assortment = st.selectbox("Ассортимент", group_Ассортимент)
        size_count = st.number_input("Кол-во размеров", 0.0, 20.0, 3.0)
        cost = st.number_input("Себестоимость", 0.0, 2000.0, 100.0)
        retail_price = st.number_input("Цена розничная", 0.0, 20000.0, 499.0)
    with col2:
        origin = st.selectbox("Страна оригинала", group_Страна_оригинала )
        country = st.selectbox("Страна производства", group_Страна_производства)
        fabric = st.selectbox("Тип ткани", group_Тип_ткани)
        color = st.selectbox("Цвет", group_Цвет)
        design = st.selectbox("Элементы дизайна", group_Элементы_дизайна)
        fit = st.selectbox("Посадка", group_Посадка)
        fashion = st.selectbox("Модность", group_Модность)
    with col3:
        product_type = st.selectbox("Тип продукта", group_Тип_продукта)
        collection = st.selectbox("Коллекция", group_Коллекция)
        prod_type = st.selectbox("Производство обобщенное", group_Производство_обобщенное)
        imu = st.number_input("IMU", 0.0, 1.5, 0.65)
        attr1 = st.selectbox("Атрибут1", group_атрибут1)
        attr2 = st.selectbox("Атрибут2", group_атрибут2)
        attr4 = st.selectbox("Атрибут4", group_атрибут4)
    with col4:   
        mono = st.selectbox("Однотонность", group_Однотонность)
        price_attr = st.selectbox("Атрибут цены", group_Атрибут_цены)
        year = st.number_input("Год", 2016, 2030, 2024)
        week = st.number_input("НЕДЕЛЯ PMM", 1, 53, 20)
        month = st.selectbox("Месяц",group_МЕСЯЦ_PMM)
        order_qty = st.number_input("Заказ", 0.0, 50000.0, 5000.0)
        depth = st.number_input("Глубина на модель", 0.0, 10.0, 3.0)

    submitted = st.form_submit_button("Предсказать")

if submitted:
    numeric_df = pd.DataFrame([{
        "Кол-во размеров": size_count,
        "Себестоимость": cost,
        "Цена розничная": retail_price,
        "IMU": imu,
        "Год": year,
        "НЕДЕЛЯ PMM": week,
        "Заказ": order_qty,
        "Глубина на модель": depth
        
    }])

    categorical_df = pd.DataFrame([{
        "Товарная категория": category,
        "Товарная группа": group,
        "Целевая группа": target_group,
        "Ассортимент": assortment,
        "Производство обобщенное": prod_type,
        "атрибут1": attr1,
        "атрибут2": attr2,
        "атрибут4": attr4,
        "Страна оригинала": origin,
        "Страна производства": country,
        "Тип ткани": fabric,
        "Цвет": color,
        "Однотонность": mono,
        "Элементы дизайна": design,
        "Посадка": fit,
        "Модность": fashion,
        "Тип продукта": product_type,
        "Коллекция": collection,
        "Атрибут цены": price_attr,
        'МЕСЯЦ PMM': month
    }])
    encoded_df = encoder.transform(categorical_df)
    full_input = pd.concat([numeric_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    scaled_input = scaler.transform(full_input)


    proba = model.predict_proba(scaled_input)[0, 1]
    pred = model.predict(scaled_input)[0]

    st.subheader("🔍 Результат")
    st.metric("Вероятность лидерства", f"{proba:.2%}")
    if pred == 1:
        st.success("✅ Модель товара, вероятно, станет лидером продаж.")
    else:
        st.warning("⚠️ Модель товара, вероятно, не станет лидером.")