import os
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import urllib.request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ===============================
# 🔹 تعريف المسارات الرئيسية
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = BASE_DIR
os.makedirs(MODELS_DIR, exist_ok=True)

GITHUB_REPO = "https://raw.githubusercontent.com/msj78598/ASD14/main/"
MODEL_FILES = {
    "autoencoder_model.keras": os.path.join(MODELS_DIR, "autoencoder_model.keras"),
    "xgboost_model.pkl": os.path.join(MODELS_DIR, "xgboost_model.pkl"),
    "lightgbm_model.pkl": os.path.join(MODELS_DIR, "lightgbm_model.pkl"),
    "stacked_model.pkl": os.path.join(MODELS_DIR, "stacked_model.pkl"),
}

def download_model_files():
    for file_name, file_path in MODEL_FILES.items():
        if not os.path.exists(file_path):
            url = GITHUB_REPO + file_name
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"✅ تم تحميل {file_name} بنجاح!")
            except Exception as e:
                print(f"❌ فشل تحميل {file_name}: {e}")

# تحميل النماذج إذا لم تكن موجودة
download_model_files()

# ===============================
train_data_path = os.path.join(DATA_DIR, "final_classified_loss_with_reasons_60_percent_ordered.xlsx")

if not os.path.exists(train_data_path):
    st.error(f"❌ خطأ: ملف البيانات غير موجود! تأكد من رفعه: {train_data_path}")
    st.stop()

df = pd.read_excel(train_data_path)
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
df["Loss_Status"] = df["Loss_Status"].apply(lambda x: 1 if x == "Loss" else 0)

features = ["V1", "V2", "V3", "A1", "A2", "A3"]
X = df[features].values
y = df["Loss_Status"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

try:
    autoencoder = load_model(MODEL_FILES["autoencoder_model.keras"], compile=False)
    xgb = joblib.load(MODEL_FILES["xgboost_model.pkl"])
    lgbm = joblib.load(MODEL_FILES["lightgbm_model.pkl"])
    stacked_model = joblib.load(MODEL_FILES["stacked_model.pkl"])
    print("✅ تم تحميل جميع النماذج بنجاح!")
except Exception as e:
    st.error(f"❌ خطأ أثناء تحميل النماذج: {e}")
    st.stop()

reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.square(reconstructions - X_scaled), axis=1)
threshold = np.percentile(mse, 95)

st.set_page_config(page_title="نظام اكتشاف الفاقد الكهربائي", page_icon="⚡", layout="wide")
st.title("⚡ نظام اكتشاف حالات الفاقد الكهربائي المحتملة")
st.markdown("### 🏢 استخدام نهج التعلم الآلي والتعلم العميق في تحليل بيانات الأحمال الكهربائية لكشف حالات الفاقد")
st.markdown("---")

st.subheader("📥 تحميل نموذج البيانات المطلوب تحليله")
template_file = os.path.join(DATA_DIR, "The_data_frame_file_to_be_analyzed.xlsx")

if os.path.exists(template_file):
    st.download_button("📥 تحميل نموذج البيانات", open(template_file, "rb"), file_name="The_data_frame_file_to_be_analyzed.xlsx")

uploaded_file = st.file_uploader("🔼 رفع ملف بيانات الأحمال")
if uploaded_file:
    df_test = pd.read_excel(uploaded_file)
    X_test = df_test[features].values
    X_test_scaled = scaler.transform(X_test)

    reconstructions = autoencoder.predict(X_test_scaled)
    mse_test = np.mean(np.square(reconstructions - X_test_scaled), axis=1)
    anomalies = mse_test > threshold

    xgb_preds = xgb.predict(X_test_scaled)
    lgbm_preds = lgbm.predict(X_test_scaled)
    stacked_preds = stacked_model.predict(X_test_scaled)

    def explain_loss(row):
        if row["Stacked_Prediction"] == 1:
            return "⚠ جميع النماذج تتفق على أن هذه حالة فاقد شبه مؤكدة بسبب التناقض في بيانات الحمل والجهد."
        elif row["Anomaly"]:
            return "⚠ محتملة بسبب تحليل الحمل والتوتر."
        return "✔ لا يوجد فاقد ظاهر."

    df_test["Anomaly"] = anomalies
    df_test["XGB_Prediction"] = xgb_preds
    df_test["LGBM_Prediction"] = lgbm_preds
    df_test["Stacked_Prediction"] = stacked_preds
    df_test["Loss_Explanation"] = df_test.apply(explain_loss, axis=1)

    high_priority_cases = df_test[(df_test["Anomaly"]) &
                                  (df_test["XGB_Prediction"] == 1) &
                                  (df_test["LGBM_Prediction"] == 1) &
                                  (df_test["Stacked_Prediction"] == 1)]

    def apply_custom_filters(df):
        all_zero_current = (df['A1'] == 0) & (df['A2'] == 0) & (df['A3'] == 0)
        normal_voltage = (df['V1'] >= 50) & (df['V2'] >= 50) & (df['V3'] >= 50)
        exclude_all_zero_current = all_zero_current & normal_voltage

        two_phases_ok = (
            ((df['V1'] >= 50) & (df['V2'] >= 50) & (df['V3'] == 0)) |
            ((df['V1'] >= 50) & (df['V3'] >= 50) & (df['V2'] == 0)) |
            ((df['V2'] >= 50) & (df['V3'] >= 50) & (df['V1'] == 0))
        )

        def is_confirmed_loss(row):
            return ((row['V1'] <= 50 and row['A1'] > 0) or
                    (row['V2'] <= 50 and row['A2'] > 0) or
                    (row['V3'] <= 50 and row['A3'] > 0))

        confirmed_loss_mask = df.apply(is_confirmed_loss, axis=1)

        final_mask = (~exclude_all_zero_current & ~two_phases_ok) | confirmed_loss_mask

        return df[final_mask]

    high_priority_cases = apply_custom_filters(high_priority_cases)

    st.write(f"🔍 عدد العدادات المصنفة كفاقد: {len(df_test[df_test['Anomaly']])}")
    st.write(f"🚨 عدد العدادات ذات الأولوية العالية بعد الفلترة: {len(high_priority_cases)}")

    st.subheader("🔍 تفاصيل الحالات ذات الأولوية العالية")
    st.dataframe(high_priority_cases[["V1", "V2", "V3", "A1", "A2", "A3", "Loss_Explanation"]])

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        high_priority_cases.to_excel(writer, index=False, sheet_name="High Priority Losses")
        df_test.to_excel(writer, index=False, sheet_name="All Predicted Losses")
        writer.close()
    excel_buffer.seek(0)

    st.download_button("📥 تحميل حالات الفاقد ذات الأولوية العالية", data=excel_buffer, file_name="High_Priority_Losses.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("📥 تحميل جميع حالات الفاقد المحتملة", data=excel_buffer, file_name="Predicted_Losses.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
