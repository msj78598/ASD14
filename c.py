import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import tensorflow as tf
import io
import urllib.request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

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

download_model_files()

train_data_path = os.path.join(DATA_DIR, "final_classified_loss_with_reasons_60_percent_ordered.xlsx")
if not os.path.exists(train_data_path):
    st.error(f"❌ خطأ: ملف البيانات غير موجود! تأكد من رفعه: {train_data_path}")
    st.stop()

df = pd.read_excel(train_data_path)
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
df["Loss_Status"] = df["Loss_Status"].apply(lambda x: 1 if x == "Loss" else 0)

features = ["V1", "V2", "V3", "A1", "A2", "A3"]
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

try:
    autoencoder = load_model(MODEL_FILES["autoencoder_model.keras"], compile=False)
    xgb = joblib.load(MODEL_FILES["xgboost_model.pkl"])
    lgbm = joblib.load(MODEL_FILES["lightgbm_model.pkl"])
    stacked_model = joblib.load(MODEL_FILES["stacked_model.pkl"])
except Exception as e:
    st.error(f"❌ خطأ أثناء تحميل النماذج: {e}")
    st.stop()

reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.square(reconstructions - X_scaled), axis=1)
threshold = np.percentile(mse, 95)

st.set_page_config(page_title="نظام اكتشاف الفاقد الكهربائي", page_icon="⚡", layout="wide")
st.title("⚡ نظام احترافي لاكتشاف وتحليل حالات الفاقد الكهربائي")
st.markdown("### 🏢 تحليل هندسي دقيق لتحديد جميع مؤشرات الفاقد المحتملة")
st.markdown("---")

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

    df_test["Anomaly"] = anomalies
    df_test["XGB_Prediction"] = xgb_preds
    df_test["LGBM_Prediction"] = lgbm_preds
    df_test["Stacked_Prediction"] = stacked_preds

    def technical_explanation(row):
        reasons = []
        
        # فازة بفازة
        for i in range(1, 4):
            v, a = row[f'V{i}'], row[f'A{i}']
            if v == 0 and a == 0:
                reasons.append(f"⚠️ فازة V{i}: لا جهد ولا تيار → فازة معطلة")
            elif v == 0 and a > 0:
                reasons.append(f"⚡ فازة V{i}: جهد صفر وتيار موجود → فاقد مؤكد")
            elif v <= 50 and a > 0:
                reasons.append(f"⚡ فازة V{i}: جهد منخفض وتيار موجود → فاقد محتمل")
            elif v >= 50 and a == 0:
                reasons.append(f"🔌 فازة V{i}: جهد طبيعي بدون حمل → تحقق من استهلاك المشترك")
        
        # من النماذج
        if row['Anomaly'] and row['Stacked_Prediction'] == 1:
            reasons.append("🤖 توافق النماذج والشذوذ → فاقد عالي الاحتمال")
        elif row['Anomaly']:
            reasons.append("📊 شذوذ بالنمط دون اتفاق النماذج → تحقق إضافي")
        
        if not reasons:
            return "✔️ لا مؤشرات فاقد واضحة"
        return " | ".join(reasons)

    df_test['Loss_Explanation'] = df_test.apply(technical_explanation, axis=1)

    confirmed_loss = df_test.apply(lambda r: any([(r[f'V{i}'] <= 50 and r[f'A{i}'] > 0) or (r[f'V{i}']==0 and r[f'A{i}']==0) for i in range(1,4)]), axis=1)
    high_priority_cases = df_test[(df_test['Anomaly'] & (df_test['Stacked_Prediction']==1)) | confirmed_loss]

    st.write(f"🔍 عدد الحالات المصنفة كشذوذ: {len(df_test[df_test['Anomaly']])}")
    st.write(f"🚨 عدد الحالات ذات الأولوية العالية بعد التحليل الفني: {len(high_priority_cases)}")

    st.subheader("🔍 تفاصيل الحالات ذات الأولوية العالية")
    st.dataframe(high_priority_cases[["Meter Number", "V1", "V2", "V3", "A1", "A2", "A3", "Loss_Explanation"]])

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        high_priority_cases.to_excel(writer, index=False, sheet_name="High Priority Losses")
        df_test.to_excel(writer, index=False, sheet_name="All Predictions")
        writer.close()
    excel_buffer.seek(0)

    st.download_button("📥 تحميل تقرير الحالات ذات الأولوية العالية", data=excel_buffer, file_name="High_Priority_Losses.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("📥 تحميل تقرير جميع الحالات", data=excel_buffer, file_name="All_Predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.markdown("👨‍💻 **تطوير  : مشهور العباس | 00966553339838**")
