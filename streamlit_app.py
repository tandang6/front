# app_dashboard_filtered.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 파일 경로
csv_file_path = 'streamlit_data.csv'
metric_file_path = 'metric_summary.csv'

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data(csv_file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

metric_summary = pd.read_csv(metric_file_path)
metric_summary.set_index('product', inplace=True)

# 전처리
cutoff_date = pd.to_datetime('2025-04-30')
cols_to_zero = ['cabbage', 'radish', 'garlic', 'onion', 'daikon', 'cilantro', 'artichoke']
df.loc[df.index > cutoff_date, cols_to_zero] = np.nan

# 한글 매핑
vegetable_kor_map = {
    'cabbage': '배추',
    'radish': '무',
    'garlic': '마늘',
    'onion': '양파',
    'daikon': '대파',
    'cilantro': '건고추',
    'artichoke': '깻잎'
}
def label_formatter(eng): return f"{eng} ({vegetable_kor_map[eng]})"

product_columns = list(vegetable_kor_map.keys())

# UI
st.title("📊 농산물 가격 예측 대시보드 (선택 품목 기준 모델 필터링)")
vegetables = st.sidebar.multiselect("조회 품목:", options=product_columns, format_func=label_formatter)

# 필터링된 모델 목록
available_model_cols = [col for col in df.columns if '_pred_' in col]
related_models = [col for col in available_model_cols if col.split('_pred_')[0] in vegetables]
related_labels = {f"{col.split('_pred_')[1]}": col for col in related_models}

selected_models_short = st.sidebar.multiselect("선택된 품목의 예측 모델:", options=related_labels.keys())
selected_models = [related_labels[m] for m in selected_models_short]

start_date = st.sidebar.date_input("시작일", df.index.min())
end_date = st.sidebar.date_input("종료일", df.index.max())
window = st.sidebar.slider("Rolling 평균일 수", 1, 30, 7)

if not vegetables and not selected_models:
    st.info("👈 왼쪽에서 품목과 모델을 선택하세요.")
else:
    filtered_df = df.loc[start_date:end_date]
    st.subheader("📈 실제 가격 및 예측 결과")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    for veg in vegetables:
        ax.plot(filtered_df.index, filtered_df[veg], label=label_formatter(veg))
        ax.plot(filtered_df.index, filtered_df[veg].rolling(window).mean(), linestyle='--', label=f"{label_formatter(veg)} ({window}일 평균)")

    for model_col in selected_models:
        ax.plot(filtered_df.index, filtered_df[model_col], linestyle=':', label=model_col)

    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
