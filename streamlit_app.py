# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 파일 경로
csv_file_path = 'streamlit.csv'
metric_file_path = 'metric_summary.csv'

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data(csv_file_path)

# 날짜 처리
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
else:
    st.error("Date column not found in the CSV file.")

# 전처리: 예측 이후 실제값 제거
def preprocess_data(df):
    cutoff_date = pd.to_datetime('2020-09-28')
    cols_to_zero = ['cabbage', 'radish', 'garlic', 'onion', 'daikon', 'cilantro', 'artichoke']
    df.loc[df.index > cutoff_date, cols_to_zero] = np.nan
    return df

df = preprocess_data(df)

# 정확도 테이블 불러오기
metric_summary = pd.read_csv(metric_file_path)
metric_summary.set_index('product', inplace=True)

# 시각화 함수
def plot_predictions_over_time(df, vegetables, rolling_mean_window):
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    num_colors = len(colors)

    for i, veg in enumerate(vegetables):
        ax.plot(df.index, df[veg], label=veg, linewidth=2, color=colors[i % num_colors])
        rolling_mean = df[veg].rolling(window=rolling_mean_window).mean()
        ax.plot(df.index, rolling_mean, label=f'{veg} ({rolling_mean_window}-day Rolling Mean)', linestyle='--', color=colors[i % num_colors])

    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, color='lightgrey', linestyle='--')
    fig.tight_layout()
    st.pyplot(fig)

# 제목
st.title('🍇농산물 가격 예측 대시보드🥭')
st.markdown("왼쪽에서 품목과 예측모델, 날짜를 입력하면 특정기간 이후 예측 가격이 표시됩니다.")

# 품목 및 예측 모델 목록
product_columns = [col for col in df.columns if '_pred_' not in col and not col.startswith('Unnamed')]
sorted_vegetables = sorted(product_columns)
pred_model_columns = sorted([col for col in df.columns if '_pred_' in col])
label_map = {f"{col.split('_pred_')[0]} ({col.split('_pred_')[1]})": col for col in pred_model_columns}

# 사이드바 UI
st.sidebar.title('조회 항목 설정')
vegetables = st.sidebar.multiselect('조회 품목:', sorted_vegetables)
selected_labels = st.sidebar.multiselect('예측 모델 선택:', list(label_map.keys()))
selected_models = [label_map[label] for label in selected_labels]
start_date = st.sidebar.date_input('시작일', df.index.min())
end_date = st.sidebar.date_input('마지막일', df.index.max())
rolling_mean_window = st.sidebar.slider('Rolling Mean Window', min_value=1, max_value=30, value=7)

# 결과 시각화 및 출력
if vegetables or selected_models:
    filtered_df = df.loc[start_date:end_date]

    # 1. 그래프
    st.subheader('📈 품목별 실제 가격 + 예측 결과')
    plot_predictions_over_time(filtered_df, vegetables + selected_models, rolling_mean_window)

    # 2. 예측값만 최신순 정렬해서 출력
    with st.expander("📈 예측값 (최신순 정렬)"):
        if selected_models:
            pred_df = filtered_df[selected_models].copy()
            pred_df_sorted = pred_df.sort_index(ascending=False)
            st.dataframe(pred_df_sorted, use_container_width=True)
        else:
            st.info("예측 모델이 선택되지 않았습니다.")

    # 3. 정확도 카드
    if selected_models:
        st.subheader('📊 선택한 예측 모델의 정확도 Summary (퍼센트)')

        for model_col in selected_models:
            product = model_col.split('_pred_')[0]
            model = model_col.split('_pred_')[1]
            try:
                value = metric_summary.loc[product, model]
                percent_value = round(value * 100, 2)
                st.metric(label=f"{product} + {model}", value=f"{percent_value}%")
            except KeyError:
                st.warning(f"{product} + {model} 에 대한 정확도 정보가 없습니다.")

        st.success("✔ 정확도는 퍼센트(%)로 변환되어 위에 표시되었습니다.")

        # 4. 정확도 테이블
        selected_rows = [col.split('_pred_')[0] for col in selected_models]
        selected_cols = [col.split('_pred_')[1] for col in selected_models]
        selected_rows = list(set(selected_rows))
        selected_cols = list(set(selected_cols))

        accuracy_df = metric_summary.loc[
            metric_summary.index.intersection(selected_rows),
            metric_summary.columns.intersection(selected_cols)
        ]

        with st.expander("📋 정확도 테이블 자세히 보기"):
            st.dataframe(accuracy_df, use_container_width=True)

    # 5. 원본 filtered_df 출력
    with st.expander("🗂 Show Original Filtered DataFrame"):
        target_columns = vegetables + selected_models
        st.dataframe(filtered_df[target_columns])

# 품목 한글 안내
st.sidebar.markdown("""
  | Korean | English    |
  |--------|------------|
  | 배추   | cabbage    |
  | 무     | radish     |
  | 마늘   | garlic     |
  | 양파   | onion      |
  | 대파   | daikon     |
  | 건고추 | cilantro   |
  | 깻잎   | artichoke  |
""")
