import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_file_path = 'streamlit.csv'
metric_file_path = 'metric_summary.csv'

# 데이터 로드 함수
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# 날짜 처리 및 인덱스 지정
df = load_data(csv_file_path)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
else:
    st.error("❌ CSV 파일에 'date' 컬럼이 없습니다.")

# 예측 데이터 이후 날짜 제거
def preprocess_data(df):
    cutoff_date = pd.to_datetime('2020-09-28')
    cols_to_zero = ['cabbage', 'radish', 'garlic', 'onion', 'daikon', 'cilantro', 'artichoke']
    df.loc[df.index > cutoff_date, cols_to_zero] = np.nan
    return df

# 시계열 그래프 함수
def plot_predictions_over_time(df, vegetables, rolling_mean_window):
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    nums_colors = len(colors)

    for i, veg in enumerate(vegetables):
        ax.plot(df.index, df[veg], label=veg, linewidth=2, color=colors[i % nums_colors])
        rolling_mean = df[veg].rolling(window=rolling_mean_window).mean()
        ax.plot(df.index, rolling_mean, label=f'{veg} ({rolling_mean_window}-day Mean)', linestyle='--', color=colors[i % nums_colors])

    ax.set_xlabel('Date')
    ax.set_ylabel('Price (KRW/kg)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    st.pyplot(fig)

# 전처리 적용
df = preprocess_data(df)

# 정확도 요약 불러오기
metric_summary = pd.read_csv(metric_file_path)
metric_summary.set_index('product', inplace=True)

# UI 시작
st.title('🍇 농산물 가격 예측 대시보드 🥭')
st.markdown("왼쪽에서 품목과 예측 설정을 선택하면 시계열 그래프와 모델 정확도 요약이 표시됩니다.")

# 사이드바 구성
st.sidebar.title('조회 조건 설정')
start_date = st.sidebar.date_input('시작일', df.index.min())
end_date = st.sidebar.date_input('마지막일', df.index.max())

st.sidebar.title('품목 선택')
sorted_vegetables = sorted(df.columns)
vegetables = st.sidebar.multiselect('조회 품목:', sorted_vegetables)
rolling_mean_window = st.sidebar.slider('이동 평균 기간 (일)', 1, 30, 7)

st.sidebar.markdown("""
| 한글명 | 변수명 |
|--------|--------|
| 배추   | cabbage |
| 무     | radish  |
| 마늘   | garlic  |
| 양파   | onion   |
| 대파   | daikon  |
| 건고추 | cilantro|
| 깻잎   | artichoke|
""")

# 선택된 조건에 따른 데이터 필터링
filtered_df = df.loc[start_date:end_date]

# 메인 콘텐츠
if vegetables:
    st.subheader('📈 품목별 가격 추이')
    plot_predictions_over_time(filtered_df, vegetables, rolling_mean_window)

if st.checkbox('📋 데이터 보기'):
    st.dataframe(filtered_df)

st.subheader('🧪 예측 모델 정확도')
st.dataframe(metric_summary)

# 하단 출처 및 설명 표시
st.markdown("""
---
📌 **데이터 출처:** [농산물유통정보(KAMIS)](http://www.kamis.or.kr)  
🔎 본 대시보드의 예측 결과는 KAMIS에서 제공한 도매가격 데이터를 기반으로 생성되었습니다.  
예측 모델은 과거 가격 패턴을 학습하여 향후 농산물 가격 변동을 추정합니다.  
본 결과는 참고용이며 실제 가격과는 차이가 발생할 수 있습니다.
""")
