import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import numpy as np
import requests

st.set_page_config(page_title="환율 예측 AI", layout="wide")
st.title("💱 환율 예측 AI 시스템")

# ====== STEP 1: 한국은행 API로 환율 가져오기 ======
@st.cache_data
def fetch_exchange_rate():
    API_KEY = "99BO6UEVOS1ZHTSHK79J" 
    start_date = "20240101"
    end_date = datetime.today().strftime("%Y%m%d")
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/036Y001/DD/{start_date}/{end_date}/0002"

    try:
        response = requests.get(url)
        data = response.json()
        rows = data['StatisticSearch']['row']
        df = pd.DataFrame(rows)
        df = df[['TIME', 'DATA_VALUE']]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)
        return df
    except Exception as e:
        st.error(f"API 데이터 불러오기 실패: {e}")
        return None

df = fetch_exchange_rate()
if df is None:
    st.stop()

# ====== STEP 2: 사용자 예측 설정 ======
mode = st.radio("예측 방식", ["Prophet 기반 예측", "시연용 더미 데이터"])
start_date = st.date_input("예측 시작 날짜", datetime.today())
days = st.slider("예측 일 수", min_value=1, max_value=30, value=7)

# ====== STEP 3: 예측 ======
if mode == "Prophet 기반 예측":
    try:
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        result = forecast[['ds', 'yhat']].tail(days)
        result.columns = ['날짜', '예측 환율 (KRW/USD)']
    except Exception as e:
        st.error(f"예측 실패: {e}")
        st.stop()
else:
    dates = [start_date + timedelta(days=i) for i in range(days)]
    rates = np.random.normal(1300, 5, size=days)
    result = pd.DataFrame({
        "날짜": dates,
        "예측 환율 (KRW/USD)": rates
    })

# ====== STEP 4: 출력 ======
st.line_chart(result.set_index("날짜"))
st.dataframe(result)