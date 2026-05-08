import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from vnstock3 import Vnstock
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from arch import arch_model

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Hệ thống Giao dịch Quant Pro", layout="wide")
st.title("📊 Hệ thống Phân tích Định lượng HMM & Monte Carlo")
st.sidebar.header("Cấu hình thông số")

# --- INPUT TỪ NGƯỜI DÙNG ---
TICKER = st.sidebar.text_input("Nhập mã cổ phiếu", value="FPT").upper()
YEARS_DATA = st.sidebar.slider("Số năm dữ liệu lịch sử", 1, 5, 2)
DAYS_TO_PREDICT = st.sidebar.number_input("Số ngày dự báo", value=60)
N_SIM = st.sidebar.select_slider("Số lượng mô phỏng (N)", options=[1000, 5000, 10000], value=10000)

# Input cho Quản trị rủi ro
st.sidebar.subheader("Quản trị rủi ro")
CAPITAL = st.sidebar.number_input("Vốn đầu tư (VNĐ)", value=100000000, step=10000000)
RISK_PER_TRADE = st.sidebar.slider("Rủi ro mỗi lệnh (%)", 0.5, 5.0, 2.0) / 100

@st.cache_data(ttl=3600)
def load_data(ticker, years):
    today = datetime.now()
    start_date = (today - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    try:
        # 1. Lấy dữ liệu từ Vnstock3 (Dùng nguồn VCI ổn định hơn)
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        df = stock.trading.history(start=start_date, end=end_date)
        
        vni = Vnstock().stock(symbol='VNINDEX', source='VCI')
        df_vni = vni.trading.history(start=start_date, end=end_date)
        
        if df.empty or df_vni.empty: return None, None

        # 2. Chuẩn hóa cột ngày và gộp dữ liệu
        df['time'] = pd.to_datetime(df['time'])
        df_vni['time'] = pd.to_datetime(df_vni['time'])
        
        df = df.set_index('time')
        df_vni = df_vni.set_index('time')
        
        df_combined = pd.merge(df[['close', 'volume']], df_vni[['close']], 
                               left_index=True, right_index=True, suffixes=('', '_vni'))
        
        # 3. Tính Return & RS
        df_combined['ret_stock'] = np.log(df_combined['close'] / df_combined['close'].shift(1))
        df_combined['ret_vni'] = np.log(df_combined['close_vni'] / df_combined['close_vni'].shift(1))
        
        window = 20
        df_combined['rs_line'] = (df_combined['close'] / df_combined['close'].shift(window)) / \
                                 (df_combined['close_vni'] / df_combined['close_vni'].shift(window))
        
        # 4. GARCH(1,1) Volatility
        returns = df_combined['ret_stock'].dropna() * 100
        garch_m = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
        res_garch = garch_m.fit(disp='off')
        df_combined['volatility'] = res_garch.conditional_volatility / 100
        
        return df_combined.dropna(), df_vni
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None, None

# --- THỰC THI PHÂN TÍCH ---
df, df_vni_raw = load_data(TICKER, YEARS_DATA)

if df is not None:
    # 1. HMM Training
    X = df[['ret_stock', 'volatility']].values
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(X)
    
    # --- LOGIC SẮP XẾP TRẠNG THÁI (ĐẢM BẢO Ý NGHĨA) ---
    means = model.means_[:, 0]
    order = np.argsort(means) # [Rủi ro, Tích lũy, Tăng mạnh]
    new_labels = {order[0]: 2, order[1]: 0, order[2]: 1}
    
    raw_states = model.predict(X)
    df['state'] = [new_labels[s] for s in raw_states]
    
    state_desc = {0: "Tích lũy (Đi ngang)", 1: "Xu hướng (Tăng mạnh)", 2: "Rủi ro (Biến động xấu)"}
    curr_st = df['state'].iloc[-1]
    S0 = df['close'].iloc[-1]
    beta_val = df['ret_stock'].cov(df['ret_vni']) / df['ret_vni'].var()

    # --- HIỂN THỊ METRICS ---
    st.subheader(f"📊 Phân tích mã: {TICKER}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Giá hiện tại", f"{S0:,.0f} đ")
    m2.metric("Trạng thái HMM", state_desc[curr_st])
    m3.metric("Hệ số Beta", f"{beta_val:.2f}")

    # --- MONTE CARLO SIMULATION ---
    state_info = df[df['state'] == curr_st]
    mu, sigma = state_info['ret_stock'].mean(), state_info['ret_stock'].std()
    
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((DAYS_TO_PREDICT, N_SIM)))
    price_paths = np.zeros((DAYS_TO_PREDICT + 1, N_SIM))
    price_paths[0] = S0
    for t in range(1, DAYS_TO_PREDICT + 1):
        price_paths[t] = price_paths[t-1] * daily_returns[t-1]
    
    final_prices = price_paths[-1, :]
    expected_price = np.mean(final_prices)
    win_rate_val = np.mean(final_prices > S0) * 100
    p25, p50, p75 = np.percentile(final_prices, [25, 50, 75])

    # --- QUẢN TRỊ RỦI RO ---
    st.divider()
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.subheader("🛡️ Kế hoạch giao dịch")
        stop_loss = p25 * 0.98 
        risk_amt = CAPITAL * RISK_PER_TRADE
        dist_to_sl = S0 - stop_loss
        shares_to_buy = int(risk_amt / dist_to_sl) if dist_to_sl > 0 else 0
        
        st.write(f"- **Điểm dừng lỗ (SL):** {stop_loss:,.0f} đ")
        st
