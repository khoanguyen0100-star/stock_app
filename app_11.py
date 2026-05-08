import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from vnstock import Quote
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

st.sidebar.subheader("Quản trị rủi ro")
CAPITAL = st.sidebar.number_input("Vốn đầu tư (VNĐ)", value=100000000, step=10000000)
RISK_PER_TRADE = st.sidebar.slider("Rủi ro mỗi lệnh (%)", 0.5, 5.0, 2.0) / 100

@st.cache_data(ttl=3600)
def load_data(ticker, years):
    today = datetime.now()
    start_date = (today - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    try:
        # Sử dụng nguồn dữ liệu 'VCI' hoặc 'TCBS' để ổn định hơn 'KBS'
        q_ticker = Quote(symbol=ticker, source='VCI')
        df = q_ticker.history(start=start_date, end=end_date, interval="1D")
        q_vni = Quote(symbol='VNINDEX', source='VCI')
        df_vni = q_vni.history(start=start_date, end=end_date, interval="1D")
        
        if df is None or df.empty or df_vni is None or df_vni.empty: return None, None

        # Chuẩn hóa giá (Fix lỗi hiển thị sai đơn vị giá)
        if df['close'].iloc[-1] < 1000: df['close'] = df['close'] * 1000
        if df_vni['close'].mean() < 100: df_vni['close'] = df_vni['close'] * 1000
        
        df_combined = pd.merge(df[['close', 'volume']], df_vni[['close']], 
                               left_index=True, right_index=True, suffixes=('', '_vni'))
        
        df_combined['ret_stock'] = np.log(df_combined['close'] / df_combined['close'].shift(1))
        df_combined['ret_vni'] = np.log(df_combined['close_vni'] / df_combined['close_vni'].shift(1))
        
        window = 20
        df_combined['rs_line'] = (df_combined['close'] / df_combined['close'].shift(window)) / \
                                 (df_combined['close_vni'] / df_combined['close_vni'].shift(window))
        
        returns = df_combined['ret_stock'].dropna() * 100
        garch_m = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
        res_garch = garch_m.fit(disp='off')
        df_combined['volatility'] = res_garch.conditional_volatility / 100
        
        return df_combined.dropna(), df_vni
    except:
        return None, None

# --- THỰC THI PHÂN TÍCH ---
df, df_vni = load_data(TICKER, YEARS_DATA)

if df is not None:
    # 1. Beta & HMM Training
    beta_val = df['ret_stock'].cov(df['ret_vni']) / df['ret_vni'].var()
    X = df[['ret_stock', 'volatility']].values
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(X)
    
    # Sắp xếp trạng thái: Rủi ro (2) -> Tích lũy (0) -> Tăng mạnh (1)
    means = model.means_[:, 0]
    order = np.argsort(means)
    new_labels = {order[0]: 2, order[1]: 0, order[2]: 1}
    
    raw_states = model.predict(X)
    df['state'] = [new_labels[s] for s in raw_states]
    
    state_desc = {0: "Tích lũy (Đi ngang)", 1: "Xu hướng (Tăng mạnh)", 2: "Rủi ro (Biến động xấu)"}
    curr_st = df['state'].iloc[-1]
    S0 = df['close'].iloc[-1]

    st.subheader(f"📊 Dữ liệu thực tế: {TICKER}")
    col_price, col_state = st.columns(2)
    with col_price:
        st.metric("Giá hiện tại", f"{S0:,.0f} đ") 
    
    with col_state:
        if curr_st == 1: st.success(f"Trạng thái hiện tại: {state_desc[curr_st]}")
        elif curr_st == 0: st.warning(f"Trạng thái hiện tại: {state_desc[curr_st]}")
        else: st.error(f"Trạng thái hiện tại: {state_desc[curr_st]}")

    # 2. Monte Carlo Simulation
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

    # 3. Quản trị rủi ro & Hiệu suất
    st.divider()
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        st.subheader("🛡️ Kế hoạch giao dịch")
        stop_loss = p25 * 0.98 
        risk_amt = CAPITAL * RISK_PER_TRADE
        dist_to_sl = S0 - stop_loss
        shares_to_buy = int(risk_amt / dist_to_sl) if dist_to_sl > 0 else 0

        st.write(f"- **Dừng lỗ (SL):** {stop_loss:,.0f} đ")
        st.write(f"- **Mua:** {shares_to_buy:,} CP")
        st.write(f"- **Vốn giải ngân:** {shares_to_buy * S0:,.0f} đ")

    with r_col2:
        st.subheader("📊 Chỉ số hiệu suất")
        st.metric("Xác suất tăng giá", f"{win_rate_val:.1f}%")
        st.metric("Hệ số Beta", f"{beta_val:.2f}")

    # 4. Backtest (Fix lỗi hiển thị nghìn phần trăm)
    st.divider()
    st.subheader("📈 Kiểm định hiệu quả chiến lược (Backtest)")
    df['strategy_ret'] = np.where(df['state'].shift(1) == 1, df['ret_stock'], 0)
    df['cum_market'] = np.exp(df['ret_stock'].cumsum())
    df['cum_strategy'] = np.exp(df['strategy_ret'].cumsum())
    
    total_ret = (df['cum_strategy'].iloc[-1] - 1) * 100
    mkt_ret = (df['cum_market'].iloc[-1] - 1) * 100
    diff = total_ret - mkt_ret
    max_dd = (df['cum_strategy'] / df['cum_strategy'].cummax() - 1).min() * 100

    b1, b2, b3 = st.columns(3)
    # FIX: Ép kiểu hiển thị để tránh Streamlit tự nhân 100
    b1.metric("Lợi nhuận HMM", f"{total_ret:.1f}%", delta=f"{diff:+.1f}% vs Market")
    b2.metric("Lợi nhuận Mua & Giữ", f"{mkt_ret:.1f}%")
    b3.metric("Sụt giảm tối đa (MDD)", f"{max_dd:.1f}%", delta_color="inverse")

    # Vẽ biểu đồ
    fig_bt, ax_bt = plt.subplots(figsize=(14, 4))
    fig_bt.patch.set_facecolor('#0E1117')
    ax_bt.plot(df.index, df['cum_strategy'], label='Chiến lược HMM', color='#00FF00', lw=2)
    ax_bt.plot(df.index, df['cum_market'], label='Mua & Giữ', color='white', alpha=0.3)
    ax_bt.set_facecolor('#0E1117')
    ax_bt.tick_params(colors='white')
    ax_bt.legend()
    st.pyplot(fig_bt)

else:
    st.error("⚠️ Không thể tải dữ liệu. Hãy thử đổi nguồn dữ liệu hoặc kiểm tra lại mã cổ phiếu.")
