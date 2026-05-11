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

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Hệ thống Giao dịch Quant Pro", layout="wide")
st.title("📊 Hệ thống Phân tích Định lượng HMM & Monte Carlo")

# --- 2. INPUT TỪ NGƯỜI DÙNG (SIDEBAR) ---
st.sidebar.header("Cấu hình thông số")
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
        q_ticker = Quote(symbol=ticker, source='KBS')
        df = q_ticker.history(start=start_date, end=end_date, interval="1D")
        q_vni = Quote(symbol='VNINDEX', source='KBS')
        df_vni = q_vni.history(start=start_date, end=end_date, interval="1D")
        
        if df.empty or df_vni.empty: return None, None

        if df['close'].iloc[-1] < 1000: df['close'] = df['close'] * 1000
        
        df_combined = pd.merge(df[['close', 'volume']], df_vni[['close']], 
                               left_index=True, right_index=True, suffixes=('', '_vni'))
        
        df_combined['ret_stock'] = np.log(df_combined['close'] / df_combined['close'].shift(1))
        df_combined['ret_vni'] = np.log(df_combined['close_vni'] / df_combined['close_vni'].shift(1))
        
        window = 20
        df_combined['rs_line'] = (df_combined['close'] / df_combined['close'].shift(window)) / \
                                 (df_combined['close_vni'] / df_combined['close_vni'].shift(window))
        
        returns = df_combined['ret_stock'].dropna() * 100
        res_garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal').fit(disp='off')
        df_combined['volatility'] = res_garch.conditional_volatility / 100
        
        return df_combined.dropna(), df_vni
    except:
        return None, None

# --- 3. THỰC THI LOGIC ---
df, df_vni = load_data(TICKER, YEARS_DATA)

if df is not None:
    # HMM Training & State Sorting
    X = df[['ret_stock', 'volatility']].values
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42).fit(X)
    
    means = model.means_[:, 0]
    order = np.argsort(means)  
    new_labels = {order[0]: 2, order[1]: 0, order[2]: 1} # 2:Rủi ro, 0:Tích lũy, 1:Tăng
    df['state'] = [new_labels[s] for s in model.predict(X)]
    
    state_desc = {0: "Tích lũy (Đi ngang)", 1: "Xu hướng (Tăng mạnh)", 2: "Rủi ro (Biến động xấu)"}
    curr_st = df['state'].iloc[-1]
    S0 = df['close'].iloc[-1]

    # Monte Carlo Simulation
    state_info = df[df['state'] == curr_st]
    # Fallback nếu dữ liệu trạng thái quá ít
    if len(state_info) < 5: state_info = df
    
    mu, sigma = state_info['ret_stock'].mean(), state_info['ret_stock'].std()
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((DAYS_TO_PREDICT, N_SIM)))
    
    price_paths = np.zeros((DAYS_TO_PREDICT + 1, N_SIM))
    price_paths[0] = S0
    for t in range(1, DAYS_TO_PREDICT + 1):
        price_paths[t] = price_paths[t-1] * daily_returns[t-1]

    # Chỉ số thống kê
    final_prices = price_paths[-1, :]
    win_rate_val = np.mean(final_prices > S0) * 100
    p25, p50, p75 = np.percentile(final_prices, [25, 50, 75])
    expected_price = np.mean(final_prices)

    # --- 4. HIỂN THỊ DASHBOARD ---
    st.subheader(f"🚀 Phân tích mã: {TICKER}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Giá hiện tại", f"{S0:,.0f} đ")
    m2.metric("Xác suất tăng giá", f"{win_rate_val:.1f}%")
    
    with m3:
        if curr_st == 1: st.success(state_desc[curr_st])
        elif curr_st == 0: st.warning(state_desc[curr_st])
        else: st.error(state_desc[curr_st])

    # Quản trị rủi ro
    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        st.subheader("🛡️ Chiến lược Giao dịch")
        stop_loss = p25 * 0.98
        dist_to_sl = S0 - stop_loss
        risk_amt = CAPITAL * RISK_PER_TRADE
        
        # Sửa lỗi mua quá số vốn (Logic Position Sizing)
        if dist_to_sl > 0:
            shares_potential = int(risk_amt / dist_to_sl)
            shares_to_buy = min(shares_potential, int(CAPITAL / S0))
        else:
            shares_to_buy = 0
            
        st.write(f"- **Dừng lỗ (SL) khuyến nghị:** {stop_loss:,.0f} đ")
        st.write(f"- **Khối lượng nên mua:** {shares_to_buy:,} CP")
        st.write(f"- **Tổng ngân sách dự kiến:** {shares_to_buy * S0:,.0f} đ")

    with r2:
        st.subheader("📊 Chỉ số Quant")
        beta_val = df['ret_stock'].cov(df['ret_vni']) / df['ret_vni'].var()
        st.write(f"- **Hệ số Beta:** {beta_val:.2f}")
        st.write(f"- **Sức mạnh RS (20D):** {'Khỏe' if df['rs_line'].iloc[-1] > 1 else 'Yếu'} hơn VNINDEX")
        st.write(f"- **Lợi nhuận kỳ vọng:** {((expected_price/S0)-1)*100:+.1f}%")

    # --- 5. BIỂU ĐỒ 4 TẦNG TIÊU CHUẨN ---
    st.divider()
    fig = plt.figure(figsize=(14, 18), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 0.8, 1.5, 1], hspace=0.3)

    # Tầng 1: HMM
    ax1 = fig.add_subplot(gs[0])
    ax1_vni = ax1.twinx()
    ax1_vni.plot(df.index, df['close_vni'], color='#FFD700', alpha=0.1, linestyle='--')
    
    ax1.plot(df.index, df['close'], color='white', alpha=0.2)
    colors_hmm = ['#F4D03F', '#2ECC71', '#E74C3C']
    for i in range(3):
        st_data = df[df['state'] == i]
        ax1.scatter(st_data.index, st_data['close'], c=colors_hmm[i], s=15, label=state_desc[i])
    ax1.set_title("HMM MARKET STATES", color='white', loc='left')
    ax1.legend(facecolor='#0E1117', labelcolor='white')

    # Tầng 2: Volume
    ax2 = fig.add_subplot(gs[1])
    colors_vol = np.where(df['close'] >= df['close'].shift(1), '#2ECC71', '#E74C3C')
    ax2.bar(df.index, df['volume'], color=colors_vol, alpha=0.5)
    ax2.set_title("VOLUME", color='white', loc='left')

    # Tầng 3: Monte Carlo Cone
    ax3 = fig.add_subplot(gs[2])
    forecast_dates = [df.index[-1] + timedelta(days=i) for i in range(DAYS_TO_PREDICT + 1)]
    ax3.fill_between(forecast_dates, np.percentile(price_paths, 5, axis=1), np.percentile(price_paths, 95, axis=1), color='#00CCFF', alpha=0.1)
    ax3.fill_between(forecast_dates, np.percentile(price_paths, 25, axis=1), np.percentile(price_paths, 75, axis=1), color='#00CCFF', alpha=0.2)
    ax3.plot(forecast_dates, p50_path := np.percentile(price_paths, 50, axis=1), color='#00CCFF', lw=2)
    ax3.axhline(S0, color='white', linestyle='--', alpha=0.5)
    ax3.set_title("MONTE CARLO PRICE FORECAST (90% CI)", color='white', loc='left')

    # Tầng 4: Backtest
    ax4 = fig.add_subplot(gs[3])
    df['strat_ret'] = np.where(df['state'].shift(1) == 1, df['ret_stock'], 0)
    df['cum_strat'] = np.exp(df['strat_ret'].cumsum())
    ax4.fill_between(df.index, df['cum_strat'], 1, color='#2ECC71', alpha=0.2)
    ax4.plot(df.index, df['cum_strat'], color='#2ECC71', lw=1.5)
    ax4.set_title("STRATEGY PERFORMANCE", color='white', loc='left')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#0E1117')
        ax.tick_params(colors='gray', labelsize=8)
        ax.grid(alpha=0.05)

    st.pyplot(fig)

    # --- 6. BẢNG DỮ LIỆU ---
    st.divider()
    st.write("**Kịch bản giá dự báo:**")
    st.table(pd.DataFrame({
        "Kịch bản": ["Xấu (P5)", "Thận trọng (P25)", "Trung vị (P50)", "Lạc quan (P75)", "Đột biến (P95)"],
        "Giá dự báo": [f"{np.percentile(final_prices, p):,.0f} đ" for p in [5, 25, 50, 75, 95]],
        "Tỷ suất (%)": [f"{(np.percentile(final_prices, p)/S0 - 1):+.1%}" for p in [5, 25, 50, 75, 95]]
    }))

else:
    st.error("⚠️ Lỗi tải dữ liệu. Vui lòng kiểm tra lại mã cổ phiếu hoặc kết nối mạng.")
