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
        # 1. Lấy dữ liệu giá Stock & VNINDEX
        q_ticker = Quote(symbol=ticker, source='KBS')
        df = q_ticker.history(start=start_date, end=end_date, interval="1D")
        q_vni = Quote(symbol='VNINDEX', source='KBS')
        df_vni = q_vni.history(start=start_date, end=end_date, interval="1D")
        
        if df.empty or df_vni.empty: return None, None

        # 2. Chuẩn hóa giá
        if df['close'].iloc[-1] < 1000: df['close'] = df['close'] * 1000
        if df_vni['close'].mean() < 100: df_vni['close'] = df_vni['close'] * 1000
        
        # 3. Gộp dữ liệu & Tính Return
        df_combined = pd.merge(df[['close', 'volume']], df_vni[['close']], 
                               left_index=True, right_index=True, suffixes=('', '_vni'))
        
        df_combined['ret_stock'] = np.log(df_combined['close'] / df_combined['close'].shift(1))
        df_combined['ret_vni'] = np.log(df_combined['close_vni'] / df_combined['close_vni'].shift(1))
        
        # 4. Tính RS (Relative Strength - Sức mạnh tương quan)
        window = 20
        df_combined['rs_line'] = (df_combined['close'] / df_combined['close'].shift(window)) / \
                                 (df_combined['close_vni'] / df_combined['close_vni'].shift(window))
        
        # 5. GARCH(1,1) Volatility
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
    df['state'] = model.predict(X)
    
    state_desc = {0: "Tích lũy (Đi ngang)", 1: "Xu hướng (Tăng mạnh)", 2: "Rủi ro (Biến động xấu)"}
    curr_st = df['state'].iloc[-1]
    S0 = df['close'].iloc[-1]

    # --- KHỐI THÔNG BÁO KHUYẾN NGHỊ ---
  

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
    expected_return = (expected_price - S0) / S0 * 100
    win_rate_val = np.mean(final_prices > S0) * 100
    p25, p50, p75 = np.percentile(final_prices, [25, 50, 75])

    # --- KHỐI QUẢN TRỊ RỦI RO & HIỆU SUẤT ---
    st.divider()
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        st.subheader("🛡️ Kế hoạch giao dịch")
        stop_loss = p25 * 0.98 
        risk_amt = CAPITAL * RISK_PER_TRADE
        dist_to_sl = S0 - stop_loss
        
        if dist_to_sl > 0:
            shares_to_buy = int(risk_amt / dist_to_sl)
            total_cost = shares_to_buy * S0
        else:
            shares_to_buy = 0
            total_cost = 0

        st.write(f"- **Điểm dừng lỗ tối ưu (SL):** {stop_loss:,.0f} đ")
        st.write(f"- **Số lượng cổ phiếu nên mua:** {shares_to_buy:,} CP")
        st.write(f"- **Tổng giá trị giải ngân:** {total_cost:,.0f} đ")
        st.write(f"- **Tỷ lệ Reward/Risk:** {(expected_price - S0)/(S0 - stop_loss) if dist_to_sl > 0 else 0:.2f}")

    with r_col2:
        st.subheader("📊 Chỉ số hiệu suất")
        st.metric("Xác suất tăng giá", f"{win_rate_val:.1f}%")
        st.metric("Hệ số Beta (vs VNINDEX)", f"{beta_val:.2f}")
        rs_status = "Khỏe hơn thị trường" if df['rs_line'].iloc[-1] > 1 else "Yếu hơn thị trường"
        st.write(f"- **Sức mạnh tương quan (RS):** {rs_status}")

    st.divider()

    # --- BIỂU ĐỒ 3 TẦNG ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 0.8, 1.2]})
    fig.patch.set_facecolor('#0E1117') 

    # Tầng 1: HMM Scatter + VNINDEX tham chiếu
    ax1_vni = ax1.twinx() 
    ax1_vni.plot(df.index, df['close_vni'], color='white', alpha=0.1, linestyle='--', label='VNINDEX')
    ax1_vni.set_ylabel("VNINDEX", color='white', alpha=0.3)
    ax1_vni.tick_params(axis='y', colors='grey', labelsize=8) # Đã sửa lỗi alpha ở đây

    ax1.plot(df.index, df['close'], color='white', alpha=0.3)
    colors_hmm = ['#FFFF00', '#00FF00', '#FF0000']
    for i in range(3):
        st_data = df[df['state'] == i]
        ax1.scatter(st_data.index, st_data['close'], c=colors_hmm[i], s=25, label=state_desc[i])
    
    ax1.set_title(f"Phân tích tương quan: {TICKER} giữa VNINDEX", fontsize=12, color='white')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_facecolor('#0E1117')
    ax1.tick_params(colors='white')

    # Tầng 2: Volume bar
    colors_vol = np.where(df['ret_stock'] >= 0, '#26a69a', '#ef5350')
    ax2.bar(df.index, df['volume'], color=colors_vol, alpha=0.7)
    ax2.set_title("Khối lượng giao dịch", fontsize=10, color='white')
    ax2.set_facecolor('#0E1117')
    ax2.tick_params(colors='white')

    # Tầng 3: Monte Carlo KDE
    kde = gaussian_kde(final_prices)
    x_range = np.linspace(min(final_prices), max(final_prices), 1000)
    ax3.plot(x_range, kde(x_range), color="#00CCFF", lw=2)
    ax3.fill_between(x_range, kde(x_range), where=(x_range >= S0), color='#00FF00', alpha=0.2)
    ax3.axvline(S0, color='white', linestyle='--', label='Giá hiện tại')
    ax3.axvline(expected_price, color='#FFFF00', label=f'Kỳ vọng: {expected_price:,.0f}')
    ax3.set_title(f"Phân phối xác suất dự báo sau {DAYS_TO_PREDICT} ngày", fontsize=12, color='white')
    ax3.legend()
    ax3.set_facecolor('#0E1117')
    ax3.tick_params(colors='white')

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

    # --- BẢNG DỮ LIỆU & HEATMAP ---
    st.divider()
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.write("**Kịch bản dự báo theo percentiles:**")
        st.table(pd.DataFrame({
            "Kịch bản": ["Thận trọng (P25)", "Trung vị (P50)", "Kỳ vọng", "Lạc quan (P75)"],
            "Giá dự báo": [f"{p25:,.0f} đ", f"{p50:,.0f} đ", f"{expected_price:,.0f} đ", f"{p75:,.0f} đ"],
            "Lợi nhuận": [f"{(p25-S0)/S0:+.1%}", f"{(p50-S0)/S0:+.1%}", f"{expected_return/100:+.1%}", f"{(p75-S0)/S0:+.1%}"]
        }))
    with col_t2:
        st.write("**Ma trận chuyển trạng thái (HMM Transition):**")
        fig_h, ax_h = plt.subplots(figsize=(4, 3))
        sns.heatmap(model.transmat_, annot=True, fmt=".2f", cmap='viridis', 
                    xticklabels=["S0","S1","S2"], yticklabels=["S0","S1","S2"], ax=ax_h, cbar=False)
        st.pyplot(fig_h)

    # --- KHỐI BACKTEST ---
    st.divider()
    st.subheader("📈 Kiểm định hiệu quả chiến lược (Backtest)")
    df['strategy_ret'] = np.where(df['state'].shift(1) == 1, df['ret_stock'], 0)
    df['cum_market'] = np.exp(df['ret_stock'].cumsum())
    df['cum_strategy'] = np.exp(df['strategy_ret'].cumsum())
    
    total_ret = (df['cum_strategy'].iloc[-1] - 1) * 100
    mkt_ret = (df['cum_market'].iloc[-1] - 1) * 100
    max_dd = (df['cum_strategy'] / df['cum_strategy'].cummax() - 1).min() * 100

    b1, b2, b3 = st.columns(3)
    b1.metric("Lợi nhuận HMM", f"{total_ret:.1f}%", f"{total_ret-mkt_ret:+.1f}% vs Market")
    b2.metric("Lợi nhuận Mua & Giữ", f"{mkt_ret:.1f}%")
    b3.metric("Sụt giảm tối đa (MDD)", f"{max_dd:.1f}%")

    fig_bt, ax_bt = plt.subplots(figsize=(14, 4))
    fig_bt.patch.set_facecolor('#0E1117')
    ax_bt.plot(df.index, df['cum_strategy'], label='Chiến lược HMM', color='#00FF00', lw=2)
    ax_bt.plot(df.index, df['cum_market'], label='Mua & Giữ', color='white', alpha=0.3)
    ax_bt.set_facecolor('#0E1117')
    ax_bt.tick_params(colors='white')
    ax_bt.legend()
    st.pyplot(fig_bt)

else:
    st.error("⚠️ Không thể tải dữ liệu. Vui lòng kiểm tra lại mã cổ phiếu.")
