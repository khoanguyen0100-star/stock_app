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
    
    # --- PHẦN THÊM VÀO: SẮP XẾP TRẠNG THÁI ---
    means = model.means_[:, 0]
    order = np.argsort(means)  # Thứ tự: Rủi ro -> Tích lũy -> Tăng
    new_labels = {order[0]: 2, order[1]: 0, order[2]: 1}
    
    raw_states = model.predict(X)
    df['state'] = [new_labels[s] for s in raw_states]
    # ----------------------------------------
    
    state_desc = {0: "Tích lũy (Đi ngang)", 1: "Xu hướng (Tăng mạnh)", 2: "Rủi ro (Biến động xấu)"}
    curr_st = df['state'].iloc[-1]
    S0 = df['close'].iloc[-1]

    # --- KHỐI HIỂN THỊ GIÁ HIỆN TẠI ---
    st.subheader(f"📊 Dữ liệu thực tế: {TICKER}")
    col_price, col_state = st.columns(2)
    with col_price:
        st.metric("Giá hiện tại", f"{S0:,.0f} đ") 
    
    # --- PHẦN IN TRẠNG THÁI HIỆN TẠI RA MÀN HÌNH ---
    with col_state:
        if curr_st == 1:
            st.success(f"Trạng thái hiện tại: {state_desc[curr_st]}")
        elif curr_st == 0:
            st.warning(f"Trạng thái hiện tại: {state_desc[curr_st]}")
        else:
            st.error(f"Trạng thái hiện tại: {state_desc[curr_st]}")

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
   # --- PHẦN THAY THẾ: BIỂU ĐỒ 4 TẦNG CHUYÊN NGHIỆP ---
    st.divider()
    st.subheader("📈 Phân tích Kỹ thuật & Dự báo Monte Carlo")
    
    # Khởi tạo khung biểu đồ với 4 tầng
    fig = plt.figure(figsize=(14, 16), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 0.8, 1.5, 1], hspace=0.3)

    # Tầng 1: HMM Trạng thái & VNINDEX
    ax1 = fig.add_subplot(gs[0])
    ax1_vni = ax1.twinx()
    ax1_vni.plot(df.index, df['close_vni'], color='#FFD700', alpha=0.15, linestyle='--', label='VNINDEX')
    ax1_vni.tick_params(axis='y', labelcolor='#FFD700', labelsize=8)
    
    ax1.plot(df.index, df['close'], color='white', alpha=0.3, lw=1)
    colors_hmm = ['#F4D03F', '#2ECC71', '#E74C3C'] # Vàng (Tích lũy), Xanh (Tăng), Đỏ (Rủi ro)
    for i in range(3):
        st_data = df[df['state'] == i]
        ax1.scatter(st_data.index, st_data['close'], c=colors_hmm[i], s=20, label=state_desc[i])
    
    ax1.set_title(f"TRẠNG THÁI THỊ TRƯỜNG & TƯƠNG QUAN VNINDEX", color='white', fontsize=12, loc='left')
    ax1.legend(loc='upper left', facecolor='#0E1117', edgecolor='white', labelcolor='white', fontsize=9)

    # Tầng 2: Khối lượng (Volume)
    ax2 = fig.add_subplot(gs[1])
    colors_vol = np.where(df['ret_stock'] >= 0, '#2ECC71', '#E74C3C')
    ax2.bar(df.index, df['volume'], color=colors_vol, alpha=0.6)
    ax2.set_title("KHỐI LƯỢNG GIAO DỊCH", color='white', fontsize=10, loc='left')

    # Tầng 3: MONTE CARLO CONE (Nón dự báo xác suất)
    ax3 = fig.add_subplot(gs[2])
    forecast_dates = [df.index[-1] + timedelta(days=i) for i in range(DAYS_TO_PREDICT + 1)]
    
    # Tính các dải xác suất
    p_upper = np.percentile(price_paths, 95, axis=1)
    p_lower = np.percentile(price_paths, 5, axis=1)
    p_mid_upper = np.percentile(price_paths, 75, axis=1)
    p_mid_lower = np.percentile(price_paths, 25, axis=1)
    p_median = np.percentile(price_paths, 50, axis=1)

    ax3.fill_between(forecast_dates, p_lower, p_upper, color='#00CCFF', alpha=0.1, label='Vùng xác suất 90%')
    ax3.fill_between(forecast_dates, p_mid_lower, p_mid_upper, color='#00CCFF', alpha=0.2, label='Vùng xác suất 50%')
    ax3.plot(forecast_dates, p_median, color='#00CCFF', lw=2, label='Đường trung vị (P50)')
    ax3.axhline(S0, color='white', linestyle='--', alpha=0.5, label='Giá hiện tại')
    
    ax3.set_title(f"MÔ PHỎNG BIẾN ĐỘ GIÁ {DAYS_TO_PREDICT} NGÀY TỚI", color='white', fontsize=12, loc='left')
    ax3.legend(loc='upper left', facecolor='#0E1117', edgecolor='white', labelcolor='white', fontsize=9)

    # Tầng 4: HIỆU SUẤT CHIẾN LƯỢC (Backtest Area)
    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(df.index, df['cum_strategy'], 1, where=(df['cum_strategy'] >= 1), color='#2ECC71', alpha=0.3)
    ax4.fill_between(df.index, df['cum_strategy'], 1, where=(df['cum_strategy'] < 1), color='#E74C3C', alpha=0.3)
    ax4.plot(df.index, df['cum_strategy'], color='#00FF00', lw=1.5, label='Lợi nhuận cộng dồn')
    ax4.set_title("HIỆU SUẤT CHIẾN LƯỢC DỰA TRÊN TRẠNG THÁI HMM", color='white', fontsize=10, loc='left')

    # Định dạng chung cho tất cả các trục
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#0E1117')
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.1, color='white')
        for spine in ax.spines.values():
            spine.set_color('#444444')

    plt.tight_layout()
    st.pyplot(fig)
    # --- HẾT PHẦN THAY THẾ ---

    # --- KHỐI BACKTEST ---
    st.divider()
    st.subheader("📈 Kiểm định hiệu quả chiến lược (Backtest)")
    df['strategy_ret'] = np.where(df['state'].shift(1) == 1, df['ret_stock'], 0)
    df['cum_market'] = np.exp(df['ret_stock'].cumsum())
    df['cum_strategy'] = np.exp(df['strategy_ret'].cumsum())
    
    total_ret = (df['cum_strategy'].iloc[-1] - 1) * 100
    mkt_ret = (df['cum_market'].iloc[-1] - 1) * 100
    max_dd = (df['cum_strategy'] / df['cum_strategy'].cummax() - 1).min() * 100
    # TÍNH TOÁN TRƯỚC (Phải nằm trên dòng b1.metric)
    total_ret = (df['cum_strategy'].iloc[-1] - 1) * 100
    mkt_ret = (df['cum_market'].iloc[-1] - 1) * 100
    diff = total_ret - mkt_ret  # ĐẢM BẢO CÓ DÒNG NÀY

    b1, b2, b3 = st.columns(3)
    b1.metric("Lợi nhuận HMM", f"{total_ret:.1f}%", delta=f"{diff:+.1f}% vs Market")
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

else:
    st.error("⚠️ Không thể tải dữ liệu. Vui lòng kiểm tra lại mã cổ phiếu.")
