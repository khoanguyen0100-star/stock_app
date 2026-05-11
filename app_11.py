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
        q_ticker = Quote(symbol=ticker, source='KBS')
        df = q_ticker.history(start=start_date, end=end_date, interval="1D")
        q_vni = Quote(symbol='VNINDEX', source='KBS')
        df_vni = q_vni.history(start=start_date, end=end_date, interval="1D")
        if df.empty or df_vni.empty: return None, None
        if df['close'].iloc[-1] < 1000: df['close'] *= 1000
        if df_vni['close'].mean() < 100: df_vni['close'] *= 1000
        
        df_combined = pd.merge(df[['close', 'volume']], df_vni[['close']], left_index=True, right_index=True, suffixes=('', '_vni'))
        df_combined['ret_stock'] = np.log(df_combined['close'] / df_combined['close'].shift(1))
        df_combined['ret_vni'] = np.log(df_combined['close_vni'] / df_combined['close_vni'].shift(1))
        
        window = 20
        df_combined['rs_line'] = (df_combined['close'] / df_combined['close'].shift(window)) / (df_combined['close_vni'] / df_combined['close_vni'].shift(window))
        
        returns = df_combined['ret_stock'].dropna() * 100
        res_garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal').fit(disp='off')
        df_combined['volatility'] = res_garch.conditional_volatility / 100
        return df_combined.dropna(), df_vni
    except: return None, None

# --- THỰC THI PHÂN TÍCH ---
df, df_vni = load_data(TICKER, YEARS_DATA)

if df is not None:
    # 1. HMM Training & Sorting
    X = df[['ret_stock', 'volatility']].values
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42).fit(X)
    order = np.argsort(model.means_[:, 0])
    new_labels = {order[0]: 2, order[1]: 0, order[2]: 1} # 2:Rủi ro, 0:Tích lũy, 1:Tăng
    df['state'] = [new_labels[s] for s in model.predict(X)]
    state_desc = {0: "Tích lũy (Vàng)", 1: "Xu hướng (Xanh)", 2: "Rủi ro (Đỏ)"}
    curr_st, S0 = df['state'].iloc[-1], df['close'].iloc[-1]

    # 2. Monte Carlo (KDE Logic)
    state_info = df[df['state'] == curr_st]
    if len(state_info) < 5: state_info = df
    mu, sigma = state_info['ret_stock'].mean(), state_info['ret_stock'].std()
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((DAYS_TO_PREDICT, N_SIM)))
    price_paths = np.zeros((DAYS_TO_PREDICT + 1, N_SIM))
    price_paths[0] = S0
    for t in range(1, DAYS_TO_PREDICT + 1):
        price_paths[t] = price_paths[t-1] * daily_returns[t-1]
    
    final_prices = price_paths[-1, :]
    p25, p50, p75 = np.percentile(final_prices, [25, 50, 75])
    expected_price = np.mean(final_prices)

    # 3. GIAO DIỆN HIỂN THỊ
    st.subheader(f"📊 Phân tích định lượng: {TICKER}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Giá hiện tại", f"{S0:,.0f} đ")
    m2.metric("Xác suất tăng giá", f"{np.mean(final_prices > S0)*100:.1f}%")
    with m3:
        if curr_st == 1: st.success(f"Trạng thái: {state_desc[curr_st]}")
        elif curr_st == 0: st.warning(f"Trạng thái: {state_desc[curr_st]}")
        else: st.error(f"Trạng thái: {state_desc[curr_st]}")

    # --- NÂNG CẤP BIỂU ĐỒ CHUYÊN NGHIỆP ---
    st.divider()
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 14), facecolor='#0E1117')
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 0.7, 1.2], hspace=0.3)

    # Tầng 1: HMM Price & VNINDEX
    ax1 = fig.add_subplot(gs[0])
    ax1_vni = ax1.twinx()
    ax1_vni.plot(df.index, df['close_vni'], color='#FFD700', alpha=0.1, lw=1.5, label='VNINDEX')
    ax1.plot(df.index, df['close'], color='white', alpha=0.3, lw=1)
    
    colors_hmm = ['#F4D03F', '#2ECC71', '#E74C3C'] # Vàng, Xanh, Đỏ
    for i in range(3):
        idx = df[df['state'] == i].index
        ax1.scatter(idx, df.loc[idx, 'close'], c=colors_hmm[i], s=20, label=state_desc[i], edgecolors='none', alpha=0.8)
    
    ax1.set_title(f"HMM STATE CLASSIFICATION vs VNINDEX", loc='left', fontsize=12, fontweight='bold', color='white')
    ax1.legend(loc='upper left', frameon=False)
    ax1.grid(True, alpha=0.05)

    # Tầng 2: Volume Professional
    ax2 = fig.add_subplot(gs[1])
    vol_colors = np.where(df['ret_stock'] >= 0, '#2ECC71', '#E74C3C')
    ax2.bar(df.index, df['volume'], color=vol_colors, alpha=0.6, width=0.8)
    ax2.set_title("TRADE VOLUME", loc='left', fontsize=10, color='gray')
    ax2.grid(False)

    # Tầng 3: Monte Carlo KDE (Sắc nét)
    ax3 = fig.add_subplot(gs[2])
    kde = gaussian_kde(final_prices)
    x = np.linspace(min(final_prices), max(final_prices), 1000)
    y = kde(x)
    ax3.plot(x, y, color='#00CCFF', lw=2.5, label='Xác suất mật độ (KDE)')
    ax3.fill_between(x, y, where=(x >= S0), color='#2ECC71', alpha=0.3, label='Vùng có lãi')
    ax3.fill_between(x, y, where=(x < S0), color='#E74C3C', alpha=0.3, label='Vùng thua lỗ')
    
    ax3.axvline(S0, color='white', ls='--', alpha=0.6, label=f'Giá hiện tại ({S0:,.0f})')
    ax3.axvline(expected_price, color='#FFD700', ls='-', lw=2, label=f'Kỳ vọng ({expected_price:,.0f})')
    
    ax3.set_title(f"DỰ BÁO XÁC SUẤT SAU {DAYS_TO_PREDICT} NGÀY", loc='left', fontsize=12, fontweight='bold')
    ax3.legend(frameon=False, fontsize=9)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#0E1117')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9, colors='gray')

    st.pyplot(fig)

    # --- THÔNG TIN CHI TIẾT ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**🛡️ Quản trị rủi ro**")
        sl = p25 * 0.98
        shares = int((CAPITAL * RISK_PER_TRADE) / (S0 - sl)) if S0 > sl else 0
        st.info(f"- Dừng lỗ (SL): {sl:,.0f} đ\n- Khối lượng: {shares:,} CP\n- RR Ratio: {(expected_price-S0)/(S0-sl):.2f}")
    with c2:
        st.write("**📈 Kịch bản giá**")
        st.table(pd.DataFrame({
            "Mức": ["Thận trọng", "Trung vị", "Lạc quan"],
            "Giá": [f"{p25:,.0f}", f"{p50:,.0f}", f"{p75:,.0f}"],
            "ROI": [f"{(p25-S0)/S0:+.1%}", f"{(p50-S0)/S0:+.1%}", f"{(p75-S0)/S0:+.1%}"]
        }))
    with c3:
        st.write("**🔥 Hiệu suất Backtest**")
        df['strat_ret'] = np.where(df['state'].shift(1) == 1, df['ret_stock'], 0)
        cum_hmm = (np.exp(df['strat_ret'].cumsum()).iloc[-1] - 1) * 100
        cum_mkt = (np.exp(df['ret_stock'].cumsum()).iloc[-1] - 1) * 100
        st.metric("Lợi nhuận HMM", f"{cum_hmm:.1f}%", delta=f"{cum_hmm - cum_mkt:.1f}% vs Market")

else:
    st.error("⚠️ Không thể tải dữ liệu.")
