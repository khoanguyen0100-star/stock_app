import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from vnstock import Quote
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Phân tích & Dự báo Chứng khoán", layout="wide")
st.title("📊 Công cụ Phân tích HMM & Monte Carlo (Dark Mode)")
st.sidebar.header("Cấu hình thông số")

# --- INPUT TỪ NGƯỜI DÙNG ---
TICKER = st.sidebar.text_input("Nhập mã cổ phiếu", value="FPT").upper()
YEARS_DATA = st.sidebar.slider("Số năm dữ liệu lịch sử", 1, 5, 2)
DAYS_TO_PREDICT = st.sidebar.number_input("Số ngày dự báo", value=60)
N_SIM = st.sidebar.select_slider("Số lượng mô phỏng (N)", options=[1000, 5000, 10000], value=10000)

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

        if df_vni['close'].mean() < 100:
            df_vni['close'] = df_vni['close'] * 1000
        
        df_combined = pd.merge(df[['close', 'volume']], df_vni[['close']], 
                               left_index=True, right_index=True, suffixes=('', '_vni'))
        
        df_combined['ret_stock'] = np.log(df_combined['close'] / df_combined['close'].shift(1))
        df_combined['ret_vni'] = np.log(df_combined['close_vni'] / df_combined['close_vni'].shift(1))
        df_combined['volatility'] = df_combined['ret_stock'].rolling(window=10).std()
        df_combined = df_combined.dropna()
        
        return df_combined, df_vni
    except:
        return None, None

# --- THỰC THI PHÂN TÍCH ---
df, df_vni = load_data(TICKER, YEARS_DATA)

if df is not None:
    # 1. Tính Beta
    beta = df['ret_stock'].cov(df['ret_vni']) / df['ret_vni'].var()

    # 2. Huấn luyện HMM
    X = df[['ret_stock', 'volatility']].values
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(X)
    df['state'] = model.predict(X)
    
    state_desc = {0: "Tích lũy (Đi ngang)", 1: "Xu hướng (Tăng mạnh)", 2: "Rủi ro (Biến động xấu)"}
    curr_st = df['state'].iloc[-1]
    
    last_price = df['close'].iloc[-1]
    S0 = last_price * 1000 if last_price < 1000 else last_price

    # 3. Monte Carlo
    state_info = df[df['state'] == curr_st]
    mu, sigma = state_info['ret_stock'].mean(), state_info['ret_stock'].std()
    
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((DAYS_TO_PREDICT, N_SIM)))
    price_paths = np.zeros((DAYS_TO_PREDICT + 1, N_SIM))
    price_paths[0] = S0
    for t in range(1, DAYS_TO_PREDICT + 1):
        price_paths[t] = price_paths[t-1] * daily_returns[t-1]
    
    final_prices = price_paths[-1, :]
    win_rate = np.mean(final_prices > S0) * 100

    # --- HIỂN THỊ KẾT QUẢ ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá hiện tại", f"{S0:,.0f} đ")
    col2.metric("Hệ số Beta", f"{beta:.2f}")
    col3.metric("Xác suất lãi", f"{win_rate:.1f}%")
    col4.metric("Trạng thái", state_desc[curr_st])

    st.divider()

    # --- BIỂU ĐỒ DARK MODE ---
    plt.style.use('dark_background') 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 22), gridspec_kw={'height_ratios': [2, 1, 1.5]})
    fig.patch.set_facecolor('#0E1117') 

    colors_hmm = ['#00FF00', '#FFFF00', '#FF0000'] 

    # Tầng 1: Giá & VNINDEX
    scale = 1000 if last_price < 1000 else 1
    ax1_vni = ax1.twinx()
    ax1_vni.plot(df.index, df['close_vni'], color='orange', alpha=0.3, linestyle='--', label='VNINDEX')
    ax1_vni.tick_params(axis='y', colors='white')
    
    ax1.plot(df.index, df['close'] * scale, color='#1E90FF', alpha=0.3)
    for i in range(3):
        st_data = df[df['state'] == i]
        ax1.scatter(st_data.index, st_data['close'] * scale, c=colors_hmm[i], s=25, label=state_desc[i])
    
    ax1.set_title(f"Tương quan {TICKER} vs VNINDEX", color='white', fontweight='bold', fontsize=16)
    ax1.tick_params(axis='both', colors='white')
    ax1.legend(loc='upper left', facecolor='#1E1E1E')

    # Tầng 2: Volume (Xanh dương - Đỏ)
    colors_vol = np.where(df['ret_stock'] >= 0, '#00CCFF', '#FF3333')
    ax2.bar(df.index, df['volume'], color=colors_vol, alpha=0.9)
    ax2.set_facecolor('#0E1117')
    ax2.tick_params(axis='both', colors='white')
    ax2.set_title("Khối lượng giao dịch (Xanh: Tăng | Đỏ: Giảm)", color='white')

    # Tầng 3: Monte Carlo
    kde = gaussian_kde(final_prices)
    x_range = np.linspace(min(final_prices), max(final_prices), 500)
    ax3.plot(x_range, kde(x_range), color="#00CCFF", lw=2.5)
    ax3.fill_between(x_range, kde(x_range), where=(x_range >= S0), color='#00FF00', alpha=0.2, label='Vùng Lãi')
    ax3.fill_between(x_range, kde(x_range), where=(x_range < S0), color='#FF0000', alpha=0.2, label='Vùng Lỗ')
    ax3.axvline(S0, color='white', linestyle='--')
    ax3.tick_params(axis='both', colors='white')
    ax3.set_title(f"Dự báo giá sau {DAYS_TO_PREDICT} ngày", color='white')

    plt.tight_layout()
    st.pyplot(fig)

    # --- MA TRẬN XÁC SUẤT (TRANSITION MATRIX) ---
    st.subheader("📌 Ma trận xác suất dịch chuyển trạng thái")
    fig_h, ax_h = plt.subplots(figsize=(8, 5))
    fig_h.patch.set_facecolor('#0E1117')
    sns.heatmap(model.transmat_, annot=True, fmt=".2f", cmap='magma',
                xticklabels=[state_desc[i] for i in range(3)], 
                yticklabels=[state_desc[i] for i in range(3)], ax=ax_h, cbar=False)
    ax_h.set_title("Xác suất chuyển trạng thái trong phiên tiếp theo", color='white')
    ax_h.tick_params(colors='white')
    st.pyplot(fig_h)
    st.info("Ví dụ: Nếu ô [Xu hướng, Xu hướng] là 0.90, nghĩa là có 90% khả năng ngày mai vẫn giữ xu hướng tăng.")

else:
    st.error(f"⚠️ Không thể tải dữ liệu cho mã {TICKER}.")
