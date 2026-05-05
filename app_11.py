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
st.title("📊 Công cụ Phân tích HMM & Monte Carlo (Professional Edition)")
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

        # Chuẩn hóa giá nếu đơn vị là 1.0 (như KBS hay bị)
        if df['close'].iloc[-1] < 1000: df['close'] = df['close'] * 1000
        if df_vni['close'].mean() < 100: df_vni['close'] = df_vni['close'] * 1000
        
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
    S0 = df['close'].iloc[-1]

    # 3. Monte Carlo & Tính toán Giá kỳ vọng
    state_info = df[df['state'] == curr_st]
    mu, sigma = state_info['ret_stock'].mean(), state_info['ret_stock'].std()
    
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((DAYS_TO_PREDICT, N_SIM)))
    price_paths = np.zeros((DAYS_TO_PREDICT + 1, N_SIM))
    price_paths[0] = S0
    for t in range(1, DAYS_TO_PREDICT + 1):
        price_paths[t] = price_paths[t-1] * daily_returns[t-1]
    
    # TRÍCH XUẤT THÔNG SỐ KỲ VỌNG
    final_prices = price_paths[-1, :]
    expected_price = np.mean(final_prices)
    expected_return = (expected_price - S0) / S0 * 100
    win_rate = np.mean(final_prices > S0) * 100
    
    # Các mốc mục tiêu (Quartiles)
    p25 = np.percentile(final_prices, 25)
    p50 = np.percentile(final_prices, 50)
    p75 = np.percentile(final_prices, 75)

    # --- HIỂN THỊ KẾT QUẢ ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Giá hiện tại", f"{S0:,.0f} đ")
    c2.metric("Giá kỳ vọng", f"{expected_price:,.0f} đ", f"{expected_return:+.1f}%")
    c3.metric("Xác suất lãi", f"{win_rate:.1f}%")
    c4.metric("Hệ số Beta", f"{beta:.2f}")
    c5.metric("Trạng thái hiện tại", state_desc[curr_st])

    st.divider()

    # --- BIỂU ĐỒ TỔNG HỢP ---
    plt.style.use('dark_background') 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [2, 1, 1.5]})
    fig.patch.set_facecolor('#0E1117') 

    # Tầng 1: HMM State Analysis
    colors_hmm = ['#00FF00', '#FFFF00', '#FF0000'] 
    ax1_vni = ax1.twinx()
    ax1_vni.plot(df.index, df['close_vni'], color='orange', alpha=0.2, linestyle='--', label='VNINDEX')
    
    ax1.plot(df.index, df['close'], color='white', alpha=0.3, lw=1)
    for i in range(3):
        st_data = df[df['state'] == i]
        ax1.scatter(st_data.index, st_data['close'], c=colors_hmm[i], s=30, label=state_desc[i], edgecolors='black', linewidth=0.5)
    
    ax1.set_title(f"Tương quan giữa {TICKER} và VNINDEX", fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left')

    # Tầng 2: Volume
    colors_vol = np.where(df['ret_stock'] >= 0, '#26a69a', '#ef5350')
    ax2.bar(df.index, df['volume'], color=colors_vol, alpha=0.8)
    ax2.set_title("Khối lượng giao dịch thực tế", fontsize=12)

    # Tầng 3: Monte Carlo Probability Density
    kde = gaussian_kde(final_prices)
    x_range = np.linspace(min(final_prices), max(final_prices), 1000)
    y_kde = kde(x_range)
    
    ax3.plot(x_range, y_kde, color="#00CCFF", lw=3, label='Mật độ xác suất')
    ax3.fill_between(x_range, y_kde, where=(x_range >= S0), color='#00FF00', alpha=0.2)
    ax3.fill_between(x_range, y_kde, where=(x_range < S0), color='#FF0000', alpha=0.2)
    
    # Kẻ các đường chỉ số kỳ vọng
    ax3.axvline(S0, color='white', linestyle='--', alpha=0.6, label='Giá hiện tại')
    ax3.axvline(expected_price, color='#FFFF00', linestyle='-', lw=2, label=f'Giá kỳ vọng: {expected_price:,.0f}')
    ax3.axvline(p75, color='#00FF00', linestyle=':', alpha=0.8, label=f'Mục tiêu P75: {p75:,.0f}')
    
    ax3.set_title(f"Phân phối xác suất giá sau {DAYS_TO_PREDICT} ngày", fontsize=14)
    ax3.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # --- BẢNG THỐNG KÊ CHI TIẾT ---
    st.subheader("📌 Chi tiết kịch bản dự báo")
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.write("**Các mốc giá quan trọng:**")
        st.table(pd.DataFrame({
            "Kịch bản": ["Thận trọng (P25)", "Trung vị (P50)", "Kỳ vọng (Mean)", "Lạc quan (P75)"],
            "Giá dự báo": [f"{p25:,.0f} đ", f"{p50:,.0f} đ", f"{expected_price:,.0f} đ", f"{p75:,.0f} đ"],
            "Lợi nhuận": [f"{(p25-S0)/S0:+.1%}", f"{(p50-S0)/S0:+.1%}", f"{expected_return:+.1%}", f"{(p75-S0)/S0:+.1%}"]
        }))

    with col_t2:
        st.write("**Ma trận xác suất chuyển trạng thái:**")
        fig_h, ax_h = plt.subplots(figsize=(10, 5))
        fig_h.patch.set_facecolor('#0E1117')
        sns.heatmap(model.transmat_, annot=True, fmt=".2f", cmap='viridis',
                    xticklabels=[state_desc[i] for i in range(3)], 
                    yticklabels=[state_desc[i] for i in range(3)], ax=ax_h)
        st.pyplot(fig_h)

else:
    st.error(f"⚠️ Không thể tải dữ liệu cho mã {TICKER}. Vui lòng kiểm tra lại kết nối hoặc mã cổ phiếu.")
