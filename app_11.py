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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    state_desc = {0: "Tích lũy (Đi ngang)", 1: "Xu hướng (Tăng mạnh)", 2: "Rủi ro (Biến động xấu)"}
    curr_st = df['state'].iloc[-1]
    S0 = df['close'].iloc[-1]

    # --- KHỐI HIỂN THỊ GIÁ HIỆN TẠI ---
    st.subheader(f"📊 Dữ liệu thực tế: {TICKER}")
    col_price, col_state = st.columns(2)
    with col_price:
        st.metric("Giá hiện tại", f"{S0:,.0f} đ") 
    
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

    # --- BIỂU ĐỒ 1: GIÁ & VOLUME GỘP CHUNG (PLOTLY TƯƠNG TÁC) ---
    fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.03, subplot_titles=(f'Phân tích HMM: {TICKER}', 'Khối lượng'), 
                              row_width=[0.3, 0.7])

    # Vẽ đường giá mờ
    fig_main.add_trace(go.Scatter(x=df.index, y=df['close'], line=dict(color='white', width=1), opacity=0.3, name='Giá đóng cửa'), row=1, col=1)

    # Vẽ các điểm trạng thái HMM
    colors_hmm = {0: '#FFFF00', 1: '#00FF00', 2: '#FF0000'}
    for i in range(3):
        st_data = df[df['state'] == i]
        fig_main.add_trace(go.Scatter(x=st_data.index, y=st_data['close'], mode='markers', 
                                      marker=dict(size=6, color=colors_hmm[i]), name=state_desc[i]), row=1, col=1)

    # Vẽ Volume tầng dưới
    vol_colors = ['#26a69a' if r >= 0 else '#ef5350' for r in df['ret_stock']]
    fig_main.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=vol_colors, name='Volume', opacity=0.7), row=2, col=1)

    fig_main.update_layout(height=600, template='plotly_dark', paper_bgcolor='#0E1117', plot_bgcolor='#0E1117', 
                          xaxis_rangeslider_visible=False, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_main, use_container_width=True)

    # --- BIỂU ĐỒ 2: DỰ BÁO MONTE CARLO (PLOTLY TƯƠNG TÁC) ---
    kde = gaussian_kde(final_prices)
    x_range = np.linspace(min(final_prices), max(final_prices), 1000)
    y_range = kde(x_range)

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(x=x_range, y=y_range, fill='tozeroy', line_color='#00CCFF', name='Mật độ xác suất'))
    
    # Tô màu xanh vùng có lãi
    mask = x_range >= S0
    fig_mc.add_trace(go.Scatter(x=x_range[mask], y=y_range[mask], fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.2)', line_color='rgba(0,0,0,0)', name='Vùng lợi nhuận'))

    fig_mc.add_vline(x=S0, line_dash="dash", line_color="white", annotation_text="Giá hiện tại")
    fig_mc.add_vline(x=expected_price, line_color="#FFFF00", annotation_text=f"Kỳ vọng: {expected_price:,.0f}")
    
    fig_mc.update_layout(height=400, template='plotly_dark', paper_bgcolor='#0E1117', plot_bgcolor='#0E1117', 
                         title=f"Phân phối xác suất dự báo sau {DAYS_TO_PREDICT} ngày", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_mc, use_container_width=True)

    # --- BẢNG DỮ LIỆU & HEATMAP (GIỮ NGUYÊN SEABORN CHO MA TRẬN) ---
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
        fig_h.patch.set_facecolor('#0E1117')
        sns.heatmap(model.transmat_, annot=True, fmt=".2f", cmap='viridis', 
                    xticklabels=["S0","S1","S2"], yticklabels=["S0","S1","S2"], ax=ax_h, cbar=False, annot_kws={"color": "white"})
        ax_h.tick_params(colors='white')
        st.pyplot(fig_h)

    # --- KHỐI BACKTEST (GIỮ NGUYÊN) ---
    st.divider()
    st.subheader("📈 Kiểm định hiệu quả chiến lược (Backtest)")
    df['strategy_ret'] = np.where(df['state'].shift(1) == 1, df['ret_stock'], 0)
    df['cum_market'] = np.exp(df['ret_stock'].cumsum())
    df['cum_strategy'] = np.exp(df['strategy_ret'].cumsum())
    
    total_ret = (df['cum_strategy'].iloc[-1] - 1) * 100
    mkt_ret = (df['cum_market'].iloc[-1] - 1) * 100
    max_dd = (df['cum_strategy'] / df['cum_strategy'].cummax() - 1).min() * 100
    diff = total_ret - mkt_ret

    b1, b2, b3 = st.columns(3)
    b1.metric("Lợi nhuận HMM", f"{total_ret:.1f}%", delta=f"{diff:+.1f}% vs Market")
    b2.metric("Lợi nhuận Mua & Giữ", f"{mkt_ret:.1f}%")
    b3.metric("Sụt giảm tối đa (MDD)", f"{max_dd:.1f}%")

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=df.index, y=df['cum_strategy'], name='Chiến lược HMM', line=dict(color='#00FF00', width=2)))
    fig_bt.add_trace(go.Scatter(x=df.index, y=df['cum_market'], name='Mua & Giữ', line=dict(color='white', width=1), opacity=0.3))
    fig_bt.update_layout(height=400, template='plotly_dark', paper_bgcolor='#0E1117', plot_bgcolor='#0E1117', margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_bt, use_container_width=True)

else:
    st.error("⚠️ Không thể tải dữ liệu. Vui lòng kiểm tra lại mã cổ phiếu.")
