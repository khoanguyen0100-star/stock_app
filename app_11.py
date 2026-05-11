import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PHẦN CODE BIỂU ĐỒ ĐƯỢC VIẾT LẠI (CÁC PHẦN KHÁC GIỮ NGUYÊN) ---

if df is not None:
    # ... (Giữ nguyên toàn bộ phần tính toán HMM, Monte Carlo bên trên) ...

    st.divider()
    
    # 1. BIỂU ĐỒ GIÁ & VOLUME TƯƠNG TÁC (GỘP CHUNG)
    # Tạo subplot: dòng 1 là giá (chiếm 80%), dòng 2 là volume (chiếm 20%)
    fig_main = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.8, 0.2]
    )

    # Màu sắc trạng thái HMM
    hmm_colors = {0: '#FFFF00', 1: '#00FF00', 2: '#FF0000'}

    # Vẽ đường giá đóng cửa
    fig_main.add_trace(go.Scatter(
        x=df.index, y=df['close'],
        mode='lines', name='Giá đóng cửa',
        line=dict(color='white', width=1), opacity=0.3
    ), row=1, col=1)

    # Vẽ các điểm trạng thái HMM
    for i in range(3):
        st_data = df[df['state'] == i]
        fig_main.add_trace(go.Scatter(
            x=st_data.index, y=st_data['close'],
            mode='markers', name=state_desc[i],
            marker=dict(size=6, color=hmm_colors[i])
        ), row=1, col=1)

    # Thêm VNINDEX làm tham chiếu (Trục phụ ẩn hoặc mờ)
    fig_main.add_trace(go.Scatter(
        x=df.index, y=df['close_vni'],
        name='VNINDEX', line=dict(color='gray', dash='dot', width=1),
        opacity=0.2
    ), row=1, col=1)

    # Vẽ Khối lượng (Volume) ở tầng dưới
    vol_colors = ['#26a69a' if r >= 0 else '#ef5350' for r in df['ret_stock']]
    fig_main.add_trace(go.Bar(
        x=df.index, y=df['volume'],
        name='Khối lượng', marker_color=vol_colors, opacity=0.8
    ), row=2, col=1)

    # Cấu hình giao diện chuẩn Dark Mode, cho phép Zoom/Pan
    fig_main.update_layout(
        height=700,
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        title=f"Phân tích tương quan & Trạng thái HMM: {TICKER}",
        xaxis_rangeslider_visible=False, # Tắt slider để zoom mượt hơn bằng chuột
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_main, use_container_width=True)

    # 2. BIỂU ĐỒ PHÂN PHỐI XÁC SUẤT MONTE CARLO (TƯƠNG TÁC)
    fig_kde = go.Figure()
    
    # Lấy dữ liệu KDE đã tính toán
    kde_x = np.linspace(min(final_prices), max(final_prices), 1000)
    kde_y = kde(kde_x)

    # Vẽ đường cong KDE
    fig_kde.add_trace(go.Scatter(
        x=kde_x, y=kde_y,
        mode='lines', name='Mật độ xác suất',
        line=dict(color='#00CCFF', width=3),
        fill='tozeroy', fillcolor='rgba(0, 204, 255, 0.1)'
    ))

    # Tô màu vùng có lãi (Xanh)
    mask_gain = kde_x >= S0
    fig_kde.add_trace(go.Scatter(
        x=kde_x[mask_gain], y=kde_y[mask_gain],
        fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.3)',
        mode='none', name='Vùng có lãi'
    ))

    # Đường giá hiện tại & Kỳ vọng
    fig_kde.add_vline(x=S0, line_dash="dash", line_color="white", annotation_text="Giá HT")
    fig_kde.add_vline(x=expected_price, line_color="#FFFF00", line_width=2, 
                      annotation_text=f"Kỳ vọng: {expected_price:,.0f}")

    fig_kde.update_layout(
        height=400,
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        title=f"Phân phối xác suất dự báo sau {DAYS_TO_PREDICT} ngày",
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig_kde, use_container_width=True)

    # --- BẢNG DỮ LIỆU & MA TRẬN HMM (GIỮ NGUYÊN) ---
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
        # Vẫn giữ Seaborn heatmap theo ý ông
        sns.heatmap(model.transmat_, annot=True, fmt=".2f", cmap='viridis', 
                    xticklabels=["S0","S1","S2"], yticklabels=["S0","S1","S2"], ax=ax_h, cbar=False)
        st.pyplot(fig_h)

    # ... (Toàn bộ phần Backtest giữ nguyên bên dưới) ...
