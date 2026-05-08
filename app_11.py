# --- KHỐI BACKTEST ---
    st.divider()
    st.subheader("📈 Kiểm định hiệu quả chiến lược (Backtest)")
    
    # Tính toán log return tích lũy
    df['strategy_ret'] = np.where(df['state'].shift(1) == 1, df['ret_stock'], 0)
    df['cum_market'] = np.exp(df['ret_stock'].cumsum())
    df['cum_strategy'] = np.exp(df['strategy_ret'].cumsum())
    
    # Chuyển sang phần trăm lợi nhuận
    total_ret = (df['cum_strategy'].iloc[-1] - 1) * 100
    mkt_ret = (df['cum_market'].iloc[-1] - 1) * 100
    max_dd = (df['cum_strategy'] / df['cum_strategy'].cummax() - 1).min() * 100

    # Hiển thị Metrics
    b1, b2, b3 = st.columns(3)
    
    # SỬA LỖI HIỂN THỊ TẠI ĐÂY
    diff = total_ret - mkt_ret
    b1.metric("Lợi nhuận HMM", f"{total_ret:.1f}%", delta=f"{diff:+.1f}% vs Market")
    b2.metric("Lợi nhuận Mua & Giữ", f"{mkt_ret:.1f}%")
    b3.metric("Sụt giảm tối đa (MDD)", f"{max_dd:.1f}%", delta_color="inverse")

    # Vẽ biểu đồ Backtest
    fig_bt, ax_bt = plt.subplots(figsize=(14, 4))
    fig_bt.patch.set_facecolor('#0E1117')
    ax_bt.plot(df.index, df['cum_strategy'], label='Chiến lược HMM', color='#00FF00', lw=2)
    ax_bt.plot(df.index, df['cum_market'], label='Mua & Giữ', color='white', alpha=0.3)
    ax_bt.set_facecolor('#0E1117')
    ax_bt.tick_params(colors='white')
    ax_bt.legend()
    st.pyplot(fig_bt)
