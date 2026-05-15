# =========================
# GROQ AI ANALYSIS - UPGRADED
# =========================
def generate_ai_analysis(
    ticker,
    current_price,
    expected_price,
    expected_return,
    win_rate,
    beta,
    current_state,
    rs_status,
    reward_risk,
    p25,
    p50,
    p75,
    sharpe_ratio,
    max_drawdown,
    var_95,
    signal_score,
    transition_prob,
    skew_val,
    kurt_val
):

    try:

        api_key = st.secrets.get("GROQ_API_KEY", None)

        if not api_key:
            return "⚠️ Chưa cấu hình GROQ_API_KEY trong Streamlit Secrets."

        client = Groq(api_key=api_key)

        # =========================
        # PORTFOLIO BIAS
        # =========================
        if signal_score >= 3:
            portfolio_bias = "STRONG LONG"

        elif signal_score >= 1:
            portfolio_bias = "TACTICAL LONG"

        elif signal_score <= -3:
            portfolio_bias = "RISK OFF"

        elif signal_score <= -1:
            portfolio_bias = "DEFENSIVE"

        else:
            portfolio_bias = "NEUTRAL"

        prompt = f"""
Bạn là CIO của một quỹ hedge fund định lượng long/short equity.

Nhiệm vụ:
        return f"⚠️ Lỗi AI Groq: {str(e)}"
