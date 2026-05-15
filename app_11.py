# FULL UPGRADED CODE (COPY-PASTE)


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hmmlearn.hmm import GaussianHMM
from vnstock.ui import Market
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from scipy.stats import skew, kurtosis
from arch import arch_model
from groq import Groq

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Hệ thống Giao dịch Quant Pro",
    layout="wide"
)

mkt = Market()

st.title("📊 Hệ thống Phân tích Định lượng HMM & Monte Carlo")
st.sidebar.header("Cấu hình thông số")

# =========================
# SIDEBAR
# =========================
TICKER = st.sidebar.text_input(
    "Nhập mã cổ phiếu",
    value="FPT"
).upper()

YEARS_DATA = st.sidebar.slider(
    "Số năm dữ liệu lịch sử",
    1,
    5,
    2
)

DAYS_TO_PREDICT = st.sidebar.number_input(
    "Số ngày dự báo",
    value=60
)

N_SIM = st.sidebar.select_slider(
    "Số lượng mô phỏng (N)",
    options=[1000, 5000, 10000],
    value=10000
)

st.sidebar.subheader("Quản trị rủi ro")

CAPITAL = st.sidebar.number_input(
    "Vốn đầu tư (VNĐ)",
    value=100000000,
    step=10000000
)

RISK_PER_TRADE = st.sidebar.slider(
    "Rủi ro mỗi lệnh (%)",
    0.5,
    5.0,
    2.0
) / 100

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
- KHÔNG mô tả lại dữ liệu
- KHÔNG lặp lại metrics
- KHÔNG viết kiểu financial blogger
- Chỉ phân tích ý nghĩa đầu tư

====================
MARKET DATA
====================

Ticker: {ticker}

Current Price: {current_price:,.0f}
Expected Price: {expected_price:,.0f}
Expected Return: {expected_return:.2f}%
Win Rate: {win_rate:.2f}%

====================
RISK
====================

Reward/Risk: {reward_risk:.2f}
Sharpe Ratio: {sharpe_ratio:.2f}
Max Drawdown: {max_drawdown:.2f}%
VaR95: {var_95:.2f}%
Beta: {beta:.2f}
Skewness: {skew_val:.2f}
Kurtosis: {kurt_val:.2f}

====================
REGIME
====================

Current Regime: {current_state}
Relative Strength: {rs_status}
Regime Persistence Probability: {transition_prob:.2%}

====================
MONTE CARLO
====================

P25: {p25:,.0f}
P50: {p50:,.0f}
P75: {p75:,.0f}

====================
PORTFOLIO ENGINE
====================

Signal Score: {signal_score}
Portfolio Bias: {portfolio_bias}

====================
RULES
====================

Ưu tiên phân tích:
1. Regime HMM
2. Distribution Monte Carlo
3. Tail-risk
4. Reward/Risk asymmetry
5. Relative Strength

Nếu tín hiệu xung đột:
- giải thích signal nào đáng tin hơn
- signal nào lagging

Phong cách:
- giống internal investment memo
- lạnh
- súc tích
- institutional
- không marketing
- không giải thích cơ bản

CẤM dùng:
- "nhà đầu tư nên"
- "theo dõi sát sao"
- "có thể xem xét"
- "triển vọng tích cực"
- "trong ngắn hạn"
- "cổ phiếu tiềm năng"

Output:
- 3 đoạn ngắn
- 180-220 từ
- Không bullet
- Không markdown
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.15,
            max_tokens=500,
            top_p=0.9
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Lỗi AI Groq: {str(e)}"

# =========================
# LOAD DATA
# =========================
@st.cache_data(ttl=3600)
def load_data(ticker, years):

    today = datetime.now()

    start_date = (
        today - timedelta(days=365 * years)
    ).strftime('%Y-%m-%d')

    end_date = today.strftime('%Y-%m-%d')

    try:

        df = mkt.equity(ticker).ohlcv(
            start=start_date,
            end=end_date
        )

        df_vni = mkt.equity("VNINDEX").ohlcv(
            start=start_date,
            end=end_date
        )

        if df.empty or df_vni.empty:
            return None, None

        if df['close'].iloc[-1] < 1000:
            df['close'] = df['close'] * 1000

        if df_vni['close'].mean() < 100:
            df_vni['close'] = df_vni['close'] * 1000

        df_combined = pd.merge(
            df[['close', 'volume']],
            df_vni[['close']],
            left_index=True,
            right_index=True,
            suffixes=('', '_vni')
        )

        df_combined['ret_stock'] = np.log(
            df_combined['close']
            /
            df_combined['close'].shift(1)
        )

        df_combined['ret_vni'] = np.log(
            df_combined['close_vni']
            /
            df_combined['close_vni'].shift(1)
        )

        window = 20

        df_combined['rs_line'] = (
            (
                df_combined['close']
                /
                df_combined['close'].shift(window)
            )
            /
            (
                df_combined['close_vni']
                /
                df_combined['close_vni'].shift(window)
            )
        )

        returns = (
            df_combined['ret_stock']
            .dropna()
            * 100
        )

        garch_m = arch_model(
            returns,
            vol='Garch',
            p=1,
            q=1,
            dist='normal'
        )

        res_garch = garch_m.fit(disp='off')

        df_combined['volatility'] = (
            res_garch.conditional_volatility / 100
        )

        return df_combined.dropna(), df_vni

    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None, None

# =========================
# MAIN
# =========================
df, df_vni = load_data(TICKER, YEARS_DATA)

if df is not None:

    beta_val = (
        df['ret_stock'].cov(df['ret_vni'])
        /
        df['ret_vni'].var()
    )

    X = df[['ret_stock', 'volatility']].values

    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=1000,
        random_state=42
    )

    model.fit(X)

    means = model.means_[:, 0]

    order = np.argsort(means)

    new_labels = {
        order[0]: 2,
        order[1]: 0,
        order[2]: 1
    }

    raw_states = model.predict(X)

    df['state'] = [new_labels[s] for s in raw_states]

    state_desc = {
        0: "Tích lũy (Đi ngang)",
        1: "Xu hướng (Tăng mạnh)",
        2: "Rủi ro (Biến động xấu)"
    }

    curr_st = df['state'].iloc[-1]

    # =========================
    # REGIME PERSISTENCE
    # =========================
    transition_prob = model.transmat_[curr_st][curr_st]

    # =========================
    # SIGNAL ENGINE
    # =========================
    signal_score = 0

    if curr_st == 1:
        signal_score += 2

    elif curr_st == 0:
        signal_score += 0

    else:
        signal_score -= 2

    S0 = df['close'].iloc[-1]

    st.subheader(f"📊 Dữ liệu thực tế: {TICKER}")

    col_price, col_state = st.columns(2)

    with col_price:

        st.metric(
            "Giá hiện tại",
            f"{S0:,.0f} đ"
        )

    with col_state:

        if curr_st == 1:

            st.success(
                f"Trạng thái hiện tại: {state_desc[curr_st]}"
            )

        elif curr_st == 0:

            st.warning(
                f"Trạng thái hiện tại: {state_desc[curr_st]}"
            )

        else:

            st.error(
                f"Trạng thái hiện tại: {state_desc[curr_st]}"
            )

    # =========================
    # MONTE CARLO
    # =========================
    state_info = df[df['state'] == curr_st]

    mu = state_info['ret_stock'].mean()

    sigma = state_info['ret_stock'].std()

    daily_returns = np.exp(
        (
            mu - 0.5 * sigma**2
        )
        +
        sigma * np.random.standard_normal(
            (DAYS_TO_PREDICT, N_SIM)
        )
    )

    price_paths = np.zeros(
        (DAYS_TO_PREDICT + 1, N_SIM)
    )

    price_paths[0] = S0

    for t in range(1, DAYS_TO_PREDICT + 1):

        price_paths[t] = (
            price_paths[t - 1]
            *
            daily_returns[t - 1]
        )

    final_prices = price_paths[-1, :]

    expected_price = np.mean(final_prices)

    expected_return = (
        (
            expected_price - S0
        )
        /
        S0
        * 100
    )

    win_rate_val = (
        np.mean(final_prices > S0)
        * 100
    )

    p25, p50, p75 = np.percentile(
        final_prices,
        [25, 50, 75]
    )

    sharpe_ratio = (
        (mu / sigma) * np.sqrt(252)
        if sigma != 0 else 0
    )

    var_95 = np.percentile(
        state_info['ret_stock'],
        5
    ) * 100

    skew_val = skew(
        state_info['ret_stock']
    )

    kurt_val = kurtosis(
        state_info['ret_stock']
    )

    st.divider()

    r_col1, r_col2 = st.columns(2)

    with r_col1:

        st.subheader("🛡️ Kế hoạch giao dịch")

        stop_loss = p25 * 0.98

        risk_amt = CAPITAL * RISK_PER_TRADE

        dist_to_sl = S0 - stop_loss

        if dist_to_sl > 0:

            shares_to_buy = int(
                risk_amt / dist_to_sl
            )

            total_cost = shares_to_buy * S0

        else:

            shares_to_buy = 0
            total_cost = 0

        reward_risk = (
            (
                expected_price - S0
            )
            /
            (
                S0 - stop_loss
            )
            if dist_to_sl > 0 else 0
        )

        # =========================
        # SIGNAL ENGINE CONTINUED
        # =========================
        if reward_risk > 2:
            signal_score += 1

        elif reward_risk < 1:
            signal_score -= 1

        if sharpe_ratio > 1:
            signal_score += 1

        elif sharpe_ratio < 0:
            signal_score -= 1

        if win_rate_val > 60:
            signal_score += 1

        elif win_rate_val < 40:
            signal_score -= 1

        if beta_val > 1.5:
            signal_score -= 1

        if var_95 < -10:
            signal_score -= 1

        confidence_score = abs(signal_score) / 6 * 100
        confidence_score = min(confidence_score, 100)

        st.write(
            f"- **Điểm dừng lỗ tối ưu (SL):** {stop_loss:,.0f} đ"
        )

        st.write(
            f"- **Số lượng cổ phiếu nên mua:** {shares_to_buy:,} CP"
        )

        st.write(
            f"- **Tổng giá trị giải ngân:** {total_cost:,.0f} đ"
        )

        st.write(
            f"- **Tỷ lệ Reward/Risk:** {reward_risk:.2f}"
        )

    with r_col2:

        st.subheader("📊 Chỉ số hiệu suất")

        st.metric(
            "Xác suất tăng giá",
            f"{win_rate_val:.1f}%"
        )

        st.metric(
            "Hệ số Beta (vs VNINDEX)",
            f"{beta_val:.2f}"
        )

        rs_status = (
            "Khỏe hơn thị trường"
            if df['rs_line'].iloc[-1] > 1
            else "Yếu hơn thị trường"
        )

        st.write(
            f"- **Sức mạnh tương quan (RS):** {rs_status}"
        )

        st.write(
            f"- **Sharpe Ratio:** {sharpe_ratio:.2f}"
        )

        st.write(
            f"- **VaR 95%:** {var_95:.2f}%"
        )

        st.write(
            f"- **Skewness:** {skew_val:.2f}"
        )

        st.write(
            f"- **Kurtosis:** {kurt_val:.2f}"
        )

    st.divider()

    # =========================
    # PORTFOLIO BIAS
    # =========================
    if signal_score >= 3:

        st.success(
            f"Portfolio Bias: STRONG LONG | Signal Score = {signal_score}"
        )

    elif signal_score >= 1:

        st.info(
            f"Portfolio Bias: TACTICAL LONG | Signal Score = {signal_score}"
        )

    elif signal_score <= -3:

        st.error(
            f"Portfolio Bias: RISK OFF | Signal Score = {signal_score}"
        )

    elif signal_score <= -1:

        st.warning(
            f"Portfolio Bias: DEFENSIVE | Signal Score = {signal_score}"
        )

    else:

        st.warning(
            f"Portfolio Bias: NEUTRAL | Signal Score = {signal_score}"
        )

    st.metric(
        "Model Confidence",
        f"{confidence_score:.0f}%"
    )

    # =========================
    # CHARTS
    # =========================
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14, 12),
        gridspec_kw={
            'height_ratios': [2, 0.8, 1.2]
        }
    )

    fig.patch.set_facecolor('#0E1117')

    # =========================
    # HMM CHART
    # =========================
    ax1_vni = ax1.twinx()

    ax1_vni.plot(
        df.index,
        df['close_vni'],
        color='white',
        alpha=0.1,
        linestyle='--',
        label='VNINDEX'
    )

    ax1_vni.set_ylabel(
        "VNINDEX",
        color='white',
        alpha=0.3
    )

    ax1_vni.tick_params(
        axis='y',
        labelcolor='yellow',
        labelsize=8
    )

    ax1.plot(
        df.index,
        df['close'],
        color='white',
        alpha=0.3
    )

    colors_hmm = [
        '#FFFF00',
        '#00FF00',
        '#FF0000'
    ]

    for i in range(3):

        st_data = df[df['state'] == i]

        ax1.scatter(
            st_data.index,
            st_data['close'],
            c=colors_hmm[i],
            s=25,
            label=state_desc[i]
        )

    ax1.set_title(
        f"Phân tích tương quan: {TICKER} giữa VNINDEX",
        fontsize=12,
        color='white'
    )

    ax1.legend(
        loc='upper left',
        fontsize=9
    )

    ax1.set_facecolor('#0E1117')

    ax1.tick_params(colors='white')

    # =========================
    # VOLUME
    # =========================
    colors_vol = np.where(
        df['ret_stock'] >= 0,
        '#26a69a',
        '#ef5350'
    )

    ax2.bar(
        df.index,
        df['volume'],
        color=colors_vol,
        alpha=0.7
    )

    ax2.set_title(
        "Khối lượng giao dịch",
        fontsize=10,
        color='white'
    )

    ax2.set_facecolor('#0E1117')

    ax2.tick_params(colors='white')

    # =========================
    # KDE
    # =========================
    kde = gaussian_kde(final_prices)

    x_range = np.linspace(
        min(final_prices),
        max(final_prices),
        1000
    )

    ax3.plot(
        x_range,
        kde(x_range),
        color="#00CCFF",
        lw=2
    )

    ax3.fill_between(
        x_range,
        kde(x_range),
        where=(x_range >= S0),
        color='#00FF00',
        alpha=0.2
    )

    ax3.axvline(
        S0,
        color='white',
        linestyle='--',
        label='Giá hiện tại'
    )

    ax3.axvline(
        expected_price,
        color='#FFFF00',
        label=f'Kỳ vọng: {expected_price:,.0f}'
    )

    ax3.set_title(
        f"Phân phối xác suất dự báo sau {DAYS_TO_PREDICT} ngày",
        fontsize=12,
        color='white'
    )

    ax3.legend()

    ax3.set_facecolor('#0E1117')

    ax3.tick_params(colors='white')

    plt.tight_layout(pad=3.0)

    st.pyplot(fig)

    # =========================
    # TABLE & HEATMAP
    # =========================
    st.divider()

    col_t1, col_t2 = st.columns(2)

    with col_t1:

        st.write(
            "**Kịch bản dự báo theo percentiles:**"
        )

        st.table(
            pd.DataFrame({
                "Kịch bản": [
                    "Thận trọng (P25)",
                    "Trung vị (P50)",
                    "Kỳ vọng",
                    "Lạc quan (P75)"
                ],

                "Giá dự báo": [
                    f"{p25:,.0f} đ",
                    f"{p50:,.0f} đ",
                    f"{expected_price:,.0f} đ",
                    f"{p75:,.0f} đ"
                ],

                "Lợi nhuận": [
                    f"{(p25-S0)/S0:+.1%}",
                    f"{(p50-S0)/S0:+.1%}",
                    f"{expected_return/100:+.1%}",
                    f"{(p75-S0)/S0:+.1%}"
                ]
            })
        )

    with col_t2:

        st.write(
            "**Ma trận chuyển trạng thái (HMM Transition):**"
        )

        fig_h, ax_h = plt.subplots(
            figsize=(4, 3)
        )

        sns.heatmap(
            model.transmat_,
            annot=True,
            fmt=".2f",
            cmap='viridis',
            xticklabels=["S0", "S1", "S2"],
            yticklabels=["S0", "S1", "S2"],
            ax=ax_h,
            cbar=False
        )

        st.pyplot(fig_h)

    # =========================
    # BACKTEST
    # =========================
    st.divider()

    st.subheader(
        "📈 Kiểm định hiệu quả chiến lược (Backtest)"
    )

    df['strategy_ret'] = np.where(
        df['state'].shift(1) == 1,
        df['ret_stock'],
        0
    )

    df['cum_market'] = np.exp(
        df['ret_stock'].cumsum()
    )

    df['cum_strategy'] = np.exp(
        df['strategy_ret'].cumsum()
    )

    total_ret = (
        df['cum_strategy'].iloc[-1] - 1
    ) * 100

    mkt_ret = (
        df['cum_market'].iloc[-1] - 1
    ) * 100

    max_dd = (
        (
            df['cum_strategy']
            /
            df['cum_strategy'].cummax()
        ) - 1
    ).min() * 100

    diff = total_ret - mkt_ret

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "Lợi nhuận HMM",
        f"{total_ret:.1f}%",
        delta=f"{diff:+.1f}% vs Market"
    )

    b2.metric(
        "Lợi nhuận Mua & Giữ",
        f"{mkt_ret:.1f}%"
    )

    b3.metric(
        "Sụt giảm tối đa (MDD)",
        f"{max_dd:.1f}%"
    )

    fig_bt, ax_bt = plt.subplots(
        figsize=(14, 4)
    )

    fig_bt.patch.set_facecolor('#0E1117')

    ax_bt.plot(
        df.index,
        df['cum_strategy'],
        label='Chiến lược HMM',
        color='#00FF00',
        lw=2
    )

    ax_bt.plot(
        df.index,
        df['cum_market'],
        label='Mua & Giữ',
        color='white',
        alpha=0.3
    )

    ax_bt.set_facecolor('#0E1117')

    ax_bt.tick_params(colors='white')

    ax_bt.legend()

    st.pyplot(fig_bt)

    # =========================
    # AI ANALYSIS
    # =========================
    st.divider()

    st.subheader("🧠 Nhận định AI từ Groq")

    risk_free_rate = 0.03 / 252

    strategy_returns = df['strategy_ret'].dropna() if 'strategy_ret' in df.columns else pd.Series([0])

    if strategy_returns.std() != 0:

        sharpe_ratio = (
            (
                strategy_returns.mean()
                - risk_free_rate
            )
            /
            strategy_returns.std()
        ) * np.sqrt(252)

    else:

        sharpe_ratio = 0

    var_price_95 = np.percentile(
        final_prices,
        5
    )

    var_95 = (
        (
            var_price_95 - S0
        )
        /
        S0
    ) * 100

    ai_analysis = generate_ai_analysis(
        ticker=TICKER,
        current_price=S0,
        expected_price=expected_price,
        expected_return=expected_return,
        win_rate=win_rate_val,
        beta=beta_val,
        current_state=state_desc[curr_st],
        rs_status=rs_status,
        reward_risk=reward_risk,
        p25=p25,
        p50=p50,
        p75=p75,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=-15,
        var_95=var_95,
        signal_score=signal_score,
        transition_prob=transition_prob,
        skew_val=skew_val,
        kurt_val=kurt_val
    )

    st.info(ai_analysis)

