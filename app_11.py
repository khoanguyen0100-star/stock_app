# =========================
# FULL POSITION-AWARE UPGRADE
# COPY PASTE NGUYÊN FILE
# =========================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
    value=1000
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
# POSITION INPUT
# =========================
st.sidebar.subheader("Vị thế hiện tại")

ENTRY_PRICE = st.sidebar.number_input(
    "Giá vốn trung bình (VNĐ)",
    value=0.0,
    step=1000.0
)

POSITION_SIZE = st.sidebar.number_input(
    "Số lượng cổ phiếu đang nắm giữ",
    value=0,
    step=100
)

# =========================
# AI ANALYSIS
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
    kurt_val,
    entry_price,
    position_size,
    unrealized_pnl,
    unrealized_return,
    drawdown_from_entry
):

    try:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            return "⚠️ Chưa cấu hình GROQ_API_KEY."

        client = Groq(api_key=api_key)

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

====================
MARKET DATA
====================

Ticker: {ticker}

Current Price: {current_price:,.0f}
Expected Price: {expected_price:,.0f}
Expected Return: {expected_return:.2f}%
Win Rate: {win_rate:.2f}%

====================
POSITION DATA
====================

Entry Price: {entry_price:,.0f}
Position Size: {position_size:,}
Unrealized PnL: {unrealized_pnl:,.0f}
Position Return: {unrealized_return:.2f}%
Potential Drawdown From Entry: {drawdown_from_entry:.2f}%

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

Đánh giá theo POSITION MANAGEMENT:
- add
- hold
- reduce
- exit

Ưu tiên:
- bảo vệ lợi nhuận hiện có
- downside asymmetry
- regime deterioration
- tail risk
- convexity risk

Không mô tả dữ liệu.
Không viết kiểu retail.
Không giải thích cơ bản.

Output:
- 3 đoạn ngắn
- 180-220 từ
- Institutional tone
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
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Lỗi AI: {str(e)}"

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
            df['close'] *= 1000

        if df_vni['close'].mean() < 100:
            df_vni['close'] *= 1000

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
    n_iter=200,
    tol=0.01,
    random_state=42
    )
    
    X = np.nan_to_num(X)
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
        0: "Tích lũy",
        1: "Xu hướng tăng",
        2: "Rủi ro"
    }

    curr_st = df['state'].iloc[-1]

    # =========================
    # REMAP TRANSITION MATRIX
    # =========================
    remapped_transmat = np.zeros_like(model.transmat_)

    for old_i, new_i in new_labels.items():
        for old_j, new_j in new_labels.items():

            remapped_transmat[new_i][new_j] = (
                model.transmat_[old_i][old_j]
            )

    transition_prob = remapped_transmat[curr_st][curr_st]

    S0 = df['close'].iloc[-1]

    # =========================
    # POSITION ANALYTICS
    # =========================
    if ENTRY_PRICE > 0 and POSITION_SIZE > 0:

        unrealized_pnl = (
            (S0 - ENTRY_PRICE)
            * POSITION_SIZE
        )

        unrealized_return = (
            (S0 - ENTRY_PRICE)
            /
            ENTRY_PRICE
            * 100
        )

        position_value = (
            S0 * POSITION_SIZE
        )

    else:

        unrealized_pnl = 0
        unrealized_return = 0
        position_value = 0

    st.subheader(f"📊 Dữ liệu thực tế: {TICKER}")

    # =========================
    # POSITION STATUS UI
    # =========================
    if POSITION_SIZE > 0:

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "PnL chưa thực hiện",
            f"{unrealized_pnl:,.0f} đ",
            delta=f"{unrealized_return:.2f}%"
        )

        p2.metric(
            "Giá vốn",
            f"{ENTRY_PRICE:,.0f} đ"
        )

        p3.metric(
            "Quy mô vị thế",
            f"{position_value:,.0f} đ"
        )
