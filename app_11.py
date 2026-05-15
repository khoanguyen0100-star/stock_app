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
# INIT
# =========================
mkt = Market()
np.random.seed(42)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hệ thống Giao dịch Quant Pro",
    layout="wide"
)

st.title("📊 Hệ thống Phân tích Định lượng HMM & Monte Carlo")
st.sidebar.header("Cấu hình thông số")

# =========================
# INPUT
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

# =========================
# RISK MANAGEMENT INPUT
# =========================
st.sidebar.subheader("Quản trị rủi ro")

CAPITAL = st.sidebar.number_input(
    "Vốn đầu tư (VNĐ)",
    value=100000000,
    step=10000000
)

RISK_PER_TRADE = st.sidebar.slider(
    "Rủi ro mỗi lệnh (%)",
    0.5,
    )
