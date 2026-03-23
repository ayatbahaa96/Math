import streamlit as st
from bolum1 import render_bolum1
from bolum2 import render_bolum2

st.set_page_config(
    page_title="Olasılık ve İstatistik Platformu",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📘 Olasılık ve İstatistik Web Platformu")

section = st.sidebar.selectbox(
    "Bölüm Seç",
    [
        "Bölüm 1 - Temel İstatistik",
        "Bölüm 2 - Merkezsel Eğilim Ölçüleri ve Dağılım Ölçüleri",
    ],
)

if section == "Bölüm 1 - Temel İstatistik":
    render_bolum1()
else:
    render_bolum2()
