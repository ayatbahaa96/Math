import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2, f


def _ci_mean_known(xbar, sigma, n, conf):
    alpha = 1 - conf
    z = norm.ppf(1 - alpha / 2)
    e = z * sigma / math.sqrt(n)
    return xbar - e, xbar + e, e, z


def _ci_mean_unknown(xbar, s, n, conf):
    alpha = 1 - conf
    tv = t.ppf(1 - alpha / 2, n - 1)
    e = tv * s / math.sqrt(n)
    return xbar - e, xbar + e, e, tv


def _plot_ci(center, low, high, title="Güven Aralığı"):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.hlines(1, low, high, linewidth=4)
    ax.plot(center, 1, marker="o")
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("Parametre değeri")
    pad = (high - low) * 0.25 if high > low else 1
    ax.set_xlim(low - pad, high + pad)
    plt.tight_layout()
    return fig


def render_bolum7():
    st.title("📚 Bölüm 7: İstatistiksel Çıkarım ve Tahmin")
    menu = [
        "7.1 Tahmin Kavramı ve Merkezi Limit Teoremi",
        "7.2 Nokta Tahmini ve Yansızlık",
        "7.3 Ortalama Güven Aralığı: σ Biliniyor",
        "7.4 Ortalama Güven Aralığı: σ Bilinmiyor",
        "7.5 Tek Yönlü Güven Sınırı",
        "7.6 Örneklem Büyüklüğü",
        "7.7 İki Ortalama Farkı",
        "7.8 Eşleştirilmiş Gözlemler",
        "7.9 Oran ve İki Oran Farkı",
        "7.10 Varyans ve Varyans Oranı",
        "🧪 Güven Aralığı Laboratuvarı",
        "✅ Mini Quiz",
    ]
    choice = st.sidebar.radio("Alt Başlık Seçin", menu)

    if choice == "7.1 Tahmin Kavramı ve Merkezi Limit Teoremi":
        st.header("7.1 Tahmin ve Merkezi Limit Teoremi")
        st.markdown("""
İstatistiksel çıkarımda amaç, örneklemden elde edilen bilgiyle anakütle parametresi hakkında yorum yapmaktır.

- **Nokta tahmini:** Parametre için tek değer verir.
- **Aralık tahmini:** Parametrenin düşmesi beklenen aralığı verir.
        """)
        st.latex(r"\bar X \approx N\left(\mu,\frac{\sigma}{\sqrt n}\right)")
        n = st.slider("Örneklem büyüklüğü n", 1, 100, 30)
        means = [np.mean(np.random.exponential(1, n)) for _ in range(1000)]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(means, bins=35)
        ax.set_title("Merkezi Limit Teoremi Simülasyonu")
        ax.set_xlabel("Örneklem ortalaması")
        ax.set_ylabel("Frekans")
        st.pyplot(fig)

    elif choice == "7.2 Nokta Tahmini ve Yansızlık":
        st.header("7.2 Nokta Tahmini ve Yansızlık")
        st.markdown("""
Bir tahmin edicinin beklenen değeri tahmin edilen parametreye eşitse bu tahmin edici **yansız**dır.
        """)
        st.latex(r"E(\hat\theta)=\theta")
        st.info("Örneklem ortalaması anakütle ortalaması için yansız tahmin edicidir. Örneklem varyansında n-1 kullanılması da yansızlık içindir.")
        st.latex(r"S^2=\frac{\sum_{i=1}^n (X_i-\bar X)^2}{n-1}")

    elif choice == "7.3 Ortalama Güven Aralığı: σ Biliniyor":
        st.header("7.3 Tek Örneklem Ortalama Tahmini: σ Biliniyor")
        st.latex(r"\bar X - z_{\alpha/2}\frac{\sigma}{\sqrt n}<\mu<\bar X + z_{\alpha/2}\frac{\sigma}{\sqrt n}")
        c1, c2, c3, c4 = st.columns(4)
        xbar = c1.number_input("Örneklem ortalaması", value=2.6)
        sigma = c2.number_input("σ", min_value=0.0001, value=0.3)
        n = c3.number_input("n", min_value=1, value=36)
        conf = c4.selectbox("Güven", [0.90, 0.95, 0.99], index=1)
        low, high, e, z = _ci_mean_known(xbar, sigma, n, conf)
        st.success(f"{conf*100:.0f}% güven aralığı: ({low:.4f}, {high:.4f})")
        st.write(f"z = {z:.4f}, hata payı = {e:.4f}")
        st.pyplot(_plot_ci(xbar, low, high))

    elif choice == "7.4 Ortalama Güven Aralığı: σ Bilinmiyor":
        st.header("7.4 Tek Örneklem Ortalama Tahmini: σ Bilinmiyor")
        st.latex(r"\bar X - t_{\alpha/2,n-1}\frac{s}{\sqrt n}<\mu<\bar X + t_{\alpha/2,n-1}\frac{s}{\sqrt n}")
        raw = st.text_area("Veri girin", "9.8,10.2,10.4,9.8,10.0,10.2,9.6", height=90)
        conf = st.selectbox("Güven düzeyi", [0.90, 0.95, 0.99], index=1, key="b7_unknown_conf")
        try:
            data = np.array([float(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()])
            n = len(data); xbar = float(np.mean(data)); s = float(np.std(data, ddof=1))
            low, high, e, tv = _ci_mean_unknown(xbar, s, n, conf)
            st.write(f"n={n}, x̄={xbar:.4f}, s={s:.4f}, t={tv:.4f}")
            st.success(f"Güven aralığı: ({low:.4f}, {high:.4f})")
            st.pyplot(_plot_ci(xbar, low, high))
        except Exception as err:
            st.error(str(err))

    elif choice == "7.5 Tek Yönlü Güven Sınırı":
        st.header("7.5 Tek Yönlü Güven Sınırı")
        st.latex(r"\text{Üst sınır}=\bar X+z_\alpha\frac{\sigma}{\sqrt n}")
        st.latex(r"\text{Alt sınır}=\bar X-z_\alpha\frac{\sigma}{\sqrt n}")
        xbar = st.number_input("x̄", value=6.2)
        sigma = st.number_input("σ", value=2.0, min_value=0.0001)
        n = st.number_input("n", value=25, min_value=1)
        conf = st.selectbox("Güven", [0.90, 0.95, 0.99], index=1, key="one_sided_conf")
        side = st.radio("Sınır", ["Üst sınır", "Alt sınır"], horizontal=True)
        z = norm.ppf(conf)
        bound = xbar + z * sigma / math.sqrt(n) if side == "Üst sınır" else xbar - z * sigma / math.sqrt(n)
        st.success(f"{conf*100:.0f}% {side}: {bound:.4f}")

    elif choice == "7.6 Örneklem Büyüklüğü":
        st.header("7.6 Örneklem Büyüklüğü")
        st.latex(r"n=\left(\frac{z_{\alpha/2}\sigma}{e}\right)^2")
        sigma = st.number_input("σ", value=0.3, min_value=0.0001)
        e = st.number_input("Maksimum hata e", value=0.05, min_value=0.0001)
        conf = st.selectbox("Güven", [0.90, 0.95, 0.99], index=1, key="sample_size_conf")
        z = norm.ppf(1 - (1-conf)/2)
        n = math.ceil((z * sigma / e) ** 2)
        st.success(f"Gerekli örneklem büyüklüğü: n = {n}")

    elif choice == "7.7 İki Ortalama Farkı":
        st.header("7.7 İki Ortalama Farkı İçin Güven Aralığı")
        st.latex(r"(\bar X_1-\bar X_2)\pm z_{\alpha/2}\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}");
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.number_input("x̄1", value=42.0); s1 = st.number_input("σ1 / s1", value=8.0); n1 = st.number_input("n1", value=75, min_value=2)
        with c2:
            x2 = st.number_input("x̄2", value=36.0); s2 = st.number_input("σ2 / s2", value=6.0); n2 = st.number_input("n2", value=50, min_value=2)
        conf = st.selectbox("Güven", [0.90, 0.95, 0.96, 0.99], index=2, key="two_mean_conf")
        z = norm.ppf(1 - (1-conf)/2)
        diff = x1 - x2
        e = z * math.sqrt(s1**2/n1 + s2**2/n2)
        st.success(f"μ1-μ2 için güven aralığı: ({diff-e:.4f}, {diff+e:.4f})")

    elif choice == "7.8 Eşleştirilmiş Gözlemler":
        st.header("7.8 Eşleştirilmiş Gözlemler")
        st.markdown("Aynı birimden iki ölçüm varsa farklar üzerinden tek örneklem t aralığı kurulur.")
        st.latex(r"\bar d \pm t_{\alpha/2,n-1}\frac{s_d}{\sqrt n}")
        before = st.text_area("1. ölçümler", "5.2,4.8,6.1,5.5,5.9", key="paired_before")
        after = st.text_area("2. ölçümler", "4.9,4.6,5.8,5.1,5.6", key="paired_after")
        conf = st.selectbox("Güven", [0.90,0.95,0.99], index=1, key="paired_conf")
        try:
            a = np.array([float(x.strip()) for x in before.replace(";", ",").split(",") if x.strip()])
            b = np.array([float(x.strip()) for x in after.replace(";", ",").split(",") if x.strip()])
            if len(a) != len(b): raise ValueError("İki liste aynı uzunlukta olmalıdır.")
            d = a - b; n = len(d); db = np.mean(d); sd = np.std(d, ddof=1); tv = t.ppf(1-(1-conf)/2, n-1)
            e = tv * sd / math.sqrt(n)
            st.success(f"Ortalama fark güven aralığı: ({db-e:.4f}, {db+e:.4f})")
            st.dataframe(pd.DataFrame({"d": d}))
        except Exception as err:
            st.error(str(err))

    elif choice == "7.9 Oran ve İki Oran Farkı":
        st.header("7.9 Oran Tahmini ve İki Oran Farkı")
        tab1, tab2 = st.tabs(["Tek oran", "İki oran farkı"])
        with tab1:
            x = st.number_input("Başarı sayısı x", value=340, min_value=0)
            n = st.number_input("n", value=500, min_value=1, key="prop_n")
            conf = st.selectbox("Güven", [0.90,0.95,0.99], index=1, key="prop_conf")
            ph = x/n; z = norm.ppf(1-(1-conf)/2); e = z*math.sqrt(ph*(1-ph)/n)
            st.success(f"p için güven aralığı: ({ph-e:.4f}, {ph+e:.4f})")
        with tab2:
            x1 = st.number_input("x1", value=75, min_value=0); n1 = st.number_input("n1", value=1500, min_value=1, key="p1n")
            x2 = st.number_input("x2", value=80, min_value=0); n2 = st.number_input("n2", value=2000, min_value=1, key="p2n")
            conf = st.selectbox("Güven", [0.90,0.95,0.99], index=0, key="propdiff_conf")
            p1=x1/n1; p2=x2/n2; z=norm.ppf(1-(1-conf)/2); e=z*math.sqrt(p1*(1-p1)/n1+p2*(1-p2)/n2)
            st.success(f"p1-p2 için güven aralığı: ({p1-p2-e:.4f}, {p1-p2+e:.4f})")

    elif choice == "7.10 Varyans ve Varyans Oranı":
        st.header("7.10 Varyans ve Varyans Oranı")
        tab1, tab2 = st.tabs(["Tek varyans", "Varyans oranı"])
        with tab1:
            raw = st.text_area("Veri", "46.4,46.1,45.8,47.0,46.1,45.9,45.8,46.9,45.2,46.0")
            conf = st.selectbox("Güven", [0.90,0.95,0.99], index=1, key="var_conf")
            data=np.array([float(v.strip()) for v in raw.replace(';', ',').split(',') if v.strip()])
            n=len(data); s2=np.var(data, ddof=1); alpha=1-conf
            low=(n-1)*s2/chi2.ppf(1-alpha/2, n-1); high=(n-1)*s2/chi2.ppf(alpha/2, n-1)
            st.success(f"σ² için güven aralığı: ({low:.4f}, {high:.4f})")
        with tab2:
            s1=st.number_input("s1", value=3.07); n1=st.number_input("n1", value=15, min_value=2, key="frn1")
            s2=st.number_input("s2", value=0.80); n2=st.number_input("n2", value=12, min_value=2, key="frn2")
            conf=st.selectbox("Güven", [0.90,0.95,0.98], index=2, key="fr_conf")
            alpha=1-conf; ratio=(s1**2)/(s2**2)
            low=ratio / f.ppf(1-alpha/2, n1-1, n2-1); high=ratio / f.ppf(alpha/2, n1-1, n2-1)
            st.success(f"σ1²/σ2² için güven aralığı: ({low:.4f}, {high:.4f})")

    elif choice == "🧪 Güven Aralığı Laboratuvarı":
        st.header("🧪 Güven Aralığı Laboratuvarı")
        calc = st.selectbox("Hesap türü", ["Ortalama - σ biliniyor", "Ortalama - σ bilinmiyor", "Tek oran", "Örneklem büyüklüğü"])
        st.info("Bu laboratuvar, önceki alt başlıklardaki formülleri hızlı uygulamak için hazırlanmıştır.")
        if calc == "Ortalama - σ biliniyor":
            xbar=st.number_input("x̄", value=100.0); sigma=st.number_input("σ", value=15.0); n=st.number_input("n", value=30, min_value=1); conf=st.slider("Güven",0.80,0.99,0.95)
            low, high, e, z = _ci_mean_known(xbar, sigma, n, conf); st.success(f"({low:.4f}, {high:.4f})")
        elif calc == "Ortalama - σ bilinmiyor":
            xbar=st.number_input("x̄", value=100.0); s=st.number_input("s", value=15.0); n=st.number_input("n", value=30, min_value=2); conf=st.slider("Güven",0.80,0.99,0.95,key="labs")
            low, high, e, tv = _ci_mean_unknown(xbar, s, n, conf); st.success(f"({low:.4f}, {high:.4f})")
        elif calc == "Tek oran":
            x=st.number_input("x", value=40, min_value=0); n=st.number_input("n", value=100, min_value=1,key="labpn"); conf=st.slider("Güven",0.80,0.99,0.95,key="labpc")
            ph=x/n; z=norm.ppf(1-(1-conf)/2); e=z*math.sqrt(ph*(1-ph)/n); st.success(f"({ph-e:.4f}, {ph+e:.4f})")
        else:
            sigma=st.number_input("σ", value=10.0,key="labss"); e=st.number_input("e", value=2.0,key="labse"); conf=st.slider("Güven",0.80,0.99,0.95,key="labsc")
            z=norm.ppf(1-(1-conf)/2); st.success(f"n={math.ceil((z*sigma/e)**2)}")

    elif choice == "✅ Mini Quiz":
        st.header("✅ Mini Quiz")
        score=0
        q1=st.radio("1) σ bilinmiyorsa ortalama güven aralığında hangi dağılım kullanılır?", ["t", "z", "F"], key="b7q1")
        if q1=="t": score+=1
        q2=st.radio("2) Güven düzeyi artarsa aralık genelde ne olur?", ["Genişler", "Daralır", "Değişmez"], key="b7q2")
        if q2=="Genişler": score+=1
        q3=st.radio("3) Tek oran tahmininde nokta tahmini nedir?", ["x/n", "s/n", "x+s"], key="b7q3")
        if q3=="x/n": score+=1
        if st.button("Bölüm 7 quiz sonucunu hesapla"):
            st.success(f"Puan: {score}/3")
