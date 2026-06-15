
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Matematiksel Çekirdek
# -----------------------------
def parse_values(text: str):
    cleaned = text.replace("\n", ",").replace(";", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        raise ValueError("Lütfen en az bir değer girin.")
    return np.array([float(p) for p in parts], dtype=float)


def parse_probs(text: str):
    arr = parse_values(text)
    if np.any(arr < 0):
        raise ValueError("Olasılık değerleri negatif olamaz.")
    total = float(np.sum(arr))
    if total <= 0:
        raise ValueError("Olasılıkların toplamı pozitif olmalıdır.")
    return arr


def discrete_table(x, p):
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)
    if len(x) != len(p):
        raise ValueError("x değerleri ile olasılık sayısı aynı olmalıdır.")
    if not np.isclose(p.sum(), 1, atol=1e-6):
        raise ValueError(f"Olasılıkların toplamı 1 olmalıdır. Şu an toplam = {p.sum():.4f}")
    order = np.argsort(x)
    x = x[order]
    p = p[order]
    cdf = np.cumsum(p)
    ex = float(np.sum(x * p))
    ex2 = float(np.sum((x ** 2) * p))
    var = float(ex2 - ex ** 2)
    std = float(np.sqrt(max(var, 0)))
    df = pd.DataFrame({
        "x": x,
        "P(X=x)": np.round(p, 6),
        "F(x)=P(X≤x)": np.round(cdf, 6),
        "x·P(X=x)": np.round(x * p, 6),
        "x²·P(X=x)": np.round((x ** 2) * p, 6),
    })
    return df, ex, ex2, var, std


def plot_pmf(x, p):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(v).rstrip('0').rstrip('.') if float(v).is_integer() else str(v) for v in x], p)
    ax.set_title("Kesikli Olasılık Fonksiyonu")
    ax.set_xlabel("x")
    ax.set_ylabel("P(X=x)")
    plt.tight_layout()
    return fig


def plot_cdf_step(x, p):
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)
    order = np.argsort(x)
    x = x[order]
    cdf = np.cumsum(p[order])
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.r_[x[0] - 1, x, x[-1] + 1]
    ys = np.r_[0, cdf, 1]
    ax.step(xs, ys, where="post")
    ax.scatter(x, cdf)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Birikimli Olasılık Fonksiyonu")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    plt.tight_layout()
    return fig


def plot_uniform_pdf(a, b):
    xs = np.linspace(a - 1, b + 1, 300)
    ys = np.where((xs >= a) & (xs <= b), 1 / (b - a), 0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys)
    ax.fill_between(xs, 0, ys, alpha=0.2)
    ax.set_title("Sürekli Olasılık Yoğunluk Fonksiyonu")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    plt.tight_layout()
    return fig


def plot_uniform_cdf(a, b):
    xs = np.linspace(a - 1, b + 1, 300)
    ys = np.piecewise(xs, [xs < a, (xs >= a) & (xs <= b), xs > b], [0, lambda z: (z-a)/(b-a), 1])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Sürekli Birikimli Dağılım Fonksiyonu")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    plt.tight_layout()
    return fig


def uniform_prob(a, b, left, right):
    l = max(left, a)
    r = min(right, b)
    if r <= l:
        return 0.0
    return (r - l) / (b - a)


def render_bolum4():
    st.title("📚 Bölüm 4: Rastgele Değişkenler ve Çeşitleri")

    menu = [
        "4.1 Rastgele Değişken Kavramı",
        "4.2 Kesikli Rastgele Değişken",
        "4.2.1 Olasılık Fonksiyonu ve Dağılım Fonksiyonu",
        "4.2.2 Beklenen Değer",
        "4.2.3 Varyans ve Standart Sapma",
        "4.3 Sürekli Rastgele Değişken",
        "4.3.1 Yoğunluk ve Dağılım Fonksiyonu",
        "4.3.2 Sürekli Beklenen Değer ve Varyans",
        "🧪 Hesaplama Laboratuvarı",
        "📝 Bölüm Sonu Alıştırmaları",
        "✅ Mini Quiz",
    ]
    choice = st.sidebar.radio("Alt Başlık Seçin", menu)

    if choice == "4.1 Rastgele Değişken Kavramı":
        st.header("4.1 Rastgele Değişken")
        st.markdown("""
Bir deneyin sonucu önceden kesin olarak bilinmiyorsa, deney sonucuna bağlı olarak sayısal değer alan değişkene **rastgele değişken** denir.

Örneğin:
- İki zar atıldığında üst yüzlerin toplamı bir rastgele değişkendir.
- Bir sınıfta sınavdan geçen öğrenci sayısı bir rastgele değişkendir.
- Bir hastanın ağırlığı sürekli rastgele değişkene örnektir.
        """)
        st.info("Rastgele değişken genellikle X, Y, Z ile; aldığı değerler ise x, y, z ile gösterilir.")
        st.latex(r"P(X=x)")

        st.subheader("Kesikli mi, sürekli mi?")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**Kesikli rastgele değişken**

Sayılabilir değerler alır.

Örnek: 0, 1, 2, 3 öğrenci; zar sonucu; hata sayısı.
            """)
        with c2:
            st.markdown("""
**Sürekli rastgele değişken**

Bir aralıkta sonsuz değer alabilir.

Örnek: süre, ağırlık, sıcaklık, uzunluk.
            """)

        example = st.selectbox("Örnek seç", ["İki zarın toplamı", "Bir pilin şarj süresi", "Bir sınıftaki hatalı cevap sayısı", "Bir öğrencinin boyu"])
        if example in ["İki zarın toplamı", "Bir sınıftaki hatalı cevap sayısı"]:
            st.success("Bu örnek kesikli rastgele değişkendir.")
        else:
            st.success("Bu örnek sürekli rastgele değişkendir.")

    elif choice == "4.2 Kesikli Rastgele Değişken":
        st.header("4.2 Kesikli Rastgele Değişken")
        st.markdown("""
Kesikli rastgele değişken, sonlu veya sayılabilir sonsuz değer alabilen rastgele değişkendir.

Bir fonksiyonun kesikli olasılık fonksiyonu olabilmesi için iki temel şart vardır:
        """)
        st.latex(r"P(X=x) \ge 0")
        st.latex(r"\sum_x P(X=x)=1")

        st.subheader("Örnek: İki zarın toplamı")
        x = np.arange(2, 13)
        counts = np.array([1,2,3,4,5,6,5,4,3,2,1])
        p = counts / 36
        df, ex, ex2, var, std = discrete_table(x, p)
        st.dataframe(df[["x", "P(X=x)", "F(x)=P(X≤x)"]], use_container_width=True)
        st.pyplot(plot_pmf(x, p))
        st.info(f"İki zarın toplamı için beklenen değer E(X) = {ex:.2f}, varyans = {var:.2f}, standart sapma = {std:.2f}")

    elif choice == "4.2.1 Olasılık Fonksiyonu ve Dağılım Fonksiyonu":
        st.header("4.2.1 Olasılık Fonksiyonu ve Birikimli Olasılık Fonksiyonu")
        st.markdown("""
**Olasılık fonksiyonu** her x değerine bir olasılık atar.

**Birikimli dağılım fonksiyonu** ise X'in belirli bir değere kadar olan toplam olasılığını verir.
        """)
        st.latex(r"F(x)=P(X \le x)=\sum_{t \le x} P(X=t)")

        st.subheader("Kendi dağılımını oluştur")
        col1, col2 = st.columns(2)
        with col1:
            x_raw = st.text_area("x değerleri", "0, 1, 2", height=80, key="b4_cdf_x")
        with col2:
            p_raw = st.text_area("P(X=x) değerleri", "0.12, 0.28, 0.60", height=80, key="b4_cdf_p")
        try:
            x = parse_values(x_raw)
            p = parse_probs(p_raw)
            df, ex, ex2, var, std = discrete_table(x, p)
            st.dataframe(df, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_pmf(x, p))
            with c2:
                st.pyplot(plot_cdf_step(x, p))
        except ValueError as err:
            st.error(str(err))

    elif choice == "4.2.2 Beklenen Değer":
        st.header("4.2.2 Kesikli Rastgele Değişkenin Beklenen Değeri")
        st.markdown("""
Beklenen değer, rastgele değişkenin uzun dönemdeki olasılıkla ağırlıklandırılmış ortalamasıdır.
        """)
        st.latex(r"E(X)=\mu_X=\sum_x xP(X=x)")

        with st.expander("📝 Örnek: Hilesiz zar"):
            st.write("Bir zar atıldığında üst yüze gelen sayının beklenen değeri:")
            st.latex(r"E(X)=1\cdot\frac16+2\cdot\frac16+\cdots+6\cdot\frac16=3.5")

        st.subheader("Beklenen değer hesaplayıcı")
        col1, col2 = st.columns(2)
        with col1:
            x_raw = st.text_area("x değerleri", "1, 2, 3, 4, 5, 6", height=80, key="b4_ev_x")
        with col2:
            p_raw = st.text_area("P(X=x)", "0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666665", height=80, key="b4_ev_p")
        try:
            x = parse_values(x_raw)
            p = parse_probs(p_raw)
            df, ex, ex2, var, std = discrete_table(x, p)
            st.dataframe(df[["x", "P(X=x)", "x·P(X=x)"]], use_container_width=True)
            st.success(f"E(X) = {ex:.4f}")
        except ValueError as err:
            st.error(str(err))

    elif choice == "4.2.3 Varyans ve Standart Sapma":
        st.header("4.2.3 Kesikli Rastgele Değişkenin Varyansı")
        st.markdown("""
Varyans, rastgele değişkenin beklenen değerden ne kadar uzaklaştığını gösterir.
Standart sapma ise varyansın kareköküdür.
        """)
        st.latex(r"Var(X)=E[(X-\mu)^2]")
        st.latex(r"Var(X)=E(X^2)-[E(X)]^2")
        st.latex(r"\sigma_X=\sqrt{Var(X)}")

        x_raw = st.text_area("x değerleri", "1, 2, 3", height=80, key="b4_var_x")
        p_raw = st.text_area("P(X=x)", "0.25, 0.25, 0.50", height=80, key="b4_var_p")
        try:
            x = parse_values(x_raw)
            p = parse_probs(p_raw)
            df, ex, ex2, var, std = discrete_table(x, p)
            st.dataframe(df[["x", "P(X=x)", "x·P(X=x)", "x²·P(X=x)"]], use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("E(X)", f"{ex:.4f}")
            c2.metric("Var(X)", f"{var:.4f}")
            c3.metric("Standart Sapma", f"{std:.4f}")
        except ValueError as err:
            st.error(str(err))

    elif choice == "4.3 Sürekli Rastgele Değişken":
        st.header("4.3 Sürekli Rastgele Değişken")
        st.markdown("""
Sürekli rastgele değişken, belirli bir aralıkta sonsuz sayıda değer alabilir.

Sürekli değişkende tek bir noktaya ait olasılık sıfırdır. Bu nedenle olasılıklar aralıklar üzerinden hesaplanır.
        """)
        st.latex(r"P(a \le X \le b)=\int_a^b f(x)\,dx")
        st.warning("Sürekli değişkende P(X=2) gibi tek nokta olasılıkları 0 kabul edilir; asıl yorum P(1<X<3) gibi aralıklarladır.")

    elif choice == "4.3.1 Yoğunluk ve Dağılım Fonksiyonu":
        st.header("4.3.1 Olasılık Yoğunluk Fonksiyonu ve Dağılım Fonksiyonu")
        st.markdown("""
Bir fonksiyonun olasılık yoğunluk fonksiyonu olabilmesi için:
        """)
        st.latex(r"f(x)\ge 0")
        st.latex(r"\int_{-\infty}^{+\infty} f(x)\,dx=1")
        st.latex(r"F(x)=P(X\le x)=\int_{-\infty}^{x} f(t)\,dt")

        st.subheader("Uniform dağılım örneği")
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Alt sınır a", value=0.0, key="b4_uni_a")
        with col2:
            b = st.number_input("Üst sınır b", value=4.0, key="b4_uni_b")
        if b <= a:
            st.error("Üst sınır alt sınırdan büyük olmalıdır.")
        else:
            st.latex(r"f(x)=\frac{1}{b-a}")
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_uniform_pdf(a, b))
            with c2:
                st.pyplot(plot_uniform_cdf(a, b))

            left = st.number_input("Olasılık için sol sınır", value=1.0, key="b4_uni_left")
            right = st.number_input("Olasılık için sağ sınır", value=3.0, key="b4_uni_right")
            if right <= left:
                st.error("Sağ sınır sol sınırdan büyük olmalıdır.")
            else:
                st.success(f"P({left} ≤ X ≤ {right}) = {uniform_prob(a, b, left, right):.4f}")

    elif choice == "4.3.2 Sürekli Beklenen Değer ve Varyans":
        st.header("4.3.2 Sürekli Rastgele Değişkenin Beklenen Değeri ve Varyansı")
        st.latex(r"E(X)=\int_{-\infty}^{+\infty} x f(x)\,dx")
        st.latex(r"Var(X)=E(X^2)-[E(X)]^2")

        st.subheader("Uniform dağılım için hızlı hesap")
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("a", value=0.0, key="b4_uni_ev_a")
        with col2:
            b = st.number_input("b", value=4.0, key="b4_uni_ev_b")
        if b <= a:
            st.error("b > a olmalıdır.")
        else:
            mean = (a + b) / 2
            var = ((b - a) ** 2) / 12
            std = math.sqrt(var)
            c1, c2, c3 = st.columns(3)
            c1.metric("E(X)", f"{mean:.4f}")
            c2.metric("Var(X)", f"{var:.4f}")
            c3.metric("σ", f"{std:.4f}")
            st.info("Uniform dağılımda ortalama aralığın tam ortasıdır.")

    elif choice == "🧪 Hesaplama Laboratuvarı":
        st.header("🧪 Bölüm 4 Hesaplama Laboratuvarı")
        mode = st.radio("Dağılım tipi", ["Kesikli dağılım", "Sürekli uniform dağılım"], horizontal=True)
        if mode == "Kesikli dağılım":
            col1, col2 = st.columns(2)
            with col1:
                x_raw = st.text_area("x değerleri", "0, 1, 2, 3", height=100, key="b4_lab_x")
            with col2:
                p_raw = st.text_area("P(X=x)", "0.125, 0.375, 0.375, 0.125", height=100, key="b4_lab_p")
            try:
                x = parse_values(x_raw)
                p = parse_probs(p_raw)
                df, ex, ex2, var, std = discrete_table(x, p)
                st.dataframe(df, use_container_width=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Toplam olasılık", f"{p.sum():.4f}")
                c2.metric("E(X)", f"{ex:.4f}")
                c3.metric("Var(X)", f"{var:.4f}")
                c4.metric("σ", f"{std:.4f}")
                g1, g2 = st.columns(2)
                with g1:
                    st.pyplot(plot_pmf(x, p))
                with g2:
                    st.pyplot(plot_cdf_step(x, p))
            except ValueError as err:
                st.error(str(err))
        else:
            col1, col2 = st.columns(2)
            with col1:
                a = st.number_input("a", value=0.0, key="b4_lab_a")
                left = st.number_input("Sol sınır", value=1.0, key="b4_lab_left")
            with col2:
                b = st.number_input("b", value=10.0, key="b4_lab_b")
                right = st.number_input("Sağ sınır", value=4.0, key="b4_lab_right")
            if b <= a or right <= left:
                st.error("b > a ve sağ sınır > sol sınır olmalıdır.")
            else:
                mean = (a + b) / 2
                var = ((b - a) ** 2) / 12
                c1, c2, c3 = st.columns(3)
                c1.metric("P(sol ≤ X ≤ sağ)", f"{uniform_prob(a,b,left,right):.4f}")
                c2.metric("E(X)", f"{mean:.4f}")
                c3.metric("Var(X)", f"{var:.4f}")
                st.pyplot(plot_uniform_pdf(a, b))

    elif choice == "📝 Bölüm Sonu Alıştırmaları":
        st.header("📝 Bölüm Sonu Alıştırmaları")
        questions = [f"Soru {i}" for i in range(1, 9)]
        q = st.selectbox("Soru seç", questions)

        if q == "Soru 1":
            st.write("İki hilesiz para atılıyor. X, gelen tura sayısı olsun. X'in olasılık fonksiyonunu bulunuz.")
            if st.button("Çözümü göster", key="b4_s1"):
                x = np.array([0,1,2])
                p = np.array([1/4,2/4,1/4])
                df, ex, ex2, var, std = discrete_table(x,p)
                st.dataframe(df, use_container_width=True)
                st.success(f"E(X)={ex:.2f}, Var(X)={var:.2f}")
        elif q == "Soru 2":
            st.write("X değerleri 1,2,3 ve olasılıkları 1/4, 1/4, 2/4 ise E(X) ve Var(X) bulunuz.")
            if st.button("Çözümü göster", key="b4_s2"):
                x = np.array([1,2,3])
                p = np.array([1/4,1/4,2/4])
                df, ex, ex2, var, std = discrete_table(x,p)
                st.dataframe(df, use_container_width=True)
                st.success(f"E(X)={ex:.4f}, E(X²)={ex2:.4f}, Var(X)={var:.4f}, σ={std:.4f}")
        elif q == "Soru 3":
            st.write("Bir zar atılıyor. X üst yüze gelen sayı olsun. Beklenen değeri bulunuz.")
            if st.button("Çözümü göster", key="b4_s3"):
                st.success("E(X)=1/6·(1+2+3+4+5+6)=21/6=3.5")
        elif q == "Soru 4":
            st.write("X iki zarın toplamı olsun. P(X=7) ve P(X≤4) bulunuz.")
            if st.button("Çözümü göster", key="b4_s4"):
                st.success("P(X=7)=6/36=1/6. P(X≤4)=(1+2+3)/36=6/36=1/6.")
        elif q == "Soru 5":
            st.write("f(x)=c(x+1), 1<x<3 için c değerini bulunuz.")
            if st.button("Çözümü göster", key="b4_s5"):
                st.latex(r"\int_1^3 c(x+1)dx=1")
                st.success("İntegral 6c verir. Bu yüzden c=1/6.")
        elif q == "Soru 6":
            st.write("X ~ Uniform(0,4) ise P(1<X<3) kaçtır?")
            if st.button("Çözümü göster", key="b4_s6"):
                st.success("P(1<X<3)=(3-1)/(4-0)=2/4=0.5")
        elif q == "Soru 7":
            st.write("X ~ Uniform(2,8) ise E(X) ve Var(X) bulunuz.")
            if st.button("Çözümü göster", key="b4_s7"):
                st.success("E(X)=(2+8)/2=5. Var(X)=(8-2)²/12=36/12=3.")
        elif q == "Soru 8":
            st.write("Y=2X+1 ve E(X)=3, Var(X)=4 ise E(Y) ve Var(Y) bulunuz.")
            if st.button("Çözümü göster", key="b4_s8"):
                st.success("E(Y)=2E(X)+1=7. Var(Y)=2²Var(X)=16.")

    elif choice == "✅ Mini Quiz":
        st.header("✅ Mini Quiz")
        score = 0
        q1 = st.radio("1) Kesikli rastgele değişken için olasılıkların toplamı kaçtır?", ["0", "1", "n"], key="b4_q1")
        if q1 == "1": score += 1
        q2 = st.radio("2) Sürekli rastgele değişkende P(X=a) kaçtır?", ["0", "1", "f(a)"], key="b4_q2")
        if q2 == "0": score += 1
        q3 = st.radio("3) Beklenen değer neyi temsil eder?", ["En büyük değeri", "Olasılıkla ağırlıklandırılmış ortalamayı", "Sadece medyanı"], key="b4_q3")
        if q3 == "Olasılıkla ağırlıklandırılmış ortalamayı": score += 1
        q4 = st.radio("4) Var(X) için pratik formül hangisidir?", ["E(X²)-[E(X)]²", "E(X)-E(X²)", "P(X=x)"], key="b4_q4")
        if q4 == "E(X²)-[E(X)]²": score += 1
        q5 = st.radio("5) Sürekli dağılımda olasılık nasıl hesaplanır?", ["Toplama ile", "İntegral ile", "Sadece faktöriyel ile"], key="b4_q5")
        if q5 == "İntegral ile": score += 1

        if st.button("Quiz sonucunu hesapla", key="b4_quiz"):
            st.subheader(f"Puan: {score} / 5")
            if score == 5:
                st.success("Mükemmel. Bölüm 4 temelini iyi kavradın.")
            elif score >= 3:
                st.info("İyi gidiyorsun. Dağılım fonksiyonu ve varyans konularını tekrar etmen faydalı olur.")
            else:
                st.warning("Konu anlatımı ve çözümlü örnekleri tekrar incele.")
