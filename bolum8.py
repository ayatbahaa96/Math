import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats


# -----------------------------
# Yardımcı Fonksiyonlar
# -----------------------------
def parse_pair_data(text: str):
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.replace(";", ",")
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("Her satır 'x, y' formatında olmalıdır.")
        rows.append((float(parts[0].replace(',', '.')), float(parts[1].replace(',', '.'))))
    if len(rows) < 3:
        raise ValueError("En az 3 ikili gözlem girilmelidir.")
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def correlation_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x ve y gözlem sayıları eşit olmalıdır.")
    if len(x) < 3:
        raise ValueError("En az 3 gözlem gerekir.")
    n = len(x)
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if sx == 0 or sy == 0:
        raise ValueError("x veya y değişkeninde değişim yok; korelasyon hesaplanamaz.")

    sxx = float(np.sum((x - xbar) ** 2))
    syy = float(np.sum((y - ybar) ** 2))
    sxy = float(np.sum((x - xbar) * (y - ybar)))
    r = float(sxy / math.sqrt(sxx * syy))
    b1 = float(sxy / sxx)
    b0 = float(ybar - b1 * xbar)
    yhat = b0 + b1 * x
    residuals = y - yhat
    sse = float(np.sum(residuals ** 2))
    ssr = float(np.sum((yhat - ybar) ** 2))
    sst = float(np.sum((y - ybar) ** 2))
    mse = sse / (n - 2)
    msr = ssr / 1
    f_value = msr / mse if mse > 0 else np.inf
    r2 = ssr / sst if sst > 0 else 0.0
    se_b1 = math.sqrt(mse / sxx) if sxx > 0 else np.nan
    t_b1 = b1 / se_b1 if se_b1 > 0 else np.inf
    p_b1 = 2 * (1 - stats.t.cdf(abs(t_b1), df=n - 2))
    t_r = r * math.sqrt((n - 2) / (1 - r ** 2)) if abs(r) < 1 else np.inf
    p_r = 2 * (1 - stats.t.cdf(abs(t_r), df=n - 2)) if np.isfinite(t_r) else 0.0

    return {
        "n": n,
        "xbar": xbar,
        "ybar": ybar,
        "sxx": sxx,
        "syy": syy,
        "sxy": sxy,
        "r": r,
        "r2": r2,
        "b0": b0,
        "b1": b1,
        "yhat": yhat,
        "residuals": residuals,
        "sst": sst,
        "ssr": ssr,
        "sse": sse,
        "msr": msr,
        "mse": mse,
        "f": f_value,
        "t_b1": t_b1,
        "p_b1": p_b1,
        "t_r": t_r,
        "p_r": p_r,
    }


def plot_scatter_regression(x, y, b0=None, b1=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y)
    ax.set_title("Saçılma Diyagramı")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if b0 is not None and b1 is not None:
        xs = np.linspace(min(x), max(x), 100)
        ax.plot(xs, b0 + b1 * xs)
    plt.tight_layout()
    return fig


def corr_comment(r):
    ar = abs(r)
    yon = "pozitif" if r > 0 else "negatif" if r < 0 else "yok"
    if ar >= 0.90:
        guc = "çok güçlü"
    elif ar >= 0.70:
        guc = "güçlü"
    elif ar >= 0.40:
        guc = "orta düzey"
    elif ar >= 0.20:
        guc = "zayıf"
    else:
        guc = "çok zayıf"
    return f"{guc} {yon} doğrusal ilişki"


def default_sales_data():
    return """15, 1.35
18, 1.63
24, 2.33
22, 2.41
25, 2.63
29, 2.93
30, 3.41
32, 3.26
35, 3.63
38, 4.15"""


def default_height_data():
    return """162, 165
163, 161
158, 156
156, 158
161, 163
166, 166
153, 154
154, 156
161, 161
157, 159"""


# -----------------------------
# Render
# -----------------------------
def render_bolum8():
    st.title("📚 Bölüm 8: Korelasyon ve Regresyon Analizi")

    menu = [
        "8.1 Korelasyon Kavramı",
        "8.2 Saçılma Diyagramı",
        "8.3 Korelasyon Katsayısı r",
        "8.4 Korelasyon Anlamlılık Testi",
        "8.5 Basit Doğrusal Regresyon",
        "8.6 En Küçük Kareler ve Tahmin",
        "8.7 Regresyon ANOVA ve R²",
        "🧪 Korelasyon-Regresyon Laboratuvarı",
        "📝 Çözümlü Örnekler",
        "✅ Mini Quiz",
    ]
    choice = st.sidebar.radio("Alt Başlık Seçin", menu)

    if choice == "8.1 Korelasyon Kavramı":
        st.header("8.1 Korelasyon")
        st.markdown(
            """
**Korelasyon**, iki değişkenin birlikte doğrusal olarak değişme derecesini inceler.

- x artarken y de artıyorsa **pozitif korelasyon** vardır.
- x artarken y azalıyorsa **negatif korelasyon** vardır.
- x ve y arasında doğrusal yapı yoksa korelasyon zayıf veya sıfıra yakın olabilir.

Önemli not: **Korelasyon nedensellik göstermez.** İki değişken birlikte değişiyor diye biri diğerinin kesin sebebidir denemez.
            """
        )
        st.warning("Korelasyon yalnızca doğrusal ilişkiyi ölçer. Doğrusal olmayan güçlü bir ilişki varsa r düşük çıkabilir.")
        st.latex(r"-1 \le r \le 1")

    elif choice == "8.2 Saçılma Diyagramı":
        st.header("8.2 Saçılma Diyagramı")
        st.markdown(
            """
Saçılma diyagramı, her gözlemi **(x, y)** noktası olarak koordinat düzleminde gösterir.
Bu grafik ilişki yönünü, gücünü ve olası doğrusal olmayan yapıları görmeye yarar.
            """
        )
        raw = st.text_area("İkili veri girin: her satır x, y", default_sales_data(), height=220)
        try:
            x, y = parse_pair_data(raw)
            st.pyplot(plot_scatter_regression(x, y))
            st.dataframe(pd.DataFrame({"x": x, "y": y}), use_container_width=True)
        except ValueError as err:
            st.error(str(err))

    elif choice == "8.3 Korelasyon Katsayısı r":
        st.header("8.3 Korelasyon Katsayısı")
        st.latex(r"r=\frac{n\sum xy-(\sum x)(\sum y)}{\sqrt{[n\sum x^2-(\sum x)^2][n\sum y^2-(\sum y)^2]}}")
        raw = st.text_area("İkili veri girin", default_sales_data(), height=220, key="b8_corr")
        try:
            x, y = parse_pair_data(raw)
            res = correlation_regression(x, y)
            c1, c2, c3 = st.columns(3)
            c1.metric("r", f"{res['r']:.4f}")
            c2.metric("r²", f"{res['r2']:.4f}")
            c3.metric("Yorum", corr_comment(res["r"]))
            st.pyplot(plot_scatter_regression(x, y))
            calc_df = pd.DataFrame({
                "x": x,
                "y": y,
                "x²": x ** 2,
                "y²": y ** 2,
                "xy": x * y,
            })
            st.dataframe(calc_df, use_container_width=True)
            st.info(
                f"Toplamlar: Σx={calc_df['x'].sum():.4f}, Σy={calc_df['y'].sum():.4f}, "
                f"Σx²={calc_df['x²'].sum():.4f}, Σy²={calc_df['y²'].sum():.4f}, Σxy={calc_df['xy'].sum():.4f}"
            )
        except ValueError as err:
            st.error(str(err))

    elif choice == "8.4 Korelasyon Anlamlılık Testi":
        st.header("8.4 Anakütle Korelasyon Katsayısının Testi")
        st.markdown("Hipotezler:")
        st.latex(r"H_0: \rho = 0")
        st.latex(r"H_1: \rho \ne 0")
        st.latex(r"t=\frac{r\sqrt{n-2}}{\sqrt{1-r^2}}")
        raw = st.text_area("İkili veri girin", default_sales_data(), height=220, key="b8_test")
        alpha = st.selectbox("Anlamlılık düzeyi", [0.10, 0.05, 0.01], index=1)
        try:
            x, y = parse_pair_data(raw)
            res = correlation_regression(x, y)
            df = res["n"] - 2
            tcrit = stats.t.ppf(1 - alpha / 2, df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("r", f"{res['r']:.4f}")
            c2.metric("t hesap", f"{res['t_r']:.4f}")
            c3.metric("t kritik", f"±{tcrit:.4f}")
            c4.metric("p-değeri", f"{res['p_r']:.6f}")
            if abs(res["t_r"]) > tcrit:
                st.success("Karar: H0 reddedilir. Korelasyon istatistiksel olarak anlamlıdır.")
            else:
                st.warning("Karar: H0 reddedilemez. Anlamlı doğrusal korelasyon için yeterli kanıt yoktur.")
        except ValueError as err:
            st.error(str(err))

    elif choice == "8.5 Basit Doğrusal Regresyon":
        st.header("8.5 Basit Doğrusal Regresyon")
        st.markdown(
            """
Basit doğrusal regresyonda amaç, bağımsız değişken x ile bağımlı değişken y arasındaki doğrusal ilişkiyi modellemektir.
            """
        )
        st.latex(r"Y=\beta_0+\beta_1X+\varepsilon")
        st.markdown(
            """
- **β₀:** x = 0 iken y'nin beklenen değeri, yani kesim noktası.
- **β₁:** x bir birim arttığında y'nin ortalama ne kadar değiştiğini gösterir.
- **ε:** hata terimidir.
            """
        )
        st.info("Basit doğrusal regresyonda tek bağımsız değişken vardır. Birden fazla bağımsız değişken varsa çoklu regresyon kullanılır.")

    elif choice == "8.6 En Küçük Kareler ve Tahmin":
        st.header("8.6 En Küçük Kareler Yöntemi")
        st.latex(r"\hat{y}=b_0+b_1x")
        st.latex(r"b_1=\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sum (x_i-\bar{x})^2}")
        st.latex(r"b_0=\bar{y}-b_1\bar{x}")
        raw = st.text_area("İkili veri girin", default_height_data(), height=220, key="b8_reg")
        predict_x = st.number_input("Tahmin için x değeri", value=160.0)
        try:
            x, y = parse_pair_data(raw)
            res = correlation_regression(x, y)
            pred = res["b0"] + res["b1"] * predict_x
            c1, c2, c3 = st.columns(3)
            c1.metric("b0", f"{res['b0']:.4f}")
            c2.metric("b1", f"{res['b1']:.4f}")
            c3.metric("Tahmin", f"ŷ={pred:.4f}")
            st.success(f"Regresyon modeli: ŷ = {res['b0']:.4f} + {res['b1']:.4f}x")
            st.pyplot(plot_scatter_regression(x, y, res["b0"], res["b1"]))
            out = pd.DataFrame({"x": x, "y": y, "ŷ": res["yhat"], "artık e=y-ŷ": res["residuals"]})
            st.dataframe(out, use_container_width=True)
        except ValueError as err:
            st.error(str(err))

    elif choice == "8.7 Regresyon ANOVA ve R²":
        st.header("8.7 Regresyon ANOVA ve Açıklama Katsayısı")
        st.markdown("Regresyon modelinin genel geçerliliği F testiyle incelenebilir.")
        st.latex(r"F_H=\frac{RKO}{RAKO}")
        st.latex(r"R^2=\frac{RKT}{YOAKT}")
        raw = st.text_area("İkili veri girin", default_height_data(), height=220, key="b8_anova")
        alpha = st.selectbox("Anlamlılık düzeyi", [0.10, 0.05, 0.01], index=1, key="b8_anova_alpha")
        try:
            x, y = parse_pair_data(raw)
            res = correlation_regression(x, y)
            n = res["n"]
            fcrit = stats.f.ppf(1 - alpha, 1, n - 2)
            anova = pd.DataFrame({
                "Varyasyon Kaynağı": ["Regresyon", "Hata / Artık", "Toplam"],
                "sd": [1, n - 2, n - 1],
                "KT": [res["ssr"], res["sse"], res["sst"]],
                "KO": [res["msr"], res["mse"], ""],
                "F": [res["f"], "", ""],
            })
            st.dataframe(anova, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("F hesap", f"{res['f']:.4f}")
            c2.metric("F kritik", f"{fcrit:.4f}")
            c3.metric("R²", f"{res['r2']:.4f}")
            if res["f"] > fcrit:
                st.success("Karar: Model anlamlıdır. Doğrusal regresyon kullanılabilir.")
            else:
                st.warning("Karar: Modelin anlamlılığı için yeterli kanıt yoktur.")
            st.info(f"Yorum: y değişkenindeki değişimin yaklaşık %{res['r2']*100:.2f}'i x tarafından açıklanır.")
        except ValueError as err:
            st.error(str(err))

    elif choice == "🧪 Korelasyon-Regresyon Laboratuvarı":
        st.header("🧪 Korelasyon ve Regresyon Laboratuvarı")
        st.write("Kendi verinizi girerek korelasyon, regresyon modeli, testler ve ANOVA tablosu elde edebilirsiniz.")
        raw = st.text_area("Her satır x, y olacak şekilde veri girin", default_sales_data(), height=250, key="b8_lab")
        alpha = st.selectbox("α", [0.10, 0.05, 0.01], index=1, key="b8_lab_alpha")
        try:
            x, y = parse_pair_data(raw)
            res = correlation_regression(x, y)
            tcrit = stats.t.ppf(1 - alpha / 2, res["n"] - 2)
            fcrit = stats.f.ppf(1 - alpha, 1, res["n"] - 2)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("r", f"{res['r']:.4f}")
            c2.metric("R²", f"{res['r2']:.4f}")
            c3.metric("b0", f"{res['b0']:.4f}")
            c4.metric("b1", f"{res['b1']:.4f}")
            st.pyplot(plot_scatter_regression(x, y, res["b0"], res["b1"]))
            st.success(f"Model: ŷ = {res['b0']:.4f} + {res['b1']:.4f}x")
            st.write(f"Korelasyon testi: t={res['t_r']:.4f}, kritik=±{tcrit:.4f}, p={res['p_r']:.6f}")
            st.write(f"Regresyon katsayısı testi: t={res['t_b1']:.4f}, p={res['p_b1']:.6f}")
            st.write(f"Model F testi: F={res['f']:.4f}, F kritik={fcrit:.4f}")
            detail = pd.DataFrame({"x": x, "y": y, "ŷ": res["yhat"], "artık": res["residuals"]})
            st.dataframe(detail, use_container_width=True)
        except ValueError as err:
            st.error(str(err))

    elif choice == "📝 Çözümlü Örnekler":
        st.header("📝 Çözümlü Örnekler")
        with st.expander("Örnek 1: Satış personeli ve satış geliri korelasyonu"):
            st.write("PPT’deki satış personeli sayısı ile satış geliri verileri kullanılır.")
            x, y = parse_pair_data(default_sales_data())
            res = correlation_regression(x, y)
            st.dataframe(pd.DataFrame({"x": x, "y": y, "x²": x**2, "y²": y**2, "xy": x*y}), use_container_width=True)
            st.write(f"r = {res['r']:.4f}. Bu sonuç güçlü pozitif korelasyon gösterir.")
            st.pyplot(plot_scatter_regression(x, y, res["b0"], res["b1"]))

        with st.expander("Örnek 2: Boy uzunluğu ve kulaç uzunluğu regresyonu"):
            x, y = parse_pair_data(default_height_data())
            res = correlation_regression(x, y)
            st.write("Bağımsız değişken: kulaç uzunluğu. Bağımlı değişken: boy uzunluğu.")
            st.success(f"ŷ = {res['b0']:.4f} + {res['b1']:.4f}x")
            st.write(f"R² = {res['r2']:.4f}. Boy uzunluğundaki değişimin yaklaşık %{res['r2']*100:.2f}'i kulaç uzunluğu ile açıklanır.")
            st.pyplot(plot_scatter_regression(x, y, res["b0"], res["b1"]))

        with st.expander("Örnek 3: Tahmin"):
            x, y = parse_pair_data(default_height_data())
            res = correlation_regression(x, y)
            px = 160
            py = res["b0"] + res["b1"] * px
            st.write(f"Kulaç uzunluğu {px} cm olan bir çocuk için tahmini boy:")
            st.success(f"ŷ = {res['b0']:.4f} + {res['b1']:.4f}({px}) = {py:.2f} cm")

    elif choice == "✅ Mini Quiz":
        st.header("✅ Mini Quiz")
        score = 0
        q1 = st.radio("1) Korelasyon katsayısı hangi aralıktadır?", ["0 ile 1", "-1 ile 1", "-∞ ile +∞"], key="b8q1")
        if q1 == "-1 ile 1":
            score += 1
        q2 = st.radio("2) r=0.98 neyi gösterir?", ["Güçlü pozitif doğrusal ilişki", "Nedensellik kesin kanıtlandı", "İlişki yok"], key="b8q2")
        if q2 == "Güçlü pozitif doğrusal ilişki":
            score += 1
        q3 = st.radio("3) Basit doğrusal regresyonda bağımlı değişken genellikle hangi harfle gösterilir?", ["x", "y", "n"], key="b8q3")
        if q3 == "y":
            score += 1
        q4 = st.radio("4) R² neyi ifade eder?", ["Açıklanan değişim oranı", "Örneklem büyüklüğü", "Kritik t değeri"], key="b8q4")
        if q4 == "Açıklanan değişim oranı":
            score += 1
        q5 = st.radio("5) Korelasyonun en yaygın yanlış yorumu hangisidir?", ["Nedensellik sanmak", "Grafik çizmek", "Veriyi sıralamak"], key="b8q5")
        if q5 == "Nedensellik sanmak":
            score += 1

        if st.button("Bölüm 8 quiz sonucunu hesapla"):
            st.subheader(f"Puan: {score} / 5")
            if score == 5:
                st.success("Harika. Korelasyon ve regresyon mantığı iyi anlaşılmış.")
            elif score >= 3:
                st.info("İyi. r, R² ve regresyon katsayısı yorumlarını tekrar etmek faydalı olur.")
            else:
                st.warning("Konu anlatımı ve çözümlü örnekleri tekrar inceleyin.")
