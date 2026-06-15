import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def _norm_table(df, rows, cols):
    arr = np.array(df, dtype=float)
    if arr.shape != (rows, cols):
        raise ValueError(f"Tablo {rows}x{cols} boyutunda olmalıdır.")
    if np.any(arr < 0):
        raise ValueError("Olasılık değerleri negatif olamaz.")
    s = arr.sum()
    if s <= 0:
        raise ValueError("Toplam olasılık pozitif olmalıdır.")
    return arr / s


def _joint_df(pxy, x_vals, y_vals):
    df = pd.DataFrame(pxy, index=[f"X={x}" for x in x_vals], columns=[f"Y={y}" for y in y_vals])
    df["P(X=x)"] = pxy.sum(axis=1)
    bottom = list(pxy.sum(axis=0)) + [1.0]
    df.loc["P(Y=y)"] = bottom
    return df


def _cov_corr(pxy, x_vals, y_vals):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    ex = float((x * px).sum())
    ey = float((y * py).sum())
    ex2 = float(((x ** 2) * px).sum())
    ey2 = float(((y ** 2) * py).sum())
    exy = float(sum(x[i] * y[j] * pxy[i, j] for i in range(len(x)) for j in range(len(y))))
    vx = ex2 - ex ** 2
    vy = ey2 - ey ** 2
    cov = exy - ex * ey
    corr = cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0
    return ex, ey, vx, vy, cov, corr


def _heatmap(pxy, x_vals, y_vals):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pxy)
    ax.set_xticks(range(len(y_vals)), [str(y) for y in y_vals])
    ax.set_yticks(range(len(x_vals)), [str(x) for x in x_vals])
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_title("Ortak Olasılık Tablosu")
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            ax.text(j, i, f"{pxy[i,j]:.3f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def render_bolum6():
    st.title("📚 Bölüm 6: İki Değişkenli Rastgele Değişkenler")
    menu = [
        "6.1 Ortak Olasılık Dağılımı",
        "6.2 Marjinal Dağılımlar",
        "6.3 Koşullu Olasılık Fonksiyonu",
        "6.4 Bağımsızlık Kontrolü",
        "6.5 Koşullu Beklenen Değer ve Varyans",
        "6.6 Sürekli Ortak Yoğunluk",
        "6.7 Kovaryans ve Korelasyon",
        "🧪 İki Değişkenli Laboratuvar",
        "✅ Mini Quiz",
    ]
    choice = st.sidebar.radio("Alt Başlık Seçin", menu)

    default = np.array([[0.14, 0.10], [0.10, 0.20], [0.26, 0.20]])
    x_vals = [0, 1, 2]
    y_vals = [0, 1]

    if choice == "6.1 Ortak Olasılık Dağılımı":
        st.header("6.1 Ortak Olasılık Dağılımı")
        st.markdown("""
İki kesikli rastgele değişken aynı örnek uzayda tanımlıysa, her `(x, y)` ikilisine verilen olasılık **ortak olasılık fonksiyonu**dur.
        """)
        st.latex(r"p(x,y)=P(X=x,Y=y)")
        st.latex(r"p(x,y)\ge 0,\qquad \sum_x\sum_y p(x,y)=1")
        st.dataframe(_joint_df(default, x_vals, y_vals).round(4), use_container_width=True)
        st.pyplot(_heatmap(default, x_vals, y_vals))

    elif choice == "6.2 Marjinal Dağılımlar":
        st.header("6.2 Marjinal Dağılımlar")
        st.markdown("Ortak dağılımdan sadece X veya sadece Y dağılımını elde etmeye **marjinal dağılım** denir.")
        st.latex(r"P(X=x)=\sum_y P(X=x,Y=y)")
        st.latex(r"P(Y=y)=\sum_x P(X=x,Y=y)")
        df = _joint_df(default, x_vals, y_vals)
        st.dataframe(df.round(4), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(pd.DataFrame({"P(X=x)": default.sum(axis=1)}, index=x_vals))
        with col2:
            st.bar_chart(pd.DataFrame({"P(Y=y)": default.sum(axis=0)}, index=y_vals))

    elif choice == "6.3 Koşullu Olasılık Fonksiyonu":
        st.header("6.3 Koşullu Olasılık Fonksiyonu")
        st.latex(r"P(X=x\mid Y=y)=\frac{P(X=x,Y=y)}{P(Y=y)}")
        y_choice = st.selectbox("Y değeri sabitlensin", y_vals)
        j = y_vals.index(y_choice)
        py = default[:, j].sum()
        cond = default[:, j] / py
        out = pd.DataFrame({"x": x_vals, f"P(X=x | Y={y_choice})": cond})
        st.dataframe(out.round(4), use_container_width=True)
        st.success(f"P(Y={y_choice}) = {py:.4f}")

    elif choice == "6.4 Bağımsızlık Kontrolü":
        st.header("6.4 Bağımsızlık Kontrolü")
        st.markdown("X ve Y bağımsızsa her hücre için şu eşitlik sağlanır:")
        st.latex(r"P(X=x,Y=y)=P(X=x)P(Y=y)")
        px = default.sum(axis=1)
        py = default.sum(axis=0)
        expected = np.outer(px, py)
        diff = default - expected
        st.write("Gerçek ortak olasılıklar")
        st.dataframe(pd.DataFrame(default, index=x_vals, columns=y_vals).round(4), use_container_width=True)
        st.write("Bağımsızlık varsayımı altında beklenen P(X=x)P(Y=y)")
        st.dataframe(pd.DataFrame(expected, index=x_vals, columns=y_vals).round(4), use_container_width=True)
        if np.allclose(default, expected, atol=1e-6):
            st.success("Bu tabloda X ve Y bağımsızdır.")
        else:
            st.warning("Bu tabloda X ve Y bağımsız değildir.")

    elif choice == "6.5 Koşullu Beklenen Değer ve Varyans":
        st.header("6.5 Koşullu Beklenen Değer ve Varyans")
        st.latex(r"E(X\mid Y=y)=\sum_x xP(X=x\mid Y=y)")
        st.latex(r"Var(X\mid Y=y)=E(X^2\mid Y=y)-[E(X\mid Y=y)]^2")
        y_choice = st.selectbox("Y değeri", y_vals, key="cond_ev_y")
        j = y_vals.index(y_choice)
        cond = default[:, j] / default[:, j].sum()
        x = np.array(x_vals, dtype=float)
        ev = float((x * cond).sum())
        ev2 = float(((x ** 2) * cond).sum())
        var = ev2 - ev ** 2
        c1, c2 = st.columns(2)
        c1.metric(f"E(X | Y={y_choice})", f"{ev:.4f}")
        c2.metric(f"Var(X | Y={y_choice})", f"{var:.4f}")

    elif choice == "6.6 Sürekli Ortak Yoğunluk":
        st.header("6.6 Sürekli Ortak Yoğunluk")
        st.markdown("Sürekli iki değişkende ortak yoğunluk fonksiyonu alan altında olasılık verir.")
        st.latex(r"f(x,y)\ge 0,\qquad \int\int f(x,y)\,dx\,dy=1")
        st.latex(r"f_X(x)=\int_{-\infty}^{\infty} f(x,y)\,dy,\qquad f_Y(y)=\int_{-\infty}^{\infty} f(x,y)\,dx")
        st.info("Örnek: 0<x<1 ve 0<y<1 için f(x,y)=x+y verilirse toplam integral 1 olur.")
        st.latex(r"\int_0^1\int_0^1 (x+y)\,dy\,dx=1")
        st.latex(r"f_X(x)=x+\frac12,\qquad f_Y(y)=y+\frac12")

    elif choice == "6.7 Kovaryans ve Korelasyon":
        st.header("6.7 Kovaryans ve Korelasyon")
        st.latex(r"Cov(X,Y)=E(XY)-E(X)E(Y)")
        st.latex(r"\rho_{XY}=\frac{Cov(X,Y)}{\sigma_X\sigma_Y}")
        ex, ey, vx, vy, cov, corr = _cov_corr(default, x_vals, y_vals)
        c1, c2, c3 = st.columns(3)
        c1.metric("Cov(X,Y)", f"{cov:.4f}")
        c2.metric("ρ", f"{corr:.4f}")
        c3.metric("Yorum", "Pozitif" if corr > 0 else "Negatif" if corr < 0 else "Yok")
        st.write(f"E(X)={ex:.4f}, E(Y)={ey:.4f}, Var(X)={vx:.4f}, Var(Y)={vy:.4f}")

    elif choice == "🧪 İki Değişkenli Laboratuvar":
        st.header("🧪 İki Değişkenli Olasılık Laboratuvarı")
        st.write("Kendi ortak olasılık tablonu gir. Değerler otomatik olarak toplamı 1 olacak şekilde normalize edilir.")
        rows = st.number_input("X değer sayısı", 2, 5, 3)
        cols = st.number_input("Y değer sayısı", 2, 5, 2)
        edited = st.data_editor(pd.DataFrame(np.ones((rows, cols)) / (rows * cols)), use_container_width=True, num_rows="fixed")
        try:
            pxy = _norm_table(edited.values, rows, cols)
            xs = list(range(rows)); ys = list(range(cols))
            st.dataframe(_joint_df(pxy, xs, ys).round(4), use_container_width=True)
            ex, ey, vx, vy, cov, corr = _cov_corr(pxy, xs, ys)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("E(X)", f"{ex:.3f}")
            c2.metric("E(Y)", f"{ey:.3f}")
            c3.metric("Cov", f"{cov:.3f}")
            c4.metric("ρ", f"{corr:.3f}")
            st.pyplot(_heatmap(pxy, xs, ys))
        except Exception as e:
            st.error(str(e))

    elif choice == "✅ Mini Quiz":
        st.header("✅ Mini Quiz")
        score = 0
        q1 = st.radio("1) Marjinal P(X=x) nasıl bulunur?", ["Y üzerinden toplanır", "X üzerinden toplanır", "Çarpılır"], key="b6q1")
        if q1 == "Y üzerinden toplanır": score += 1
        q2 = st.radio("2) Bağımsızlık şartı hangisidir?", ["P(x,y)=P(x)P(y)", "P(x,y)=P(x)+P(y)", "Cov her zaman 1"], key="b6q2")
        if q2 == "P(x,y)=P(x)P(y)": score += 1
        q3 = st.radio("3) Kovaryans sıfırsa ne söylenebilir?", ["Doğrusal ilişki yoktur", "Mutlaka bağımsızdır", "Olasılık toplamı sıfırdır"], key="b6q3")
        if q3 == "Doğrusal ilişki yoktur": score += 1
        if st.button("Bölüm 6 quiz sonucunu hesapla"):
            st.success(f"Puan: {score}/3")
