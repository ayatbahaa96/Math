# bolum2.py

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Genel yardımcılar
# -----------------------------
def parse_numeric_input(text: str):
    cleaned = text.replace("\n", ",").replace(";", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        raise ValueError("Lütfen en az bir sayı girin.")
    try:
        return np.array([float(p) for p in parts], dtype=float)
    except ValueError as exc:
        raise ValueError("Bu alan yalnızca sayısal veriler içermelidir.") from exc


def parse_grouped_discrete_input(text: str):
    """
    Her satır: değer:frekans
    Örn:
    1:6
    2:12
    3:14
    """
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    if not rows:
        raise ValueError("Lütfen en az bir satır girin.")

    x_vals, freqs = [], []
    for row in rows:
        if ":" not in row:
            raise ValueError("Kesikli gruplanmış veri için her satır 'değer:frekans' biçiminde olmalıdır.")
        x_str, f_str = row.split(":", 1)
        x = float(x_str.strip())
        f = float(f_str.strip())
        if f < 0 or int(f) != f:
            raise ValueError("Frekanslar negatif olmayan tam sayılar olmalıdır.")
        x_vals.append(x)
        freqs.append(int(f))

    if sum(freqs) == 0:
        raise ValueError("Toplam frekans sıfır olamaz.")

    return np.array(x_vals, dtype=float), np.array(freqs, dtype=int)


def parse_grouped_interval_input(text: str):
    """
    Her satır: alt-üst:frekans
    Örn:
    0-20:5
    20-40:10
    40-60:15
    """
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    if not rows:
        raise ValueError("Lütfen en az bir satır girin.")

    lowers, uppers, freqs = [], [], []

    for row in rows:
        if ":" not in row:
            raise ValueError("Sınıf aralıklı veri için her satır 'alt-üst:frekans' biçiminde olmalıdır.")
        interval_str, freq_str = row.split(":", 1)

        if "-" not in interval_str:
            raise ValueError("Sınıf aralığı 'alt-üst' biçiminde olmalıdır.")

        low_str, high_str = interval_str.split("-", 1)
        low = float(low_str.strip())
        high = float(high_str.strip())
        f = float(freq_str.strip())

        if high <= low:
            raise ValueError("Üst sınır alt sınırdan büyük olmalıdır.")
        if f < 0 or int(f) != f:
            raise ValueError("Frekanslar negatif olmayan tam sayılar olmalıdır.")

        lowers.append(low)
        uppers.append(high)
        freqs.append(int(f))

    lower = np.array(lowers, dtype=float)
    upper = np.array(uppers, dtype=float)
    freq = np.array(freqs, dtype=int)

    mids = (lower + upper) / 2
    widths = upper - lower

    if np.sum(freq) == 0:
        raise ValueError("Toplam frekans sıfır olamaz.")

    return lower, upper, mids, freq, widths


def expand_discrete_grouped_data(x_vals, freqs):
    return np.repeat(x_vals, freqs)


def grouped_interval_table(lower, upper, mids, freqs):
    cumulative = np.cumsum(freqs)
    rel = freqs / np.sum(freqs)
    return pd.DataFrame(
        {
            "Sınıf": [f"{l:.2f}-{u:.2f}" for l, u in zip(lower, upper)],
            "Alt Sınır": np.round(lower, 2),
            "Üst Sınır": np.round(upper, 2),
            "Sınıf Orta Değeri": np.round(mids, 2),
            "Frekans": freqs,
            "Birikimli Frekans": cumulative,
            "Oransal Frekans": np.round(rel, 4),
            "Yüzde": np.round(rel * 100, 2),
        }
    )


def grouped_discrete_table(x_vals, freqs):
    cumulative = np.cumsum(freqs)
    rel = freqs / np.sum(freqs)
    return pd.DataFrame(
        {
            "Değer": x_vals,
            "Frekans": freqs,
            "Birikimli Frekans": cumulative,
            "Oransal Frekans": np.round(rel, 4),
            "Yüzde": np.round(rel * 100, 2),
        }
    )


def show_group_note():
    st.info("Not: Gruplanmış verilerde bazı sonuçlar sınıf orta değeri ve sınıf içi düzgün dağılım varsayımı altında yaklaşık hesaplanır.")


def data_structure_selector(key: str):
    return st.radio(
        "Veri yapısı",
        ["Gruplanmamış Veri", "Gruplanmış Veri"],
        horizontal=True,
        key=key,
    )


def grouped_data_type_selector(key: str):
    return st.radio(
        "Gruplanmış veri türü",
        ["Kesikli Gruplanmış Veri", "Sınıf Aralıklı Veri"],
        horizontal=True,
        key=key,
    )


# -----------------------------
# Grafik yardımcıları
# -----------------------------
def plot_boxplot(data, title="Kutu Grafiği"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, vert=False)
    ax.set_title(title)
    ax.set_xlabel("Değer")
    plt.tight_layout()
    return fig


def plot_hist(data, bins, title="Histogram"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Değer")
    ax.set_ylabel("Frekans")
    plt.tight_layout()
    return fig


def plot_grouped_hist(lower, upper, freqs, title="Gruplanmış Veri Histogramı"):
    fig, ax = plt.subplots(figsize=(8, 4))
    widths = upper - lower
    ax.bar(lower, freqs, width=widths, align="edge", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Sınıf Aralıkları")
    ax.set_ylabel("Frekans")
    plt.tight_layout()
    return fig


def plot_two_boxplots(data1, data2, labels=("Grup 1", "Grup 2"), title="Karşılaştırmalı Kutu Grafiği"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([data1, data2], labels=list(labels))
    ax.set_title(title)
    ax.set_ylabel("Değer")
    plt.tight_layout()
    return fig


# -----------------------------
# Gruplanmamış veri hesapları
# -----------------------------
def arithmetic_mean(data: np.ndarray):
    return float(np.mean(data))


def median_value(data: np.ndarray):
    return float(np.median(data))


def mode_value(data: np.ndarray):
    return pd.Series(data).mode().tolist()


def geometric_mean_value(data: np.ndarray):
    if np.any(data <= 0):
        raise ValueError("Geometrik ortalama için tüm değerler pozitif olmalıdır.")
    return float(np.exp(np.mean(np.log(data))))


def harmonic_mean_value(data: np.ndarray):
    if np.any(data <= 0):
        raise ValueError("Harmonik ortalama için tüm değerler pozitif olmalıdır.")
    return float(len(data) / np.sum(1 / data))


def range_value(data: np.ndarray):
    return float(np.max(data) - np.min(data))


def quartiles_value(data: np.ndarray):
    q1 = float(np.percentile(data, 25))
    q2 = float(np.percentile(data, 50))
    q3 = float(np.percentile(data, 75))
    iqr = float(q3 - q1)
    semi_iqr = float(iqr / 2)
    return q1, q2, q3, iqr, semi_iqr


def quartiles_by_median_halves(data: np.ndarray):
    data = np.sort(np.array(data, dtype=float))
    n = len(data)
    q2 = float(np.median(data))

    if n % 2 == 1:
        lower_half = data[: n // 2]
        upper_half = data[n // 2 + 1 :]
    else:
        lower_half = data[: n // 2]
        upper_half = data[n // 2 :]

    q1 = float(np.median(lower_half))
    q3 = float(np.median(upper_half))
    q = float((q3 - q1) / 2)
    return q1, q2, q3, q


def mean_absolute_deviation_value(data: np.ndarray):
    mean = np.mean(data)
    return float(np.mean(np.abs(data - mean)))


def variance_std_value(data: np.ndarray):
    return {
        "variance_pop": float(np.var(data, ddof=0)),
        "variance_sample": float(np.var(data, ddof=1)) if len(data) > 1 else 0.0,
        "std_pop": float(np.std(data, ddof=0)),
        "std_sample": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
    }


def coefficient_of_variation_value(data: np.ndarray, use_sample=False):
    mean = np.mean(data)
    std = np.std(data, ddof=1 if use_sample and len(data) > 1 else 0)
    if mean == 0:
        raise ValueError("Değişim katsayısı için ortalama sıfır olmamalıdır.")
    return float(std / mean)


def pearson_skewness_mode(mean: float, mode: float, std: float):
    if std == 0:
        return 0.0
    return float((mean - mode) / std)


def pearson_skewness_median(mean: float, median: float, std: float):
    if std == 0:
        return 0.0
    return float(3 * (mean - median) / std)


def bowley_skewness_value(q1: float, q2: float, q3: float):
    denom = q3 - q1
    if denom == 0:
        return 0.0
    return float((q3 - 2 * q2 + q1) / denom)


def skewness_comment(value: float):
    if value > 0:
        return "Dağılım sağa çarpıktır."
    if value < 0:
        return "Dağılım sola çarpıktır."
    return "Dağılım simetriktir."


# -----------------------------
# Gruplanmış veri hesapları
# -----------------------------
def grouped_mean(x_vals, freqs):
    return float(np.sum(x_vals * freqs) / np.sum(freqs))


def grouped_median_interval(lower, upper, freqs):
    N = np.sum(freqs)
    cumulative = np.cumsum(freqs)
    idx = int(np.searchsorted(cumulative, N / 2, side="left"))
    L = lower[idx]
    c = upper[idx] - lower[idx]
    f = freqs[idx]
    d = cumulative[idx - 1] if idx > 0 else 0
    med = float(L + (c / f) * ((N / 2) - d))
    return med, idx, L, c, f, d, cumulative


def grouped_mode_interval(lower, upper, freqs):
    idx = int(np.argmax(freqs))
    if idx == 0 or idx == len(freqs) - 1:
        raise ValueError("PDF kuralına göre ilk veya son sınıf tepe değer sınıfı ise bu mod formülü kullanılmaz.")

    LA = lower[idx]
    LO = upper[idx]
    c = upper[idx] - lower[idx]
    f_prev = freqs[idx - 1]
    f_next = freqs[idx + 1]
    f_modal = freqs[idx]

    mode_formula1 = float(LA + (f_next / (f_prev + f_next)) * c)

    delta1 = f_modal - f_prev
    delta2 = f_modal - f_next
    mode_formula2a = float(LA + (delta1 / (delta1 + delta2)) * c)
    mode_formula2b = float(LO - (delta2 / (delta1 + delta2)) * c)

    return {
        "index": idx,
        "LA": LA,
        "LO": LO,
        "c": c,
        "f_prev": int(f_prev),
        "f_next": int(f_next),
        "f_modal": int(f_modal),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "mode_formula1": mode_formula1,
        "mode_formula2a": mode_formula2a,
        "mode_formula2b": mode_formula2b,
    }


def grouped_geometric_mean(x_vals, freqs):
    if np.any(x_vals <= 0):
        raise ValueError("Geometrik ortalama için sınıf orta değerleri pozitif olmalıdır.")
    N = np.sum(freqs)
    return float(np.exp(np.sum(freqs * np.log(x_vals)) / N))


def grouped_harmonic_mean(x_vals, freqs):
    if np.any(x_vals <= 0):
        raise ValueError("Harmonik ortalama için sınıf orta değerleri pozitif olmalıdır.")
    N = np.sum(freqs)
    return float(N / np.sum(freqs / x_vals))


def grouped_range_interval(lower, upper):
    return float(np.max(upper) - np.min(lower))


def grouped_quartile_interval(lower, upper, freqs, which=1):
    N = np.sum(freqs)
    target = N / 4 if which == 1 else 3 * N / 4
    cumulative = np.cumsum(freqs)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    L = lower[idx]
    c = upper[idx] - lower[idx]
    f = freqs[idx]
    d = cumulative[idx - 1] if idx > 0 else 0
    q_value = float(L + (c / f) * (target - d))
    return q_value, idx, L, c, f, d


def grouped_mean_absolute_deviation(x_vals, freqs):
    mean_val = grouped_mean(x_vals, freqs)
    N = np.sum(freqs)
    mad = float(np.sum(np.abs(x_vals - mean_val) * freqs) / N)
    return mad, mean_val


def grouped_variance_std(x_vals, freqs, sample=False):
    N = np.sum(freqs)
    mean_val = grouped_mean(x_vals, freqs)
    ss = np.sum(((x_vals - mean_val) ** 2) * freqs)
    denom = (N - 1) if sample and N > 1 else N
    var_val = float(ss / denom)
    std_val = float(np.sqrt(var_val))
    return {
        "mean": mean_val,
        "ss": float(ss),
        "variance": var_val,
        "std": std_val,
        "N": int(N),
    }


def grouped_variance_shortcut(x_vals, freqs, sample=False):
    N = np.sum(freqs)
    mean_val = grouped_mean(x_vals, freqs)
    mean_sq = float(np.sum((x_vals ** 2) * freqs) / N)
    var_pop = float(mean_sq - mean_val**2)
    if sample and N > 1:
        ss = np.sum(((x_vals - mean_val) ** 2) * freqs)
        var_val = float(ss / (N - 1))
    else:
        var_val = var_pop
    return {
        "mean_sq": mean_sq,
        "mean_val_sq": float(mean_val**2),
        "variance": var_val,
        "variance_pop": var_pop,
    }


def grouped_cv(x_vals, freqs, sample=False):
    vals = grouped_variance_std(x_vals, freqs, sample=sample)
    if vals["mean"] == 0:
        raise ValueError("Değişim katsayısı için ortalama sıfır olmamalıdır.")
    return float(vals["std"] / vals["mean"]), vals


# -----------------------------
# Sayfa içerikleri
# -----------------------------
def render_intro_page():
    c1, c2, c3 = st.columns(3)
    c1.metric("Modül Türü", "Web Uygulaması")
    c2.metric("İçerik Katmanı", "14")
    c3.metric("Ana Hedef", "Ölç + Yorumla")

    st.markdown(
        """
### Bu modülde neler var?
Bu bölüm, veriyi yalnızca tabloya dönüştürmekle kalmaz; verinin **merkezini**, **yayılımını** ve **şeklini** de yorumlamayı öğretir.

#### İçerik bileşenleri
- Aritmetik ortalama, medyan ve mod
- Geometrik ve harmonik ortalama
- Açıklık, çeyrekler ve çeyrek ayrılış
- Kutu grafiği
- Ortalama sapma
- Varyans ve standart sapma
- Değişim katsayısı
- Çarpıklık ölçüleri
- Çözümlü sorular
- Mini quiz
- Öğrenci veri laboratuvarı

#### PDF'ye göre temel ayrım
- **Gruplanmamış veri:** gözlemler doğrudan verilir.
- **Gruplanmış veri:** değerler frekanslarla ya da sınıf aralıklarıyla verilir.
        """
    )


def render_mean_page():
    st.header("2. Aritmetik Ortalama")
    st.markdown(
        """
Aritmetik ortalama, birimlerin belli bir özelliğe ilişkin değerlerinin cebirsel toplamının birim sayısına bölünmesiyle elde edilir.

### Gruplanmamış Veri Kuralları
- Yığın için: **μ = Σxᵢ / N**
- Örnek için: **x̄ = Σxᵢ / n**

### Gruplanmış Veri Kuralları
- Frekanslı veri için: **x̄ = Σ(xⱼ·fⱼ) / n**
- Sınıf aralıklı veride **xⱼ**, sınıf orta değeridir.
- PDF’de ayrıca kodlanmış değerlerle ortalama hesabı da gösterilir.
        """
    )

    mode = data_structure_selector("b2_mean_mode")

    if mode == "Gruplanmamış Veri":
        raw = st.text_area("Sayıları girin", "20,35,42,36,50,27", height=100, key="b2_mean_raw")
        try:
            data = parse_numeric_input(raw)
            mean_val = arithmetic_mean(data)
            st.success(f"Aritmetik Ortalama = {mean_val:.4f}")
            st.write(f"Toplam = {np.sum(data):.2f}")
            st.write(f"Gözlem sayısı = {len(data)}")
        except ValueError as err:
            st.error(str(err))
    else:
        grouped_type = grouped_data_type_selector("b2_mean_group_type")
        if grouped_type == "Kesikli Gruplanmış Veri":
            raw = st.text_area(
                "Her satır: değer:frekans",
                "1:6\n2:12\n3:14\n4:9\n5:6",
                height=140,
                key="b2_mean_group_discrete",
            )
            try:
                x_vals, freqs = parse_grouped_discrete_input(raw)
                mean_val = grouped_mean(x_vals, freqs)
                df = grouped_discrete_table(x_vals, freqs)
                df["xⱼ·fⱼ"] = np.round(x_vals * freqs, 4)
                st.dataframe(df, use_container_width=True)
                st.success(f"Aritmetik Ortalama = {mean_val:.4f}")
            except ValueError as err:
                st.error(str(err))
        else:
            raw = st.text_area(
                "Her satır: alt-üst:frekans",
                "0-20:5\n20-40:10\n40-60:20\n60-80:14\n80-100:6",
                height=140,
                key="b2_mean_group_interval",
            )
            try:
                lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
                mean_val = grouped_mean(mids, freqs)
                df = grouped_interval_table(lower, upper, mids, freqs)
                df["xⱼ·fⱼ"] = np.round(mids * freqs, 4)
                st.dataframe(df, use_container_width=True)
                st.success(f"Aritmetik Ortalama = {mean_val:.4f}")
                show_group_note()
            except ValueError as err:
                st.error(str(err))

    st.markdown("### Ayrı Örnekler")
    st.code("Gruplanmamış örnek: 20, 35, 42, 36, 50, 27")
    st.code("Gruplanmış örnek: 0-20:5 | 20-40:10 | 40-60:20 | 60-80:14 | 80-100:6")


def render_median_page():
    st.header("3. Ortanca (Medyan)")
    st.markdown(
        """
Ortanca, sıralanmış veri dizisini iki eşit parçaya ayıran değerdir.

### Gruplanmamış Veri Kuralları
- Gözlem sayısı tek ise ortadaki değer alınır.
- Gözlem sayısı çift ise ortadaki iki değerin aritmetik ortalaması alınır.

### Gruplanmış Veri Kuralları
- Önce **ortanca sınıfı** bulunur.
- Formül: **Ortanca = L + (c / f) · (N/2 - d)**
- Burada:
  - **L**: ortanca sınıfının alt sınırı
  - **c**: sınıf genişliği
  - **f**: ortanca sınıfının frekansı
  - **d**: önceki sınıfa kadar birikimli frekans
        """
    )

    mode = data_structure_selector("b2_median_mode")

    if mode == "Gruplanmamış Veri":
        raw = st.text_area("Sayıları girin", "32,58,27,44,36,28,42,51,39,50", height=100, key="b2_median_raw")
        try:
            data = parse_numeric_input(raw)
            sorted_data = np.sort(data)
            med = median_value(data)
            st.write("Sıralanmış veri:", sorted_data.tolist())
            st.success(f"Ortanca = {med:.4f}")
        except ValueError as err:
            st.error(str(err))
    else:
        grouped_type = grouped_data_type_selector("b2_median_group_type")
        if grouped_type == "Kesikli Gruplanmış Veri":
            raw = st.text_area(
                "Her satır: değer:frekans",
                "7:1\n8:2\n9:3\n10:7\n11:12\n12:16\n13:15\n14:13\n15:9",
                height=180,
                key="b2_median_group_discrete",
            )
            try:
                x_vals, freqs = parse_grouped_discrete_input(raw)
                expanded = expand_discrete_grouped_data(x_vals, freqs)
                med = median_value(expanded)
                st.dataframe(grouped_discrete_table(x_vals, freqs), use_container_width=True)
                st.success(f"Ortanca = {med:.4f}")
                st.caption("Kesikli gruplanmış veri, frekanslara göre açılarak hesaplandı.")
            except ValueError as err:
                st.error(str(err))
        else:
            raw = st.text_area(
                "Her satır: alt-üst:frekans",
                "170-206:30\n206-242:83\n242-278:68\n278-314:35\n314-350:34",
                height=160,
                key="b2_median_group_interval",
            )
            try:
                lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
                med, idx, L, c, f, d, cumulative = grouped_median_interval(lower, upper, freqs)
                df = grouped_interval_table(lower, upper, mids, freqs)
                st.dataframe(df, use_container_width=True)
                st.success(f"Ortanca = {med:.4f}")
                st.write(f"Ortanca sınıfı: {lower[idx]:.2f}-{upper[idx]:.2f}")
                st.write(f"L = {L:.2f}, c = {c:.2f}, f = {f}, d = {d}, N = {freqs.sum()}")
                show_group_note()
            except ValueError as err:
                st.error(str(err))

    st.markdown("### Ayrı Örnekler")
    st.code("Gruplanmamış örnek: 32,58,27,44,36,28,42,51,39,50")
    st.code("Gruplanmış örnek: 170-206:30 | 206-242:83 | 242-278:68 | 278-314:35 | 314-350:34")


def render_mode_page():
    st.header("4. Tepedeğer (Mod)")
    st.markdown(
        """
Tepedeğer, veriler içinde en çok tekrar eden değerdir.

### Gruplanmamış Veri Kuralları
- En çok tekrar eden değer moddur.

### Gruplanmış Veri Kuralları
- Önce **tepe değer sınıfı** belirlenir.
- PDF’de iki ayrı formül verilir:
  1. **Mod = L_A + (f₂ / (f₁ + f₂)) · c**
  2. **Mod = L_A + (Δ₁ / (Δ₁ + Δ₂)) · c**
     veya eşdeğeri: **Mod = L_O - (Δ₂ / (Δ₁ + Δ₂)) · c**
- İlk veya son sınıf tepe değer sınıfı ise bu formül kullanılmaz.
        """
    )

    mode = data_structure_selector("b2_mode_mode")

    if mode == "Gruplanmamış Veri":
        raw = st.text_area("Sayıları girin", "14,15,15,16,17,17,17,17,18,19", height=100, key="b2_mode_raw")
        try:
            data = parse_numeric_input(raw)
            modes = mode_value(data)
            counts = Counter(data)
            freq_df = pd.DataFrame({"Değer": list(counts.keys()), "Frekans": list(counts.values())}).sort_values("Değer")
            st.dataframe(freq_df, use_container_width=True)
            st.success("Mod(lar): " + ", ".join(f"{m:.2f}" for m in modes))
        except ValueError as err:
            st.error(str(err))
    else:
        grouped_type = grouped_data_type_selector("b2_mode_group_type")
        if grouped_type == "Kesikli Gruplanmış Veri":
            raw = st.text_area(
                "Her satır: değer:frekans",
                "0:80\n1:270\n2:421\n3:380\n4:311\n5:176\n6:95\n7:31\n8:26",
                height=180,
                key="b2_mode_group_discrete",
            )
            try:
                x_vals, freqs = parse_grouped_discrete_input(raw)
                max_f = freqs.max()
                modes = x_vals[freqs == max_f]
                st.dataframe(grouped_discrete_table(x_vals, freqs), use_container_width=True)
                st.success("Mod(lar): " + ", ".join(f"{m:.2f}" for m in modes))
            except ValueError as err:
                st.error(str(err))
        else:
            raw = st.text_area(
                "Her satır: alt-üst:frekans",
                "140-150:5\n150-160:100\n160-170:250\n170-180:60\n180-190:10",
                height=160,
                key="b2_mode_group_interval",
            )
            try:
                lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
                vals = grouped_mode_interval(lower, upper, freqs)
                st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)
                st.success(f"Formül 1 sonucu = {vals['mode_formula1']:.4f}")
                st.success(f"Formül 2 sonucu = {vals['mode_formula2a']:.4f}")
                st.write(f"Üst sınırla eşdeğer yazım = {vals['mode_formula2b']:.4f}")
                st.write(f"Tepe değer sınıfı: {lower[vals['index']]:.2f}-{upper[vals['index']]:.2f}")
                show_group_note()
            except ValueError as err:
                st.error(str(err))


def render_geometric_page():
    st.header("5. Geometrik Ortalama")
    st.markdown(
        """
Geometrik ortalama, verilerin çarpımının derece sayısına göre köküdür.

### Gruplanmamış Veri Kuralları
- **G = (x₁x₂...xₙ)^(1/n)**
- Logaritma ile de hesaplanabilir.
- Tüm değerler pozitif olmalıdır.

### Gruplanmış Veri Kuralları
- Sınıf orta değerleri kullanılır.
- **log G = (1/N) Σ(fⱼ log xⱼ)**
- Kesikli gruplamada değerler doğrudan, sınıf aralıklı veride sınıf orta değerleri kullanılır.
        """
    )

    mode = data_structure_selector("b2_geo_mode")

    if mode == "Gruplanmamış Veri":
        raw = st.text_area("Pozitif sayıları girin", "3,6,9,12,15", height=100, key="b2_geo_raw")
        try:
            data = parse_numeric_input(raw)
            g = geometric_mean_value(data)
            st.success(f"Geometrik Ortalama = {g:.4f}")
        except ValueError as err:
            st.error(str(err))
    else:
        grouped_type = grouped_data_type_selector("b2_geo_group_type")
        if grouped_type == "Kesikli Gruplanmış Veri":
            raw = st.text_area(
                "Her satır: değer:frekans",
                "2:18\n5:12\n8:4\n11:2",
                height=140,
                key="b2_geo_group_discrete",
            )
            try:
                x_vals, freqs = parse_grouped_discrete_input(raw)
                g = grouped_geometric_mean(x_vals, freqs)
                st.dataframe(grouped_discrete_table(x_vals, freqs), use_container_width=True)
                st.success(f"Geometrik Ortalama = {g:.4f}")
            except ValueError as err:
                st.error(str(err))
        else:
            raw = st.text_area(
                "Her satır: alt-üst:frekans",
                "1-3:18\n4-6:12\n7-9:4\n10-12:2",
                height=140,
                key="b2_geo_group_interval",
            )
            try:
                lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
                g = grouped_geometric_mean(mids, freqs)
                st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)
                st.success(f"Geometrik Ortalama = {g:.4f}")
                show_group_note()
            except ValueError as err:
                st.error(str(err))


def render_harmonic_page():
    st.header("6. Harmonik Ortalama")
    st.markdown(
        """
Harmonik ortalama, özellikle hız, oran, birim başına zaman ve birim fiyat problemlerinde kullanılır.

### Gruplanmamış Veri Kuralları
- **H = n / Σ(1/xᵢ)**

### Gruplanmış Veri Kuralları
- **H = N / Σ(fⱼ / xⱼ)**
- Sınıf aralıklı veride **xⱼ** sınıf orta değeridir.
- Tüm değerler pozitif olmalıdır.
        """
    )

    mode = data_structure_selector("b2_harm_mode")

    if mode == "Gruplanmamış Veri":
        raw = st.text_area("Pozitif sayıları girin", "10,1,100,20,40", height=100, key="b2_harm_raw")
        try:
            data = parse_numeric_input(raw)
            h = harmonic_mean_value(data)
            st.success(f"Harmonik Ortalama = {h:.4f}")
        except ValueError as err:
            st.error(str(err))
    else:
        grouped_type = grouped_data_type_selector("b2_harm_group_type")
        if grouped_type == "Kesikli Gruplanmış Veri":
            raw = st.text_area(
                "Her satır: değer:frekans",
                "0.50:1\n0.25:1",
                height=120,
                key="b2_harm_group_discrete",
            )
            try:
                x_vals, freqs = parse_grouped_discrete_input(raw)
                h = grouped_harmonic_mean(x_vals, freqs)
                st.dataframe(grouped_discrete_table(x_vals, freqs), use_container_width=True)
                st.success(f"Harmonik Ortalama = {h:.4f}")
            except ValueError as err:
                st.error(str(err))
        else:
            raw = st.text_area(
                "Her satır: alt-üst:frekans",
                "10-20:4\n20-30:8\n30-40:12\n40-50:6",
                height=140,
                key="b2_harm_group_interval",
            )
            try:
                lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
                h = grouped_harmonic_mean(mids, freqs)
                st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)
                st.success(f"Harmonik Ortalama = {h:.4f}")
                show_group_note()
            except ValueError as err:
                st.error(str(err))


def render_range_quartile_page():
    st.header("7. Açıklık ve Çeyrek Ayrılış")
    st.markdown(
        """
### Açıklık Kuralları
- Gruplanmamış veri: **R = en büyük değer - en küçük değer**
- Gruplanmış veri: **son sınıfın üst sınırı - ilk sınıfın alt sınırı**

### Çeyrek Ayrılış Kuralları
- **Q = (Q3 - Q1) / 2**
- Gruplanmamış veride önce ortanca bulunur, sonra alt ve üst yarının ortancaları alınır.
- Gruplanmış veride:
  - **Q1 = X(N/4)**
  - **Q3 = X(3N/4)**
  - Sonra sınıf içi doğrusal artış varsayımıyla:
    - **Q1 = L + (c/f)(N/4 - d)**
    - **Q3 = L + (c/f)(3N/4 - d)**
        """
    )

    mode = data_structure_selector("b2_rq_mode")

    if mode == "Gruplanmamış Veri":
        raw = st.text_area("Sayıları girin", "50,75,90,110,125,140,142", height=100, key="b2_rq_raw")
        try:
            data = parse_numeric_input(raw)
            q1, q2, q3, q = quartiles_by_median_halves(data)
            c1, c2 = st.columns(2)
            c1.metric("Açıklık", f"{range_value(data):.4f}")
            c2.metric("Çeyrek Ayrılış", f"{q:.4f}")

            st.dataframe(
                pd.DataFrame(
                    {
                        "Ölçü": ["Q1", "Q2 (Ortanca)", "Q3", "Çeyrek Ayrılış"],
                        "Değer": [q1, q2, q3, q],
                    }
                ),
                use_container_width=True,
            )
        except ValueError as err:
            st.error(str(err))
    else:
        grouped_type = grouped_data_type_selector("b2_rq_group_type")
        if grouped_type == "Kesikli Gruplanmış Veri":
            raw = st.text_area(
                "Her satır: değer:frekans",
                "0:18\n1:27\n2:27\n3:18\n4:15\n5:5",
                height=140,
                key="b2_rq_group_discrete",
            )
            try:
                x_vals, freqs = parse_grouped_discrete_input(raw)
                expanded = expand_discrete_grouped_data(x_vals, freqs)
                q1, q2, q3, q = quartiles_by_median_halves(expanded)
                c1, c2 = st.columns(2)
                c1.metric("Açıklık", f"{range_value(expanded):.4f}")
                c2.metric("Çeyrek Ayrılış", f"{q:.4f}")
                st.dataframe(grouped_discrete_table(x_vals, freqs), use_container_width=True)
            except ValueError as err:
                st.error(str(err))
        else:
            raw = st.text_area(
                "Her satır: alt-üst:frekans",
                "0-20:5\n20-40:9\n40-60:20\n60-80:29\n80-100:13",
                height=160,
                key="b2_rq_group_interval",
            )
            try:
                lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
                q1, idx1, L1, c1v, f1, d1 = grouped_quartile_interval(lower, upper, freqs, which=1)
                q3, idx3, L3, c3v, f3, d3 = grouped_quartile_interval(lower, upper, freqs, which=3)
                q = (q3 - q1) / 2

                c1, c2 = st.columns(2)
                c1.metric("Açıklık", f"{grouped_range_interval(lower, upper):.4f}")
                c2.metric("Çeyrek Ayrılış", f"{q:.4f}")

                st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)
                st.write(f"Q1 sınıfı: {lower[idx1]:.2f}-{upper[idx1]:.2f}")
                st.write(f"Q3 sınıfı: {lower[idx3]:.2f}-{upper[idx3]:.2f}")
                st.write(f"Q1 = {q1:.4f}, Q3 = {q3:.4f}")
                show_group_note()
            except ValueError as err:
                st.error(str(err))


def render_boxplot_page():
    st.header("8. Kutu Grafiği")
    st.markdown(
        """
Kutu grafiği, nicel verinin merkezini, yayılımını, dağılımını ve aykırı değerlerini göstermeye yarar.

### Kutu grafiği için gerekli 5 sayı
- En küçük değer
- Q1
- Ortanca (Q2)
- Q3
- En büyük değer

### Aykırı değer kuralları
- **x < Q1 - 3(Q3-Q1)** veya **x > Q3 + 3(Q3-Q1)** ise aykırı değer
- **Q1 - 3IQR < x < Q1 - 1.5IQR** veya **Q3 + 1.5IQR < x < Q3 + 3IQR** ise hafif aykırı değer
        """
    )

    compare = st.checkbox("PDF’deki gibi iki grubu karşılaştır", value=True, key="b2_box_compare")

    if compare:
        raw1 = st.text_area(
            "Kadın verisi",
            "100,120,80,95,110,235,36,75,90,98,120,30,48,60,65,80,115,240,50,70",
            key="b2_box_women",
        )
        raw2 = st.text_area(
            "Erkek verisi",
            "30,45,60,20,10,30,65,68,90,18,35,25,83,65,70,130,90,90,46,80",
            key="b2_box_men",
        )
        try:
            d1 = parse_numeric_input(raw1)
            d2 = parse_numeric_input(raw2)
            q1a, q2a, q3a, iqra, _ = quartiles_value(d1)
            q1b, q2b, q3b, iqrb, _ = quartiles_value(d2)

            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Ölçü": ["Min", "Q1", "Medyan", "Q3", "Max", "IQR"],
                            "Değer": [np.min(d1), q1a, q2a, q3a, np.max(d1), iqra],
                        }
                    ),
                    use_container_width=True,
                )
            with c2:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Ölçü": ["Min", "Q1", "Medyan", "Q3", "Max", "IQR"],
                            "Değer": [np.min(d2), q1b, q2b, q3b, np.max(d2), iqrb],
                        }
                    ),
                    use_container_width=True,
                )

            st.pyplot(plot_two_boxplots(d1, d2, labels=("Kadın", "Erkek"), title="Karşılaştırmalı Kutu Grafiği"))
        except ValueError as err:
            st.error(str(err))
    else:
        raw = st.text_area(
            "Sayıları girin",
            "100,120,80,95,110,235,36,75,90,98,120,30,48,60,65,80,115,240,50,70",
            height=120,
            key="b2_box_single",
        )
        try:
            data = parse_numeric_input(raw)
            q1, q2, q3, iqr, _ = quartiles_value(data)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Ölçü": ["Min", "Q1", "Medyan", "Q3", "Max", "IQR"],
                        "Değer": [np.min(data), q1, q2, q3, np.max(data), iqr],
                    }
                ),
                use_container_width=True,
            )
            st.pyplot(plot_boxplot(data))
        except ValueError as err:
            st.error(str(err))


def render_mad_page():
    st.header("9. Ortalama Sapma")
    st.markdown(
        """
Ortalama sapma, verilerin aritmetik ortalamadan mutlak farklarının ortalamasıdır.

### Gruplanmamış Veri Kuralları
- **OS = Σ|xᵢ - x̄| / n**

### Gruplanmış Veri Kuralları
- **OS = Σ|xⱼ - x̄| fⱼ / n**
- Burada **xⱼ**, kesikli veride değer; sınıf aralıklı veride sınıf orta değeridir.
- PDF’de ayrıca ortancadan sapma olarak da hesaplanabileceği belirtilir.
        """
    )

    mode = data_structure_selector("b2_mad_mode")

    if mode == "Gruplanmamış Veri":
        raw = st.text_area("Sayıları girin", "5,8,12,16,17,21,29,36", height=100, key="b2_mad_raw")
        try:
            data = parse_numeric_input(raw)
            mean_val = arithmetic_mean(data)
            abs_dev = np.abs(data - mean_val)
            mad = mean_absolute_deviation_value(data)
            st.dataframe(pd.DataFrame({"xᵢ": data, "|xᵢ-x̄|": np.round(abs_dev, 4)}), use_container_width=True)
            st.success(f"Ortalama Sapma = {mad:.4f}")
        except ValueError as err:
            st.error(str(err))
    else:
        grouped_type = grouped_data_type_selector("b2_mad_group_type")
        if grouped_type == "Kesikli Gruplanmış Veri":
            raw = st.text_area(
                "Her satır: değer:frekans",
                "15:2\n25:8\n35:14\n45:10\n55:6",
                height=140,
                key="b2_mad_group_discrete",
            )
            try:
                x_vals, freqs = parse_grouped_discrete_input(raw)
                mad, mean_val = grouped_mean_absolute_deviation(x_vals, freqs)
                df = grouped_discrete_table(x_vals, freqs)
                df["|xⱼ-x̄|"] = np.round(np.abs(x_vals - mean_val), 4)
                df["|xⱼ-x̄|·fⱼ"] = np.round(np.abs(x_vals - mean_val) * freqs, 4)
                st.dataframe(df, use_container_width=True)
                st.success(f"Ortalama Sapma = {mad:.4f}")
            except ValueError as err:
                st.error(str(err))
        else:
            raw = st.text_area(
                "Her satır: alt-üst:frekans",
                "10-20:2\n20-30:8\n30-40:14\n40-50:10\n50-60:6",
                height=140,
                key="b2_mad_group_interval",
            )
            try:
                lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
                mad, mean_val = grouped_mean_absolute_deviation(mids, freqs)
                df = grouped_interval_table(lower, upper, mids, freqs)
                df["|xⱼ-x̄|"] = np.round(np.abs(mids - mean_val), 4)
                df["|xⱼ-x̄|·fⱼ"] = np.round(np.abs(mids - mean_val) * freqs, 4)
                st.dataframe(df, use_container_width=True)
                st.success(f"Ortalama Sapma = {mad:.4f}")
                show_group_note()
            except ValueError as err:
                st.error(str(err))


def render_variance_page():
    st.header("10. Varyans ve Standart Sapma")
    st.markdown(
        """
Varyans, değerlerin ortalamadan sapmalarının karelerinin ortalamasıdır. Standart sapma ise varyansın pozitif kareköküdür.

### Gruplanmamış Veri Kuralları
- Yığın: **σ² = Σ(xᵢ-μ)² / N**
- Örnek: **s² = Σ(xᵢ-x̄)² / (n-1)**
- **Standart sapma = √varyans**

### Gruplanmış Veri Kuralları
- **s² = Σ(xⱼ-x̄)²fⱼ / (n-1)** ya da yığın için **N** ile
- Sınıf aralıklı veride **xⱼ**, sınıf orta değeridir.
- Kısa yol:
  - **Varyans = Kareler ortalaması - Ortalamanın karesi**
        """
    )

    mode = data_structure_selector("b2_var_mode")
    sample = st.checkbox("Örnek varyansı kullan", value=True, key="b2_var_sample")

    if mode == "Gruplanmamış Veri":
        raw = st.text_area("Sayıları girin", "10,15,22,26,31,40", height=100, key="b2_var_raw")
        try:
            data = parse_numeric_input(raw)
            mean_val = arithmetic_mean(data)
            sq = (data - mean_val) ** 2
            vals = variance_std_value(data)
            var_key = "variance_sample" if sample else "variance_pop"
            std_key = "std_sample" if sample else "std_pop"

            st.dataframe(pd.DataFrame({"xᵢ": data, "(xᵢ-x̄)²": np.round(sq, 4)}), use_container_width=True)
            st.success(f"Varyans = {vals[var_key]:.4f}")
            st.success(f"Standart Sapma = {vals[std_key]:.4f}")
        except ValueError as err:
            st.error(str(err))
    else:
        grouped_type = grouped_data_type_selector("b2_var_group_type")
        if grouped_type == "Kesikli Gruplanmış Veri":
            raw = st.text_area(
                "Her satır: değer:frekans",
                "1:4\n3:6\n5:31\n7:2\n9:1",
                height=140,
                key="b2_var_group_discrete",
            )
            try:
                x_vals, freqs = parse_grouped_discrete_input(raw)
                vals = grouped_variance_std(x_vals, freqs, sample=sample)
                short = grouped_variance_shortcut(x_vals, freqs, sample=sample)

                df = grouped_discrete_table(x_vals, freqs)
                df["(xⱼ-x̄)²"] = np.round((x_vals - vals["mean"]) ** 2, 4)
                df["(xⱼ-x̄)²·fⱼ"] = np.round(((x_vals - vals["mean"]) ** 2) * freqs, 4)
                st.dataframe(df, use_container_width=True)

                st.success(f"Varyans = {vals['variance']:.4f}")
                st.success(f"Standart Sapma = {vals['std']:.4f}")
                st.caption(f"Kısa yol kontrolü (kareler ortalaması - ortalamanın karesi): {short['variance_pop']:.4f}")
            except ValueError as err:
                st.error(str(err))
        else:
            raw = st.text_area(
                "Her satır: alt-üst:frekans",
                "0-20:5\n20-40:32\n40-60:54\n60-80:5\n80-100:4",
                height=160,
                key="b2_var_group_interval",
            )
            try:
                lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
                vals = grouped_variance_std(mids, freqs, sample=sample)
                short = grouped_variance_shortcut(mids, freqs, sample=sample)

                df = grouped_interval_table(lower, upper, mids, freqs)
                df["(xⱼ-x̄)²"] = np.round((mids - vals["mean"]) ** 2, 4)
                df["(xⱼ-x̄)²·fⱼ"] = np.round(((mids - vals["mean"]) ** 2) * freqs, 4)
                st.dataframe(df, use_container_width=True)

                st.success(f"Varyans = {vals['variance']:.4f}")
                st.success(f"Standart Sapma = {vals['std']:.4f}")
                st.caption(f"Kısa yol kontrolü (kareler ortalaması - ortalamanın karesi): {short['variance_pop']:.4f}")
                show_group_note()
            except ValueError as err:
                st.error(str(err))


def render_cv_skew_page():
    st.header("11. Değişim Katsayısı ve Çarpıklık")
    st.markdown(
        """
### Değişim Katsayısı
- **C = σ / μ** ya da örnek için **s / x̄**
- Ortalama farklı iki dağılımı homojenlik açısından karşılaştırmak için kullanılır.
- C küçükse dağılım daha homojendir.

### Pearson Çarpıklık Katsayıları
- **Çp1 = (Aritmetik Ortalama - Tepedeğer) / Standart Sapma**
- **Çp2 = 3(Aritmetik Ortalama - Ortanca) / Standart Sapma**

### Bowley Çarpıklık Katsayısı
- **ÇQ = (Q3 - 2Q2 + Q1) / (Q3 - Q1)**

### Yorum
- Katsayı pozitifse sağa çarpık
- Katsayı negatifse sola çarpık
- Sıfıra yakınsa simetriye yakın
        """
    )

    mode = data_structure_selector("b2_cv_mode")

    if mode == "Gruplanmamış Veri":
        raw = st.text_area(
            "Sayıları girin",
            "7,8,9,10,11,12,13,14,15,11,12,12,12,13,13,13,14",
            height=100,
            key="b2_cv_raw",
        )
        try:
            data = parse_numeric_input(raw)
            mean_val = arithmetic_mean(data)
            median_val = median_value(data)
            modes = mode_value(data)
            q1, q2, q3, _, _ = quartiles_value(data)
            vals = variance_std_value(data)
            std = vals["std_pop"]
            cv = coefficient_of_variation_value(data, use_sample=False)
            p1 = pearson_skewness_mode(mean_val, modes[0], std)
            p2 = pearson_skewness_median(mean_val, median_val, std)
            b = bowley_skewness_value(q1, q2, q3)

            c1, c2, c3 = st.columns(3)
            c1.metric("Değişim Katsayısı", f"{cv:.4f}")
            c2.metric("Pearson Çp1", f"{p1:.4f}")
            c3.metric("Pearson Çp2", f"{p2:.4f}")
            st.metric("Bowley ÇQ", f"{b:.4f}")
            st.info(skewness_comment(p2))
        except ValueError as err:
            st.error(str(err))
    else:
        grouped_type = grouped_data_type_selector("b2_cv_group_type")
        if grouped_type == "Kesikli Gruplanmış Veri":
            raw = st.text_area(
                "Her satır: değer:frekans",
                "7:1\n8:2\n9:3\n10:7\n11:12\n12:16\n13:15\n14:13\n15:9",
                height=180,
                key="b2_cv_group_discrete",
            )
            try:
                x_vals, freqs = parse_grouped_discrete_input(raw)
                cv, vals = grouped_cv(x_vals, freqs, sample=False)
                expanded = expand_discrete_grouped_data(x_vals, freqs)
                med = median_value(expanded)
                modes = x_vals[freqs == freqs.max()]
                q1, q2, q3, _, _ = quartiles_value(expanded)
                p1 = pearson_skewness_mode(vals["mean"], float(modes[0]), vals["std"])
                p2 = pearson_skewness_median(vals["mean"], med, vals["std"])
                b = bowley_skewness_value(q1, q2, q3)

                c1, c2, c3 = st.columns(3)
                c1.metric("Değişim Katsayısı", f"{cv:.4f}")
                c2.metric("Pearson Çp1", f"{p1:.4f}")
                c3.metric("Pearson Çp2", f"{p2:.4f}")
                st.metric("Bowley ÇQ", f"{b:.4f}")
                st.info(skewness_comment(p2))
                st.dataframe(grouped_discrete_table(x_vals, freqs), use_container_width=True)
            except ValueError as err:
                st.error(str(err))
        else:
            raw = st.text_area(
                "Her satır: alt-üst:frekans",
                "0-20:5\n20-40:10\n40-60:15\n60-80:20\n80-100:16",
                height=160,
                key="b2_cv_group_interval",
            )
            try:
                lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
                cv, vals = grouped_cv(mids, freqs, sample=False)
                med, _, _, _, _, _, _ = grouped_median_interval(lower, upper, freqs)
                q1, _, _, _, _, _ = grouped_quartile_interval(lower, upper, freqs, which=1)
                q3, _, _, _, _, _ = grouped_quartile_interval(lower, upper, freqs, which=3)
                mode_info = grouped_mode_interval(lower, upper, freqs)
                p1 = pearson_skewness_mode(vals["mean"], mode_info["mode_formula2a"], vals["std"])
                p2 = pearson_skewness_median(vals["mean"], med, vals["std"])
                b = bowley_skewness_value(q1, med, q3)

                c1, c2, c3 = st.columns(3)
                c1.metric("Değişim Katsayısı", f"{cv:.4f}")
                c2.metric("Pearson Çp1", f"{p1:.4f}")
                c3.metric("Pearson Çp2", f"{p2:.4f}")
                st.metric("Bowley ÇQ", f"{b:.4f}")
                st.info(skewness_comment(p2))
                st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)
                show_group_note()
            except ValueError as err:
                st.error(str(err))


def render_examples_page():
    st.header("12. Çözümlü Örnek Sorular")

    with st.expander("Örnek 1 - Gruplanmamış veri için aritmetik ortalama"):
        data = parse_numeric_input("20,35,42,36,50,27")
        st.code("20, 35, 42, 36, 50, 27")
        st.write(f"Ortalama = {arithmetic_mean(data):.2f}")

    with st.expander("Örnek 2 - Gruplanmış veri için aritmetik ortalama"):
        raw = "0-20:5\n20-40:10\n40-60:20\n60-80:14\n80-100:6"
        lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
        st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)
        st.write(f"Ortalama = {grouped_mean(mids, freqs):.2f}")

    with st.expander("Örnek 3 - Gruplanmış veri için ortanca"):
        raw = "170-206:30\n206-242:83\n242-278:68\n278-314:35\n314-350:34"
        lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
        med, idx, _, _, _, _, _ = grouped_median_interval(lower, upper, freqs)
        st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)
        st.write(f"Ortanca sınıfı = {lower[idx]:.0f}-{upper[idx]:.0f}")
        st.write(f"Ortanca = {med:.2f}")

    with st.expander("Örnek 4 - Gruplanmış veri için tepe değer"):
        raw = "140-150:5\n150-160:100\n160-170:250\n170-180:60\n180-190:10"
        lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
        vals = grouped_mode_interval(lower, upper, freqs)
        st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)
        st.write(f"Formül 1 = {vals['mode_formula1']:.2f}")
        st.write(f"Formül 2 = {vals['mode_formula2a']:.2f}")

    with st.expander("Örnek 5 - Gruplanmış veri için çeyrek ayrılış"):
        raw = "0-20:5\n20-40:9\n40-60:20\n60-80:29\n80-100:13"
        lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
        q1, _, _, _, _, _ = grouped_quartile_interval(lower, upper, freqs, which=1)
        q3, _, _, _, _, _ = grouped_quartile_interval(lower, upper, freqs, which=3)
        q = (q3 - q1) / 2
        st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)
        st.write(f"Q1 = {q1:.2f}, Q3 = {q3:.2f}, Çeyrek Ayrılış = {q:.2f}")

    with st.expander("Örnek 6 - Gruplanmış veri için varyans ve standart sapma"):
        raw = "0-20:5\n20-40:32\n40-60:54\n60-80:5\n80-100:4"
        lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)
        vals = grouped_variance_std(mids, freqs, sample=False)
        st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)
        st.write(f"Varyans = {vals['variance']:.2f}")
        st.write(f"Standart Sapma = {vals['std']:.2f}")


def render_quiz_page():
    st.header("13. Mini Quiz")
    score = 0

    q1 = st.radio(
        "1) Gruplanmış veride ortancayı bulmak için önce ne belirlenir?",
        ["Tepe değer sınıfı", "Ortanca sınıfı", "Sınıf orta değeri"],
        key="b2_quiz_q1",
    )
    if q1 == "Ortanca sınıfı":
        score += 1

    q2 = st.radio(
        "2) Geometrik ortalama için hangi koşul gerekir?",
        ["Tüm değerler pozitif olmalı", "Tüm frekanslar eşit olmalı", "Tüm değerler tam sayı olmalı"],
        key="b2_quiz_q2",
    )
    if q2 == "Tüm değerler pozitif olmalı":
        score += 1

    q3 = st.radio(
        "3) Ortalama farklı dağılımları homojenlik açısından karşılaştırmak için hangisi daha uygundur?",
        ["Mod", "Değişim katsayısı", "Ortanca"],
        key="b2_quiz_q3",
    )
    if q3 == "Değişim katsayısı":
        score += 1

    q4 = st.radio(
        "4) Bowley çarpıklığı hangi değerlere dayanır?",
        ["Q1-Q2-Q3", "Min-Ortalama-Max", "Mod-Standart sapma"],
        key="b2_quiz_q4",
    )
    if q4 == "Q1-Q2-Q3":
        score += 1

    if st.button("Quiz Sonucunu Hesapla", key="b2_quiz_btn"):
        st.subheader(f"Puan: {score} / 4")
        if score == 4:
            st.success("Harika. Bölüm 2 kavramlarını çok iyi anlamışsın.")
        elif score >= 2:
            st.info("İyi gidiyorsun. Birkaç ölçüyü tekrar etmen faydalı olur.")
        else:
            st.warning("Bölüm 2 konu anlatımı ve örnek soruları tekrar incele.")


def render_lab_page():
    st.header("14. Öğrenci Veri Laboratuvarı")
    st.write("Kendi verinizi girerek Bölüm 2 ölçülerini hesaplayabilirsiniz.")

    mode = data_structure_selector("b2_lab_mode")

    if mode == "Gruplanmamış Veri":
        raw = st.text_area(
            "Sayıları girin",
            "100,130,90,140,118,120,105,96,99,128",
            height=120,
            key="b2_lab_raw",
        )
        try:
            data = parse_numeric_input(raw)

            mean_val = arithmetic_mean(data)
            med_val = median_value(data)
            modes = mode_value(data)
            q1, q2, q3, iqr, semi_iqr = quartiles_value(data)
            mad = mean_absolute_deviation_value(data)
            var_std = variance_std_value(data)
            cv = coefficient_of_variation_value(data)
            p2 = pearson_skewness_median(mean_val, med_val, var_std["std_pop"])
            b = bowley_skewness_value(q1, q2, q3)

            summary_df = pd.DataFrame(
                {
                    "Ölçü": [
                        "Gözlem Sayısı",
                        "Ortalama",
                        "Medyan",
                        "Mod",
                        "Geometrik Ortalama",
                        "Harmonik Ortalama",
                        "Minimum",
                        "Maksimum",
                        "Açıklık",
                        "Q1",
                        "Q3",
                        "IQR",
                        "Çeyrek Ayrılış",
                        "Ortalama Sapma",
                        "Varyans",
                        "Standart Sapma",
                        "Değişim Katsayısı",
                        "Pearson Çp2",
                        "Bowley ÇQ",
                    ],
                    "Değer": [
                        len(data),
                        round(mean_val, 4),
                        round(med_val, 4),
                        ", ".join(f"{m:.2f}" for m in modes),
                        round(geometric_mean_value(data), 4) if np.all(data > 0) else "Tanımsız",
                        round(harmonic_mean_value(data), 4) if np.all(data > 0) else "Tanımsız",
                        round(np.min(data), 4),
                        round(np.max(data), 4),
                        round(range_value(data), 4),
                        round(q1, 4),
                        round(q3, 4),
                        round(iqr, 4),
                        round(semi_iqr, 4),
                        round(mad, 4),
                        round(var_std["variance_pop"], 4),
                        round(var_std["std_pop"], 4),
                        round(cv, 4),
                        round(p2, 4),
                        round(b, 4),
                    ],
                }
            )
            st.dataframe(summary_df, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_boxplot(data))
            with c2:
                _, edges = np.histogram(data, bins=5)
                st.pyplot(plot_hist(data, edges))
        except ValueError as err:
            st.error(str(err))

    else:
        grouped_type = grouped_data_type_selector("b2_lab_group_type")

        if grouped_type == "Kesikli Gruplanmış Veri":
            raw = st.text_area(
                "Her satır: değer:frekans",
                "7:1\n8:2\n9:3\n10:7\n11:12\n12:16\n13:15\n14:13\n15:9",
                height=180,
                key="b2_lab_group_discrete",
            )
            try:
                x_vals, freqs = parse_grouped_discrete_input(raw)
                expanded = expand_discrete_grouped_data(x_vals, freqs)

                mean_val = grouped_mean(x_vals, freqs)
                med_val = median_value(expanded)
                modes = x_vals[freqs == freqs.max()]
                q1, q2, q3, iqr, semi_iqr = quartiles_value(expanded)
                mad, _ = grouped_mean_absolute_deviation(x_vals, freqs)
                var_std = grouped_variance_std(x_vals, freqs, sample=False)
                cv, _ = grouped_cv(x_vals, freqs, sample=False)
                p2 = pearson_skewness_median(mean_val, med_val, var_std["std"])
                b = bowley_skewness_value(q1, q2, q3)

                summary_df = pd.DataFrame(
                    {
                        "Ölçü": [
                            "Toplam Frekans",
                            "Ortalama",
                            "Medyan",
                            "Mod",
                            "Geometrik Ortalama",
                            "Harmonik Ortalama",
                            "Açıklık",
                            "Q1",
                            "Q3",
                            "IQR",
                            "Çeyrek Ayrılış",
                            "Ortalama Sapma",
                            "Varyans",
                            "Standart Sapma",
                            "Değişim Katsayısı",
                            "Pearson Çp2",
                            "Bowley ÇQ",
                        ],
                        "Değer": [
                            int(freqs.sum()),
                            round(mean_val, 4),
                            round(med_val, 4),
                            ", ".join(map(lambda x: f"{x:.2f}", modes)),
                            round(grouped_geometric_mean(x_vals, freqs), 4) if np.all(x_vals > 0) else "Tanımsız",
                            round(grouped_harmonic_mean(x_vals, freqs), 4) if np.all(x_vals > 0) else "Tanımsız",
                            round(range_value(expanded), 4),
                            round(q1, 4),
                            round(q3, 4),
                            round(iqr, 4),
                            round(semi_iqr, 4),
                            round(mad, 4),
                            round(var_std["variance"], 4),
                            round(var_std["std"], 4),
                            round(cv, 4),
                            round(p2, 4),
                            round(b, 4),
                        ],
                    }
                )
                st.dataframe(summary_df, use_container_width=True)
                st.dataframe(grouped_discrete_table(x_vals, freqs), use_container_width=True)
            except ValueError as err:
                st.error(str(err))

        else:
            raw = st.text_area(
                "Her satır: alt-üst:frekans",
                "0-20:5\n20-40:10\n40-60:15\n60-80:20\n80-100:16",
                height=160,
                key="b2_lab_group_interval",
            )
            try:
                lower, upper, mids, freqs, _ = parse_grouped_interval_input(raw)

                mean_val = grouped_mean(mids, freqs)
                med_val, _, _, _, _, _, _ = grouped_median_interval(lower, upper, freqs)
                mode_info = grouped_mode_interval(lower, upper, freqs)
                q1, _, _, _, _, _ = grouped_quartile_interval(lower, upper, freqs, which=1)
                q3, _, _, _, _, _ = grouped_quartile_interval(lower, upper, freqs, which=3)
                mad, _ = grouped_mean_absolute_deviation(mids, freqs)
                var_std = grouped_variance_std(mids, freqs, sample=False)
                cv, _ = grouped_cv(mids, freqs, sample=False)
                p2 = pearson_skewness_median(mean_val, med_val, var_std["std"])
                b = bowley_skewness_value(q1, med_val, q3)

                summary_df = pd.DataFrame(
                    {
                        "Ölçü": [
                            "Toplam Frekans",
                            "Ortalama",
                            "Ortanca",
                            "Tepe Değer",
                            "Geometrik Ortalama",
                            "Harmonik Ortalama",
                            "Açıklık",
                            "Q1",
                            "Q3",
                            "Çeyrek Ayrılış",
                            "Ortalama Sapma",
                            "Varyans",
                            "Standart Sapma",
                            "Değişim Katsayısı",
                            "Pearson Çp2",
                            "Bowley ÇQ",
                        ],
                        "Değer": [
                            int(freqs.sum()),
                            round(mean_val, 4),
                            round(med_val, 4),
                            round(mode_info["mode_formula2a"], 4),
                            round(grouped_geometric_mean(mids, freqs), 4) if np.all(mids > 0) else "Tanımsız",
                            round(grouped_harmonic_mean(mids, freqs), 4) if np.all(mids > 0) else "Tanımsız",
                            round(grouped_range_interval(lower, upper), 4),
                            round(q1, 4),
                            round(q3, 4),
                            round((q3 - q1) / 2, 4),
                            round(mad, 4),
                            round(var_std["variance"], 4),
                            round(var_std["std"], 4),
                            round(cv, 4),
                            round(p2, 4),
                            round(b, 4),
                        ],
                    }
                )

                st.dataframe(summary_df, use_container_width=True)
                st.dataframe(grouped_interval_table(lower, upper, mids, freqs), use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(plot_grouped_hist(lower, upper, freqs))
                with c2:
                    st.pyplot(plot_hist(np.repeat(mids, freqs), bins=5, title="Sınıf Orta Değerleri Histogramı"))

                show_group_note()
            except ValueError as err:
                st.error(str(err))


def render_bolum2():
    st.sidebar.title("📚 İçindekiler")

    page = st.sidebar.selectbox(
        "Konu",
        [
            "1. Giriş ve Amaç",
            "2. Aritmetik Ortalama",
            "3. Ortanca (Medyan)",
            "4. Tepedeğer (Mod)",
            "5. Geometrik Ortalama",
            "6. Harmonik Ortalama",
            "7. Açıklık ve Çeyrek Ayrılış",
            "8. Kutu Grafiği",
            "9. Ortalama Sapma",
            "10. Varyans ve Standart Sapma",
            "11. Değişim Katsayısı ve Çarpıklık",
            "12. Çözümlü Örnek Sorular",
            "13. Mini Quiz",
            "14. Öğrenci Veri Laboratuvarı",
        ],
        key="bolum2_page",
    )

    if page == "1. Giriş ve Amaç":
        render_intro_page()
    elif page == "2. Aritmetik Ortalama":
        render_mean_page()
    elif page == "3. Ortanca (Medyan)":
        render_median_page()
    elif page == "4. Tepedeğer (Mod)":
        render_mode_page()
    elif page == "5. Geometrik Ortalama":
        render_geometric_page()
    elif page == "6. Harmonik Ortalama":
        render_harmonic_page()
    elif page == "7. Açıklık ve Çeyrek Ayrılış":
        render_range_quartile_page()
    elif page == "8. Kutu Grafiği":
        render_boxplot_page()
    elif page == "9. Ortalama Sapma":
        render_mad_page()
    elif page == "10. Varyans ve Standart Sapma":
        render_variance_page()
    elif page == "11. Değişim Katsayısı ve Çarpıklık":
        render_cv_skew_page()
    elif page == "12. Çözümlü Örnek Sorular":
        render_examples_page()
    elif page == "13. Mini Quiz":
        render_quiz_page()
    elif page == "14. Öğrenci Veri Laboratuvarı":
        render_lab_page()