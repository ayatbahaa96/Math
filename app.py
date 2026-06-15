import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from bolum3 import render_bolum3
from bolum4 import render_bolum4
from bolum5 import render_bolum5
from bolum6 import render_bolum6
from bolum7 import render_bolum7
from bolum8 import render_bolum8


st.set_page_config(
    page_title="Olasılık ve İstatistik Platformu",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Helpers
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


def parse_categorical_input(text: str):
    cleaned = text.replace("\n", ",").replace(";", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        raise ValueError("Lütfen en az bir kategori girin.")
    return parts


def basic_stats(data: np.ndarray):
    data = np.array(data, dtype=float)
    modes = pd.Series(data).mode().tolist()
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return {
        "n": len(data),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "range": float(np.max(data) - np.min(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "mode": modes,
        "variance_pop": float(np.var(data, ddof=0)),
        "variance_sample": float(np.var(data, ddof=1)) if len(data) > 1 else 0.0,
        "std_pop": float(np.std(data, ddof=0)),
        "std_sample": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def categorical_frequency_table(values):
    series = pd.Series(values, dtype="object")
    freq = series.value_counts(dropna=False)
    rel = freq / len(series)
    cumulative = freq.cumsum()
    cumulative_rel = rel.cumsum()
    return pd.DataFrame(
        {
            "Kategori": freq.index.astype(str),
            "Frekans": freq.values,
            "Oransal Frekans": np.round(rel.values, 4),
            "Yüzde": np.round(rel.values * 100, 2),
            "Birikimli Frekans": cumulative.values,
            "Birikimli Yüzde": np.round(cumulative_rel.values * 100, 2),
        }
    )


def grouped_frequency_table(data: np.ndarray, class_count: int):
    data = np.array(data, dtype=float)
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    if min_val == max_val:
        bins = np.array([min_val - 0.5, min_val + 0.5])
        class_count = 1
    else:
        bins = np.linspace(min_val, max_val, class_count + 1)
    freq, edges = np.histogram(data, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    cumulative = np.cumsum(freq)
    rel = freq / len(data)
    width = edges[1] - edges[0]
    df = pd.DataFrame(
        {
            "Sınıf Aralığı": [
                f"[{edges[i]:.2f}, {edges[i+1]:.2f})" if i < len(freq) - 1 else f"[{edges[i]:.2f}, {edges[i+1]:.2f}]"
                for i in range(len(freq))
            ],
            "Alt Sınır": np.round(edges[:-1], 2),
            "Üst Sınır": np.round(edges[1:], 2),
            "Sınıf Orta Noktası": np.round(mids, 2),
            "Frekans": freq,
            "Oransal Frekans": np.round(rel, 4),
            "Yüzde": np.round(rel * 100, 2),
            "Birikimli Frekans": cumulative,
        }
    )
    return df, edges, width


def simple_frequency_table(data: np.ndarray):
    s = pd.Series(data)
    freq = s.value_counts().sort_index()
    rel = freq / len(s)
    cumulative = freq.cumsum()
    return pd.DataFrame(
        {
            "Değer": freq.index,
            "Frekans": freq.values,
            "Oransal Frekans": np.round(rel.values, 4),
            "Yüzde": np.round(rel.values * 100, 2),
            "Birikimli Frekans": cumulative.values,
        }
    )


def make_stem_leaf(data: np.ndarray):
    values = sorted(int(round(v)) for v in data)
    stems = {}
    for v in values:
        stem = v // 10
        leaf = abs(v) % 10
        stems.setdefault(stem, []).append(str(leaf))
    lines = [f"{stem} | {' '.join(leaves)}" for stem, leaves in stems.items()]
    return "\n".join(lines) if lines else "Veri bulunamadı."


def measurement_level_explainer(option: str):
    mapping = {
        "Sınıflama (Nominal)": "Sadece kategori belirtir. Sıralama yoktur. Örnek: göz rengi, kan grubu, bölüm adı.",
        "Sıralama (Ordinal)": "Kategoriler arasında sıra vardır; fakat farkların büyüklüğü eşit kabul edilmez. Örnek: memnuniyet düzeyi, yarış derecesi.",
        "Eşit Aralıklı (Interval)": "Sıra ve fark anlamlıdır; ancak gerçek sıfır yoktur. Örnek: Celsius sıcaklığı, takvim yılı.",
        "Oranlama (Ratio)": "Sıra, fark ve gerçek sıfır vardır. Oran yorumları yapılabilir. Örnek: boy, kilo, yaş, gelir.",
    }
    return mapping[option]


def plot_bar(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table.iloc[:, 0].astype(str), table["Frekans"])
    ax.set_title("Çubuk Grafik")
    ax.set_xlabel(table.columns[0])
    ax.set_ylabel("Frekans")
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig


def plot_pie(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(table["Frekans"], labels=table.iloc[:, 0].astype(str), autopct="%1.1f%%", startangle=90)
    ax.set_title("Daire Grafik")
    return fig


def plot_hist(data, bins):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins, edgecolor="black")
    ax.set_title("Histogram")
    ax.set_xlabel("Değer")
    ax.set_ylabel("Frekans")
    plt.tight_layout()
    return fig


def plot_frequency_polygon(grouped_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grouped_df["Sınıf Orta Noktası"], grouped_df["Frekans"], marker="o")
    ax.set_title("Frekans Poligonu")
    ax.set_xlabel("Sınıf Orta Noktası")
    ax.set_ylabel("Frekans")
    plt.tight_layout()
    return fig


def plot_ogive(grouped_df: pd.DataFrame, kind="less"):
    fig, ax = plt.subplots(figsize=(8, 4))
    if kind == "less":
        x = grouped_df["Üst Sınır"]
        y = grouped_df["Birikimli Frekans"]
        title = "Küçüktür Birikimli Frekans Eğrisi (Ogive)"
    else:
        x = grouped_df["Alt Sınır"]
        total = int(grouped_df["Frekans"].sum())
        y = total - grouped_df["Birikimli Frekans"].shift(fill_value=0)
        title = "Büyüktür Birikimli Frekans Eğrisi (Ogive)"
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Sınıf Sınırı")
    ax.set_ylabel("Birikimli Frekans")
    plt.tight_layout()
    return fig


def plot_time_series(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(data) + 1), data, marker="o")
    ax.set_title("Zaman Serisi Grafiği")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Değer")
    plt.tight_layout()
    return fig


# -----------------------------
# Bölüm 2 Helpers
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


def coefficient_of_variation_value(data: np.ndarray):
    mean = np.mean(data)
    std_sample = np.std(data, ddof=1) if len(data) > 1 else 0.0
    if mean == 0:
        raise ValueError("Değişim katsayısı için ortalama sıfır olmamalıdır.")
    return float((std_sample / mean) * 100)


def pearson_skewness_value(data: np.ndarray):
    mean = np.mean(data)
    median = np.median(data)
    std_sample = np.std(data, ddof=1) if len(data) > 1 else 0.0
    if std_sample == 0:
        return 0.0
    return float(3 * (mean - median) / std_sample)


def bowley_skewness_value(data: np.ndarray):
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    denom = q3 - q1
    if denom == 0:
        return 0.0
    return float((q3 + q1 - 2 * q2) / denom)


def skewness_comment(value: float):
    if value > 0.2:
        return "Dağılım sağa çarpıktır."
    elif value < -0.2:
        return "Dağılım sola çarpıktır."
    return "Dağılım yaklaşık simetriktir."


def plot_boxplot(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, vert=False)
    ax.set_title("Kutu Grafiği")
    ax.set_xlabel("Değer")
    plt.tight_layout()
    return fig


# -----------------------------
# Content
# -----------------------------
st.title("📘 Olasılık ve İstatistik Web Platformu")

st.sidebar.title("📚 İçindekiler")

section = st.sidebar.selectbox(
    "Bölüm Seç",
    [
        "Bölüm 1 - Temel İstatistik",
        "Bölüm 2 - Merkezsel Eğilim Ölçüleri ve Dağılım Ölçüleri",
        "Bölüm 3 - Olasılık, Permütasyon, Kombinasyon ve Bayes",
        "Bölüm 4 - Rastgele Değişkenler ve Çeşitleri",
        "Bölüm 5 - Olasılık Dağılımları ve Veri Analizi",
        "Bölüm 6 - İki Değişkenli Rastgele Değişkenler",
        "Bölüm 7 - İstatistiksel Çıkarım ve Tahmin",
        "Bölüm 8 - Korelasyon ve Regresyon Analizi",
    ]
)

if section == "Bölüm 1 - Temel İstatistik":
    page = st.sidebar.selectbox(
        "Konu",
        [
            "1. Giriş ve Amaç",
            "2. Konu Anlatımı",
            "3. Değişken Türleri ve Ölçme Düzeyleri",
            "4. Frekans Tabloları",
            "5. Grafik Atölyesi",
            "6. Çözümlü Örnek Sorular",
            "7. Mini Quiz",
            "8. Öğrenci Veri Laboratuvarı",
        ],
    )

elif section == "Bölüm 2 - Merkezsel Eğilim Ölçüleri ve Dağılım Ölçüleri":
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
    )

elif section == "Bölüm 3 - Olasılık, Permütasyon, Kombinasyon ve Bayes":
    page = None
elif section == "Bölüm 4 - Rastgele Değişkenler ve Çeşitleri":
    page = None
elif section == "Bölüm 5 - Olasılık Dağılımları ve Veri Analizi":
    page = None
elif section == "Bölüm 6 - İki Değişkenli Rastgele Değişkenler":
    page = None
elif section == "Bölüm 7 - İstatistiksel Çıkarım ve Tahmin":
    page = None
elif section == "Bölüm 8 - Korelasyon ve Regresyon Analizi":
    page = None


# -----------------------------
# BÖLÜM 1
# -----------------------------
if section == "Bölüm 1 - Temel İstatistik":

    if page == "1. Giriş ve Amaç":
        c1, c2, c3 = st.columns(3)
        c1.metric("Modül Türü", "Web Uygulaması")
        c2.metric("İçerik Katmanı", "8")
        c3.metric("Ana Hedef", "Öğret + Uygulat")

        st.markdown(
            """
### Bu modülde neler var?
Bu uygulama, 1. bölümü sadece özetleyen bir sayfa değildir. Bölüm içeriğini **öğrenciye öğreten**, **soru çözdüren**,
**veriyi tabloya dönüştüren**, **grafik üreten** ve **hocaya gösterilebilir bir proje çıktısı** sunan tam bir modüldür.

#### İçerik bileşenleri
- Konu anlatımı ve kavram açıklamaları
- Değişken türü ve ölçme düzeyi sınıflandırma aracı
- Nitel ve nicel veri için frekans tabloları
- Çubuk grafik, daire grafik, histogram, frekans poligonu, ogive, dal-yaprak ve zaman serisi
- Çözümlü örnek sorular
- Mini quiz
- Öğrenci veri laboratuvarı
            """
        )

    elif page == "2. Konu Anlatımı":
        st.header("Konu Anlatımı")
        tab1, tab2, tab3, tab4 = st.tabs([
            "İstatistik Nedir?",
            "Temel Kavramlar",
            "Veri Düzenleme",
            "Grafiklerin Rolü",
        ])

        with tab1:
            st.markdown(
                """
**İstatistik**, verilerin toplanması, düzenlenmesi, özetlenmesi, analiz edilmesi ve yorumlanması ile ilgilenir.

İki temel yaklaşım vardır:
- **Betimsel istatistik:** Mevcut veriyi tablo, grafik ve özet ölçüler ile açıklar.
- **Çıkarımsal istatistik:** Örneklemden hareketle anakütle hakkında yorum yapar.
                """
            )

        with tab2:
            st.markdown(
                """
**Anakütle (evren):** İncelenmek istenen bütün birimler.

**Örneklem:** Anakütleden seçilen alt grup.

**Parametre:** Anakütleye ait sayısal özellik.

**İstatistik:** Örneklemden hesaplanan sayısal özellik.
                """
            )

        with tab3:
            st.markdown(
                """
Veri düzenlemede amaç, ham veriyi anlamlı hale getirmektir.

Bu amaçla:
- frekans tablosu kurulur,
- sınıflandırma yapılır,
- yüzdeler hesaplanır,
- grafiklerle yorum kolaylaştırılır.
                """
            )

        with tab4:
            st.markdown(
                """
Grafikler, verinin şeklini tek bakışta göstermeye yarar.

- **Çubuk grafik:** Kategorik veri
- **Daire grafik:** Kategorik veri yüzdeleri
- **Histogram:** Sürekli nicel veri
- **Frekans poligonu:** Sınıflı dağılımın çizgisel görünümü
- **Ogive:** Birikimli frekans yorumu
- **Zaman serisi:** Dönemsel değişim
                """
            )

    elif page == "3. Değişken Türleri ve Ölçme Düzeyleri":
        st.header("Değişken Türleri ve Ölçme Düzeyleri")
        left, right = st.columns(2)

        with left:
            st.subheader("Değişken Türleri")
            st.markdown(
                """
- **Nitel (kategorik):** Sayısal olmayan sınıflar. Örnek: bölüm, cinsiyet, göz rengi.
- **Nicel (sayısal):** Sayısal ölçümler.
  - **Kesikli:** Sayılabilir. Örnek: öğrenci sayısı, hata sayısı.
  - **Sürekli:** Ölçülebilir. Örnek: boy, kilo, süre.
                """
            )
            example = st.selectbox(
                "Örnek değişken seçin",
                ["Boy", "Kan grubu", "Memnuniyet düzeyi", "Öğrenci sayısı", "Sıcaklık (Celsius)"]
            )
            explanations = {
                "Boy": "Nicel - sürekli - oranlama düzeyi",
                "Kan grubu": "Nitel - sınıflama düzeyi",
                "Memnuniyet düzeyi": "Nitel - sıralama düzeyi",
                "Öğrenci sayısı": "Nicel - kesikli - oranlama düzeyi",
                "Sıcaklık (Celsius)": "Nicel - sürekli - eşit aralıklı düzey",
            }
            st.info(explanations[example])

        with right:
            st.subheader("Ölçme Düzeyleri")
            level = st.selectbox(
                "Ölçme düzeyi",
                [
                    "Sınıflama (Nominal)",
                    "Sıralama (Ordinal)",
                    "Eşit Aralıklı (Interval)",
                    "Oranlama (Ratio)",
                ],
            )
            st.write(measurement_level_explainer(level))

        st.markdown("### Hızlı sınıflandırma etkinliği")
        query = st.text_input("Bir değişken yazın", placeholder="Örn: aylık gelir")
        if query:
            q = query.lower()
            if any(k in q for k in ["gelir", "boy", "kilo", "yaş", "uzunluk", "ağırlık"]):
                st.success("Büyük olasılıkla nicel ve oranlama düzeyinde bir değişkendir.")
            elif any(k in q for k in ["renk", "şehir", "bölüm", "kan"]):
                st.success("Büyük olasılıkla nitel ve sınıflama düzeyindedir.")
            elif any(k in q for k in ["memnuniyet", "başarı düzeyi", "sıra", "derece"]):
                st.success("Büyük olasılıkla sıralama düzeyindedir.")
            elif any(k in q for k in ["celsius", "sıcaklık", "takvim", "yıl"]):
                st.success("Büyük olasılıkla eşit aralıklı ölçme düzeyindedir.")
            else:
                st.warning("Bu örnek için kullanıcı yorumu gerekir; uygulama kaba bir ön tahmin sunuyor.")

    elif page == "4. Frekans Tabloları":
        st.header("Frekans Tabloları")
        mode = st.radio("Veri tipi", ["Kategorik Veri", "Sayısal Veri"], horizontal=True)

        if mode == "Kategorik Veri":
            sample = "Mühendis, Öğretmen, Mühendis, Doktor, Öğretmen, Öğretmen, Avukat"
            raw = st.text_area("Kategorileri virgül ile girin", sample, height=120)
            try:
                values = parse_categorical_input(raw)
                table = categorical_frequency_table(values)
                st.dataframe(table, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(plot_bar(table))
                with c2:
                    st.pyplot(plot_pie(table))
            except ValueError as err:
                st.error(str(err))

        else:
            sample = "12,15,16,16,17,18,18,19,20,21,21,22,24,25,26,28,30,31,33,35"
            raw = st.text_area("Sayıları virgül ile girin", sample, height=120)
            class_count = st.slider("Sınıf sayısı", 3, 10, 5)
            table_type = st.radio("Tablo tipi", ["Basit frekans tablosu", "Sınıflı frekans tablosu"], horizontal=True)
            try:
                data = parse_numeric_input(raw)
                if table_type == "Basit frekans tablosu":
                    table = simple_frequency_table(data)
                    st.dataframe(table, use_container_width=True)
                else:
                    grouped_df, edges, _ = grouped_frequency_table(data, class_count)
                    st.dataframe(grouped_df, use_container_width=True)
                    st.caption("Sınıf aralıkları otomatik oluşturuldu.")
            except ValueError as err:
                st.error(str(err))

    elif page == "5. Grafik Atölyesi":
        st.header("Grafik Atölyesi")
        graph_type = st.selectbox(
            "Grafik seçin",
            [
                "Çubuk Grafik",
                "Daire Grafik",
                "Histogram",
                "Frekans Poligonu",
                "Ogive (Küçüktür)",
                "Ogive (Büyüktür)",
                "Dal-Yaprak Gösterimi",
                "Zaman Serisi",
            ],
        )

        if graph_type in ["Çubuk Grafik", "Daire Grafik"]:
            raw = st.text_area(
                "Kategori verisi girin",
                "A, B, A, C, B, A, D, C, B, A",
                height=100,
            )
            try:
                values = parse_categorical_input(raw)
                table = categorical_frequency_table(values)
                st.dataframe(table, use_container_width=True)
                if graph_type == "Çubuk Grafik":
                    st.pyplot(plot_bar(table))
                else:
                    st.pyplot(plot_pie(table))
            except ValueError as err:
                st.error(str(err))

        elif graph_type == "Dal-Yaprak Gösterimi":
            raw = st.text_area("Tam sayıları girin", "12,13,14,15,21,22,22,25,31,34,35,39", height=100)
            try:
                data = parse_numeric_input(raw)
                st.code(make_stem_leaf(data))
            except ValueError as err:
                st.error(str(err))

        elif graph_type == "Zaman Serisi":
            raw = st.text_area("Zamana göre değerleri girin", "120,132,128,140,150,148,160", height=100)
            try:
                data = parse_numeric_input(raw)
                st.pyplot(plot_time_series(data))
            except ValueError as err:
                st.error(str(err))

        else:
            raw = st.text_area(
                "Sayısal veri girin",
                "10,12,13,15,16,18,18,19,21,22,24,25,25,27,28,30,31,33,35,36",
                height=100,
            )
            class_count = st.slider("Sınıf sayısı", 3, 10, 5, key="graph_class_count")
            try:
                data = parse_numeric_input(raw)
                grouped_df, edges, _ = grouped_frequency_table(data, class_count)
                st.dataframe(grouped_df, use_container_width=True)
                if graph_type == "Histogram":
                    st.pyplot(plot_hist(data, edges))
                elif graph_type == "Frekans Poligonu":
                    st.pyplot(plot_frequency_polygon(grouped_df))
                elif graph_type == "Ogive (Küçüktür)":
                    st.pyplot(plot_ogive(grouped_df, kind="less"))
                else:
                    st.pyplot(plot_ogive(grouped_df, kind="more"))
            except ValueError as err:
                st.error(str(err))

    elif page == "6. Çözümlü Örnek Sorular":
        st.header("Çözümlü Örnek Sorular")

        with st.expander("Soru 1 - Kategorik veri için frekans tablosu ve grafik"):
            st.write("Bir sınıftaki öğrencilerin tercih ettiği kulüpler şu şekildedir:")
            st.code("Müzik, Spor, Spor, Tiyatro, Müzik, Spor, Satranç, Müzik, Spor, Tiyatro")
            if st.button("Soru 1 çözümünü göster"):
                vals = parse_categorical_input("Müzik, Spor, Spor, Tiyatro, Müzik, Spor, Satranç, Müzik, Spor, Tiyatro")
                table = categorical_frequency_table(vals)
                st.markdown("**Çözüm adımları**")
                st.write("1. Kategoriler sayılır.")
                st.write("2. Frekans ve yüzde hesaplanır.")
                st.write("3. Uygun grafik çubuk veya daire grafiktir.")
                st.dataframe(table, use_container_width=True)
                st.pyplot(plot_bar(table))

        with st.expander("Soru 2 - Sayısal veri için temel istatistikler"):
            st.write("Aşağıdaki veri için ortalama, medyan ve standart sapmayı bulunuz:")
            st.code("4, 6, 8, 10, 12")
            if st.button("Soru 2 çözümünü göster"):
                data = np.array([4, 6, 8, 10, 12], dtype=float)
                stats = basic_stats(data)
                st.write(f"Ortalama = (4+6+8+10+12)/5 = {stats['mean']:.2f}")
                st.write(f"Medyan = {stats['median']:.2f}")
                st.write(f"Örnek standart sapma = {stats['std_sample']:.2f}")

        with st.expander("Soru 3 - Sınıflı frekans tablosu ve histogram"):
            st.write("Bir ölçüm dizisi için sınıflı frekans tablosu oluşturup histogram çiziniz.")
            st.code("11,12,13,15,15,16,18,19,20,21,22,24,25,27,28,28,29,30,32,35")
            if st.button("Soru 3 çözümünü göster"):
                data = parse_numeric_input("11,12,13,15,15,16,18,19,20,21,22,24,25,27,28,28,29,30,32,35")
                grouped_df, edges, _ = grouped_frequency_table(data, 5)
                st.dataframe(grouped_df, use_container_width=True)
                st.pyplot(plot_hist(data, edges))
                st.pyplot(plot_frequency_polygon(grouped_df))

    elif page == "7. Mini Quiz":
        st.header("Mini Quiz")
        score = 0

        q1 = st.radio(
            "1) 'Göz rengi' hangi tür değişkendir?",
            ["Nicel-sürekli", "Nitel-kategorik", "Nicel-kesikli"],
            key="b1_q1",
        )
        if q1 == "Nitel-kategorik":
            score += 1

        q2 = st.radio(
            "2) Hangisi oranlama düzeyindedir?",
            ["Takvim yılı", "Sıcaklık (Celsius)", "Ağırlık"],
            key="b1_q2",
        )
        if q2 == "Ağırlık":
            score += 1

        q3 = st.radio(
            "3) Sürekli nicel veri için en uygun grafik hangisidir?",
            ["Histogram", "Daire grafik", "Sadece pasta grafik"],
            key="b1_q3",
        )
        if q3 == "Histogram":
            score += 1

        q4 = st.radio(
            "4) Birikimli frekansı izlemek için hangi grafik kullanılır?",
            ["Ogive", "Çubuk grafik", "Dal-yaprak"],
            key="b1_q4",
        )
        if q4 == "Ogive":
            score += 1

        if st.button("Quiz sonucunu hesapla", key="b1_quiz_button"):
            st.subheader(f"Puan: {score} / 4")
            if score == 4:
                st.success("Mükemmel. Bölüm 1 kavramlarını çok iyi anlamışsın.")
            elif score >= 2:
                st.info("İyi gidiyorsun. Birkaç kavramı tekrar etmen faydalı olur.")
            else:
                st.warning("Konu anlatımı ve çözümlü sorular kısmını tekrar incele.")

    elif page == "8. Öğrenci Veri Laboratuvarı":
        st.header("Öğrenci Veri Laboratuvarı")
        st.write("Kendi verinizi girip tablo, grafik ve özet istatistik üretebilirsiniz.")

        data_mode = st.radio("Veri yapısı", ["Sayısal Veri", "Kategorik Veri"], horizontal=True)

        if data_mode == "Sayısal Veri":
            raw = st.text_area(
                "Sayıları girin",
                "5,7,8,3,10,12,6,9,11,15,18,21",
                height=120,
            )
            class_count = st.slider("Sınıf sayısı", 3, 10, 5, key="lab_class_count")
            try:
                data = parse_numeric_input(raw)
                stats = basic_stats(data)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ortalama", f"{stats['mean']:.2f}")
                c2.metric("Medyan", f"{stats['median']:.2f}")
                c3.metric("Std. Sapma", f"{stats['std_sample']:.2f}")
                c4.metric("Açıklık", f"{stats['range']:.2f}")

                st.markdown("#### Temel istatistik tablosu")
                summary_df = pd.DataFrame(
                    {
                        "Ölçü": [
                            "Gözlem Sayısı",
                            "Minimum",
                            "Maksimum",
                            "Ortalama",
                            "Medyan",
                            "Mod",
                            "Q1",
                            "Q3",
                            "IQR",
                            "Örnek Varyans",
                            "Örnek Standart Sapma",
                        ],
                        "Değer": [
                            stats["n"],
                            round(stats["min"], 2),
                            round(stats["max"], 2),
                            round(stats["mean"], 2),
                            round(stats["median"], 2),
                            ", ".join(str(int(m)) if float(m).is_integer() else f"{m:.2f}" for m in stats["mode"]),
                            round(stats["q1"], 2),
                            round(stats["q3"], 2),
                            round(stats["iqr"], 2),
                            round(stats["variance_sample"], 2),
                            round(stats["std_sample"], 2),
                        ],
                    }
                )
                st.dataframe(summary_df, use_container_width=True)

                st.markdown("#### Frekans tabloları")
                left, right = st.columns(2)
                with left:
                    st.write("Basit frekans tablosu")
                    st.dataframe(simple_frequency_table(data), use_container_width=True)
                with right:
                    st.write("Sınıflı frekans tablosu")
                    grouped_df, edges, _ = grouped_frequency_table(data, class_count)
                    st.dataframe(grouped_df, use_container_width=True)

                st.markdown("#### Grafikler")
                g1, g2 = st.columns(2)
                with g1:
                    st.pyplot(plot_hist(data, edges))
                    st.pyplot(plot_frequency_polygon(grouped_df))
                with g2:
                    st.pyplot(plot_ogive(grouped_df, kind="less"))
                    st.pyplot(plot_time_series(data))

                st.markdown("#### Dal-yaprak gösterimi")
                st.code(make_stem_leaf(data))

            except ValueError as err:
                st.error(str(err))

        else:
            raw = st.text_area(
                "Kategorileri girin",
                "Mavi, Yeşil, Mavi, Kırmızı, Yeşil, Mavi, Sarı, Kırmızı",
                height=120,
            )
            try:
                values = parse_categorical_input(raw)
                table = categorical_frequency_table(values)
                st.dataframe(table, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(plot_bar(table))
                with c2:
                    st.pyplot(plot_pie(table))
            except ValueError as err:
                st.error(str(err))



# -----------------------------
# BÖLÜM 3
# -----------------------------
elif section == "Bölüm 3 - Olasılık, Permütasyon, Kombinasyon ve Bayes":
    render_bolum3()


# -----------------------------
# BÖLÜM 4
# -----------------------------
elif section == "Bölüm 4 - Rastgele Değişkenler ve Çeşitleri":
    render_bolum4()

elif section == "Bölüm 5 - Olasılık Dağılımları ve Veri Analizi":
    render_bolum5()

elif section == "Bölüm 6 - İki Değişkenli Rastgele Değişkenler":
    render_bolum6()

elif section == "Bölüm 7 - İstatistiksel Çıkarım ve Tahmin":
    render_bolum7()

elif section == "Bölüm 8 - Korelasyon ve Regresyon Analizi":
    render_bolum8()


# -----------------------------
# BÖLÜM 2
# -----------------------------
elif section == "Bölüm 2 - Merkezsel Eğilim Ölçüleri ve Dağılım Ölçüleri":

    if page == "1. Giriş ve Amaç":
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
            """
        )

    elif page == "2. Aritmetik Ortalama":
        st.header("Aritmetik Ortalama")
        st.markdown(
            """
Aritmetik ortalama, verilerin toplamının gözlem sayısına bölünmesiyle elde edilir.

Formül:
**x̄ = (x₁ + x₂ + ... + xₙ) / n**
            """
        )

        raw = st.text_area("Sayıları girin", "5,7,8,3,10,12,6,9", height=100, key="b2_mean_raw")
        try:
            data = parse_numeric_input(raw)
            mean_val = arithmetic_mean(data)
            st.success(f"Aritmetik Ortalama = {mean_val:.4f}")
            st.write(f"Toplam = {np.sum(data):.2f}")
            st.write(f"Gözlem sayısı = {len(data)}")
            st.write(f"Ortalama = {np.sum(data):.2f} / {len(data)} = {mean_val:.4f}")
        except ValueError as err:
            st.error(str(err))

    elif page == "3. Ortanca (Medyan)":
        st.header("Ortanca (Medyan)")
        st.markdown(
            """
Medyan, sıralanmış verinin ortasında bulunan değerdir.

- n tek ise ortadaki değer alınır.
- n çift ise ortadaki iki değerin ortalaması alınır.
            """
        )

        raw = st.text_area("Sayıları girin", "5,7,8,3,10,12,6,9", height=100, key="b2_median_raw")
        try:
            data = parse_numeric_input(raw)
            sorted_data = np.sort(data)
            med = median_value(data)
            st.write("Sıralanmış veri:", sorted_data.tolist())
            st.success(f"Medyan = {med:.4f}")
        except ValueError as err:
            st.error(str(err))

    elif page == "4. Tepedeğer (Mod)":
        st.header("Tepedeğer (Mod)")
        st.markdown(
            """
Mod, en çok tekrar eden değerdir.

Bir veri setinde:
- tek mod olabilir,
- birden fazla mod olabilir,
- hiç belirgin mod olmayabilir.
            """
        )

        raw = st.text_area("Sayıları girin", "2,3,3,4,5,5,5,6,7,8", height=100, key="b2_mode_raw")
        try:
            data = parse_numeric_input(raw)
            modes = mode_value(data)
            counts = Counter(data)
            freq_df = pd.DataFrame({"Değer": list(counts.keys()), "Frekans": list(counts.values())}).sort_values("Değer")
            st.dataframe(freq_df, use_container_width=True)
            st.success("Mod(lar): " + ", ".join(f"{m:.2f}" for m in modes))
        except ValueError as err:
            st.error(str(err))

    elif page == "5. Geometrik Ortalama":
        st.header("Geometrik Ortalama")
        st.markdown(
            """
Geometrik ortalama, özellikle büyüme oranları ve çarpımsal süreçlerde kullanılır.

Formül:
**G = (x₁·x₂·...·xₙ)^(1/n)**

Not: Tüm değerler pozitif olmalıdır.
            """
        )

        raw = st.text_area("Pozitif sayıları girin", "2,4,8,16", height=100, key="b2_geo_raw")
        try:
            data = parse_numeric_input(raw)
            gmean = geometric_mean_value(data)
            st.success(f"Geometrik Ortalama = {gmean:.4f}")
        except ValueError as err:
            st.error(str(err))

    elif page == "6. Harmonik Ortalama":
        st.header("Harmonik Ortalama")
        st.markdown(
            """
Harmonik ortalama, özellikle hız, oran ve birim başına değer türü problemlerde kullanılır.

Formül:
**H = n / (1/x₁ + 1/x₂ + ... + 1/xₙ)**

Not: Tüm değerler pozitif olmalıdır.
            """
        )

        raw = st.text_area("Pozitif sayıları girin", "2,3,6", height=100, key="b2_harm_raw")
        try:
            data = parse_numeric_input(raw)
            hmean = harmonic_mean_value(data)
            st.success(f"Harmonik Ortalama = {hmean:.4f}")
        except ValueError as err:
            st.error(str(err))

    elif page == "7. Açıklık ve Çeyrek Ayrılış":
        st.header("Açıklık ve Çeyrek Ayrılış")
        st.markdown(
            """
- **Açıklık = Maksimum - Minimum**
- **Çeyrekler** veriyi dört parçaya ayırır.
- **Çeyrekler arası açıklık (IQR) = Q3 - Q1**
- **Çeyrek ayrılış = IQR / 2**
            """
        )

        raw = st.text_area("Sayıları girin", "5,7,8,3,10,12,6,9,11,15,18,21", height=100, key="b2_range_raw")
        try:
            data = parse_numeric_input(raw)
            q1, q2, q3, iqr, semi_iqr = quartiles_value(data)
            c1, c2, c3 = st.columns(3)
            c1.metric("Açıklık", f"{range_value(data):.2f}")
            c2.metric("IQR", f"{iqr:.2f}")
            c3.metric("Çeyrek Ayrılış", f"{semi_iqr:.2f}")

            table = pd.DataFrame(
                {
                    "Ölçü": ["Q1", "Q2 (Medyan)", "Q3", "IQR", "Çeyrek Ayrılış"],
                    "Değer": [round(q1, 2), round(q2, 2), round(q3, 2), round(iqr, 2), round(semi_iqr, 2)],
                }
            )
            st.dataframe(table, use_container_width=True)
        except ValueError as err:
            st.error(str(err))

    elif page == "8. Kutu Grafiği":
        st.header("Kutu Grafiği")
        st.markdown(
            """
Kutu grafiği, medyanı, çeyrekleri ve olası aykırı değerleri görmeye yarar.
            """
        )

        raw = st.text_area("Sayıları girin", "5,7,8,3,10,12,6,9,11,15,18,21", height=100, key="b2_box_raw")
        try:
            data = parse_numeric_input(raw)
            q1, q2, q3, iqr, _ = quartiles_value(data)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Ölçü": ["Minimum", "Q1", "Medyan", "Q3", "Maksimum", "IQR"],
                        "Değer": [
                            round(np.min(data), 2),
                            round(q1, 2),
                            round(q2, 2),
                            round(q3, 2),
                            round(np.max(data), 2),
                            round(iqr, 2),
                        ],
                    }
                ),
                use_container_width=True,
            )
            st.pyplot(plot_boxplot(data))
        except ValueError as err:
            st.error(str(err))

    elif page == "9. Ortalama Sapma":
        st.header("Ortalama Sapma")
        st.markdown(
            """
Ortalama sapma, verilerin ortalamadan mutlak uzaklıklarının ortalamasıdır.

Formül:
**OS = Σ|xᵢ - x̄| / n**
            """
        )

        raw = st.text_area("Sayıları girin", "4,6,8,10,12", height=100, key="b2_mad_raw")
        try:
            data = parse_numeric_input(raw)
            mad = mean_absolute_deviation_value(data)
            mean_val = arithmetic_mean(data)
            detail_df = pd.DataFrame(
                {
                    "Değer": data,
                    "|x - Ortalama|": np.round(np.abs(data - mean_val), 4),
                }
            )
            st.write(f"Ortalama = {mean_val:.4f}")
            st.dataframe(detail_df, use_container_width=True)
            st.success(f"Ortalama Sapma = {mad:.4f}")
        except ValueError as err:
            st.error(str(err))

    elif page == "10. Varyans ve Standart Sapma":
        st.header("Varyans ve Standart Sapma")
        st.markdown(
            """
Varyans, verilerin ortalamadan sapmalarının karelerinin ortalamasıdır.
Standart sapma ise varyansın kareköküdür.
            """
        )

        raw = st.text_area("Sayıları girin", "4,6,8,10,12", height=100, key="b2_var_raw")
        try:
            data = parse_numeric_input(raw)
            vals = variance_std_value(data)
            table = pd.DataFrame(
                {
                    "Ölçü": [
                        "Anakütle Varyansı",
                        "Örnek Varyansı",
                        "Anakütle Std. Sapma",
                        "Örnek Std. Sapma",
                    ],
                    "Değer": [
                        round(vals["variance_pop"], 4),
                        round(vals["variance_sample"], 4),
                        round(vals["std_pop"], 4),
                        round(vals["std_sample"], 4),
                    ],
                }
            )
            st.dataframe(table, use_container_width=True)
        except ValueError as err:
            st.error(str(err))

    elif page == "11. Değişim Katsayısı ve Çarpıklık":
        st.header("Değişim Katsayısı ve Çarpıklık")
        raw = st.text_area("Sayıları girin", "5,7,8,3,10,12,6,9,11,15,18,21", height=100, key="b2_cv_raw")
        try:
            data = parse_numeric_input(raw)
            cv = coefficient_of_variation_value(data)
            pearson_sk = pearson_skewness_value(data)
            bowley_sk = bowley_skewness_value(data)

            c1, c2, c3 = st.columns(3)
            c1.metric("Değişim Katsayısı (%)", f"{cv:.2f}")
            c2.metric("Pearson Çarpıklık", f"{pearson_sk:.4f}")
            c3.metric("Bowley Çarpıklık", f"{bowley_sk:.4f}")

            st.info("Pearson yorumu: " + skewness_comment(pearson_sk))
            st.info("Bowley yorumu: " + skewness_comment(bowley_sk))
        except ValueError as err:
            st.error(str(err))

    elif page == "12. Çözümlü Örnek Sorular":
        st.header("Çözümlü Örnek Sorular")

        with st.expander("Soru 1 - Ortalama, medyan ve mod"):
            st.write("Aşağıdaki veri için ortalama, medyan ve modu bulunuz:")
            st.code("2, 3, 3, 4, 5, 5, 5, 6, 7, 8")
            if st.button("Bölüm 2 Soru 1 çözümünü göster"):
                data = parse_numeric_input("2,3,3,4,5,5,5,6,7,8")
                st.write(f"Ortalama = {arithmetic_mean(data):.2f}")
                st.write(f"Medyan = {median_value(data):.2f}")
                st.write("Mod = " + ", ".join(f"{m:.2f}" for m in mode_value(data)))

        with st.expander("Soru 2 - Geometrik ve harmonik ortalama"):
            st.write("Aşağıdaki veri için geometrik ve harmonik ortalamayı bulunuz:")
            st.code("2, 4, 8, 16")
            if st.button("Bölüm 2 Soru 2 çözümünü göster"):
                data = parse_numeric_input("2,4,8,16")
                st.write(f"Geometrik Ortalama = {geometric_mean_value(data):.4f}")
                st.write(f"Harmonik Ortalama = {harmonic_mean_value(data):.4f}")

        with st.expander("Soru 3 - Varyans, standart sapma ve kutu grafiği"):
            st.write("Aşağıdaki veri için yayılım ölçülerini inceleyiniz:")
            st.code("5,7,8,3,10,12,6,9,11,15,18,21")
            if st.button("Bölüm 2 Soru 3 çözümünü göster"):
                data = parse_numeric_input("5,7,8,3,10,12,6,9,11,15,18,21")
                vals = variance_std_value(data)
                q1, q2, q3, iqr, _ = quartiles_value(data)
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Ölçü": ["Q1", "Medyan", "Q3", "IQR", "Örnek Varyans", "Örnek Std. Sapma"],
                            "Değer": [
                                round(q1, 2),
                                round(q2, 2),
                                round(q3, 2),
                                round(iqr, 2),
                                round(vals["variance_sample"], 2),
                                round(vals["std_sample"], 2),
                            ],
                        }
                    ),
                    use_container_width=True,
                )
                st.pyplot(plot_boxplot(data))

    elif page == "13. Mini Quiz":
        st.header("Mini Quiz")
        score = 0

        q1 = st.radio(
            "1) En büyük ve en küçük değer farkı hangi ölçüdür?",
            ["Mod", "Açıklık", "Medyan"],
            key="b2_q1",
        )
        if q1 == "Açıklık":
            score += 1

        q2 = st.radio(
            "2) Aşağıdakilerden hangisi merkezi eğilim ölçüsüdür?",
            ["Standart sapma", "Varyans", "Aritmetik ortalama"],
            key="b2_q2",
        )
        if q2 == "Aritmetik ortalama":
            score += 1

        q3 = st.radio(
            "3) Hız ve oran türü problemler için hangi ortalama uygundur?",
            ["Harmonik Ortalama", "Mod", "Medyan"],
            key="b2_q3",
        )
        if q3 == "Harmonik Ortalama":
            score += 1

        q4 = st.radio(
            "4) Birikimli değil, yayılım ölçüsü olan seçenek hangisidir?",
            ["Varyans", "Mod", "Medyan"],
            key="b2_q4",
        )
        if q4 == "Varyans":
            score += 1

        if st.button("Bölüm 2 quiz sonucunu hesapla", key="b2_quiz_button"):
            st.subheader(f"Puan: {score} / 4")
            if score == 4:
                st.success("Harika. Bölüm 2 kavramlarını çok iyi anlamışsın.")
            elif score >= 2:
                st.info("İyi gidiyorsun. Birkaç ölçüyü tekrar etmen faydalı olur.")
            else:
                st.warning("Bölüm 2 konu anlatımı ve örnek soruları tekrar incele.")

    elif page == "14. Öğrenci Veri Laboratuvarı":
        st.header("Öğrenci Veri Laboratuvarı")
        st.write("Kendi sayısal verinizi girip merkezsel eğilim ve yayılım ölçülerini hesaplayabilirsiniz.")

        raw = st.text_area(
            "Sayıları girin",
            "5,7,8,3,10,12,6,9,11,15,18,21",
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
            pearson_sk = pearson_skewness_value(data)
            bowley_sk = bowley_skewness_value(data)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ortalama", f"{mean_val:.2f}")
            c2.metric("Medyan", f"{med_val:.2f}")
            c3.metric("Örnek Std. Sapma", f"{var_std['std_sample']:.2f}")
            c4.metric("Açıklık", f"{range_value(data):.2f}")

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
                        "Örnek Varyans",
                        "Örnek Standart Sapma",
                        "Değişim Katsayısı (%)",
                        "Pearson Çarpıklık",
                        "Bowley Çarpıklık",
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
                        round(var_std["variance_sample"], 4),
                        round(var_std["std_sample"], 4),
                        round(cv, 4),
                        round(pearson_sk, 4),
                        round(bowley_sk, 4),
                    ],
                }
            )
            st.dataframe(summary_df, use_container_width=True)

            g1, g2 = st.columns(2)
            with g1:
                st.pyplot(plot_boxplot(data))
            with g2:
                grouped_df, edges, _ = grouped_frequency_table(data, 5)
                st.pyplot(plot_hist(data, edges))

        except ValueError as err:
            st.error(str(err))