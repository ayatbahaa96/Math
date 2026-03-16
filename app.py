import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Bölüm 1 - Temel İstatistik Platformu",
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
            "Sınıf Aralığı": [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" if i < len(freq) - 1 else f"[{edges[i]:.2f}, {edges[i+1]:.2f}]" for i in range(len(freq))],
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
    ax.hist(data, bins=bins)
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
# Content
# -----------------------------
st.title("📘 Bölüm 1: Temel İstatistik, Veri Düzenleme ve Grafikler")

st.sidebar.title("📚 İçindekiler")

section = st.sidebar.selectbox(
    "Bölüm Seç",
    [
        "Bölüm 1 - Temel İstatistik"
    ]
)
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
        key="q1",
    )
    if q1 == "Nitel-kategorik":
        score += 1

    q2 = st.radio(
        "2) Hangisi oranlama düzeyindedir?",
        ["Takvim yılı", "Sıcaklık (Celsius)", "Ağırlık"],
        key="q2",
    )
    if q2 == "Ağırlık":
        score += 1

    q3 = st.radio(
        "3) Sürekli nicel veri için en uygun grafik hangisidir?",
        ["Histogram", "Daire grafik", "Sadece pasta grafik"],
        key="q3",
    )
    if q3 == "Histogram":
        score += 1

    q4 = st.radio(
        "4) Birikimli frekansı izlemek için hangi grafik kullanılır?",
        ["Ogive", "Çubuk grafik", "Dal-yaprak"],
        key="q4",
    )
    if q4 == "Ogive":
        score += 1

    if st.button("Quiz sonucunu hesapla"):
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
