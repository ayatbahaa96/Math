import math
import streamlit as st

# -----------------------------
# Matematiksel Çekirdek
# -----------------------------
def nPr(n, r):
    return math.perm(n, r) if n >= r else 0

def nCr(n, r):
    return math.comb(n, r) if n >= r else 0

def render_bolum3():
    st.title("📚 Bölüm 3: Olasılık, Permütasyon ve Bayes")
    
    # PDF Akışına Göre Navigasyon
    menu = [
        "3.1 & 3.2 Olasılık ve Örnek Uzay",
        "3.3 Sayma Teknikleri (Toplama/Çarpım)",
        "3.4 Permütasyon (Diziliş)",
        "3.5 Kombinasyon (Seçme)",
        "3.6 Olasılık Kuralları",
        "3.7 Koşullu Olasılık ve Bağımsızlık",
        "3.8 Bayes Kuralı",
        "📝 Bölüm Sonu Alıştırmaları (10 Soru)"
    ]
    
    choice = st.sidebar.radio("Alt Başlık Seçin", menu)

    # --- 3.1 & 3.2 ---
    if choice == "3.1 & 3.2 Olasılık ve Örnek Uzay":
        st.header("3.1 Olasılığa Giriş")
        st.write("Olasılık, belirsiz sonuçları olan olayların ölçülmesidir.")
        st.subheader("3.2 Deney ve Örnek Uzay")
        st.info("**Örnek Uzay (S):** Deneyin tüm sonuçlarının kümesidir.")
        st.latex(r"S = \{s_1, s_2, ..., s_n\}")
        
        st.markdown("**Örnek:** İki madeni para atıldığında:")
        st.code("S = {YY, YT, TY, TT}")

    # --- 3.3 ---
    elif choice == "3.3 Sayma Teknikleri (Toplama/Çarpım)":
        st.header("3.3 Örnek Noktalarını Sayma")
        
        st.subheader("1. Toplama Kuralı")
        st.write("Ayrık olaylarda 'veya' bağlacı kullanılır: $n_1 + n_2$")
        with st.expander("📝 Örnek 1"):
            st.write("3 mavi, 4 kırmızı kalem arasından 1 kalem kaç yolla seçilir?")
            st.success("Çözüm: 3 + 4 = 7")

        st.subheader("2. Çarpım Kuralı")
        st.write("Ardışık olaylarda 've' bağlacı kullanılır: $n_1 \\times n_2$")
        with st.expander("📝 Örnek 2"):
            st.write("3 pantolonu ve 5 gömleği olan bir kişi kaç farklı takım giyebilir?")
            st.success("Çözüm: 3 x 5 = 15")

    # --- 3.4 ---
    elif choice == "3.4 Permütasyon (Diziliş)":
        st.header("3.4 Permütasyon")
        st.latex(r"P(n, r) = \frac{n!}{(n-r)!}")
        
        with st.expander("📝 Örnek 1: Yarışma"):
            st.write("10 kişi arasından ilk 3 derece (1., 2., 3.) kaç yolla oluşur?")
            st.success(f"Çözüm: P(10, 3) = {nPr(10, 3)}")

        with st.expander("📝 Örnek 2: Tekrarlı Permütasyon"):
            st.write("'İSTATİSTİK' kelimesindeki harflerle kaç farklı diziliş yapılır?")
            # 9 harf: İ:2, S:2, T:3, A:1, K:1
            res = math.factorial(9) // (math.factorial(2)*math.factorial(2)*math.factorial(3))
            st.success(f"Çözüm: 9! / (2! * 2! * 3!) = {res}")

    # --- 3.5 ---
    elif choice == "3.5 Kombinasyon (Seçme)":
        st.header("3.5 Kombinasyon")
        st.latex(r"C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}")
        
        with st.expander("📝 Örnek 1: Ekip"):
            st.write("8 kişi arasından 3 kişilik bir ekip kaç yolla seçilir?")
            st.success(f"Çözüm: C(8, 3) = {nCr(8, 3)}")

    # --- 3.6 ---
    elif choice == "3.6 Olasılık Kuralları":
        st.header("3.6 Olasılık Kuralları")
        st.write("1. $0 \le P(A) \le 1$")
        st.write("2. $P(A \cup B) = P(A) + P(B) - P(A \cap B)$")

        with st.expander("📝 Örnek 1: Zar"):
            st.write("Bir zar atıldığında sayının çift veya asal olma olasılığı?")
            st.write("Çift: {2,4,6}, Asal: {2,3,5}, Kesişim: {2}")
            st.success("Çözüm: (3/6) + (3/6) - (1/6) = 5/6")

    # --- 3.7 ---
    elif choice == "3.7 Koşullu Olasılık ve Bağımsızlık":
        st.header("3.7 Koşullu Olasılık")
        st.latex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}")
        
        with st.expander("📝 Örnek 1"):
            st.write("Toplamın 8 olduğu bilindiğine göre iki zarın da çift olma olasılığı?")
            st.write("B = {(2,6), (3,5), (4,4), (5,3), (6,2)} (5 durum)")
            st.write("A ∩ B = {(2,6), (4,4), (6,2)} (3 durum)")
            st.success("Sonuç: 3 / 5 = 0.60")

    # --- 3.8 ---
    elif choice == "3.8 Bayes Kuralı":
        st.header("3.8 Bayes Kuralı")
        st.write("Bilinen sonuçtan nedeni tahmin etme.")
        st.latex(r"P(A|B) = \frac{P(A)P(B|A)}{P(B)}")
        
        st.info("Aşağıdaki hesaplayıcıyı kullanarak Bayes denemesi yapabilirsiniz:")
        pa = st.number_input(
            "Hastalık Oranı P(A)",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            format="%.4f",
        )
        pb_a = st.number_input(
            "Hastayken Testin Pozitif Çıkma Olasılığı P(B|A)",
            min_value=0.0,
            max_value=1.0,
            value=0.99,
            step=0.01,
            format="%.4f",
        )
        pb_not_a = st.number_input(
            "Sağlıklıyken Testin Pozitif Çıkma Olasılığı P(B|A')",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.4f",
        )
        
        pb = (pa * pb_a) + ((1-pa) * pb_not_a)
        result = (pa * pb_a) / pb
        st.warning(f"Testi Pozitif Çıkan Birinin Hasta Olma Olasılığı: {result:.4f}")

    # --- BÖLÜM SONU ---
    elif choice == "📝 Bölüm Sonu Alıştırmaları (10 Soru)":
        st.header("Bölüm Sonu Soruları (Çözümlü)")
        
        q_list = [f"Soru {i}" for i in range(1, 11)]
        selected_q = st.selectbox("Soru Seçin", q_list)

        if selected_q == "Soru 1":
            st.write("3 madeni para atılıyor. En az 2'sinin Yazı gelme olasılığı?")
            if st.button("Çözümü Göster"):
                st.success("n(S)=8, Yazı durumları: {YYY, YYT, YTY, TYY} -> 4/8 = 0.5")
        
        elif selected_q == "Soru 2":
            st.write("Bir zar ve bir para atılıyor. Paranın Yazı ve zarın 4'ten büyük gelme olasılığı?")
            if st.button("Çözümü Göster"):
                st.success("P(Y) = 1/2, P(>4) = 2/6. Bağımsız: (1/2)*(2/6) = 1/6")

        elif selected_q == "Soru 3":
            st.write("52'lik desteden bir kart çekiliyor. As veya Kırmızı olma olasılığı?")
            if st.button("Çözümü Göster"):
                st.success("P(As)=4/52, P(Kırmızı)=26/52, P(Kırmızı As)=2/52. Sonuç: (4+26-2)/52 = 28/52")

        elif selected_q == "Soru 4":
            st.write("4 Matematik, 3 Fizik kitabı bir rafa matematikler yanyana olmak üzere kaç yolla dizilir?")
            if st.button("Çözümü Göster"):
                st.success("Matematikleri 1 blok say: 4! (blok içi) * 4! (blok + fizik kitapları) = 24 * 24 = 576")

        elif selected_q == "Soru 5":
            st.write("10 kişilik sınıftan bir başkan ve bir başkan yardımcısı kaç yolla seçilir?")
            if st.button("Çözümü Göster"):
                st.success(f"P(10, 2) = {nPr(10, 2)}")

        elif selected_q == "Soru 6":
            st.write("Bir torbada 5 kırmızı, 4 beyaz bilye var. Çekilen bilye geri atılmaksızın 2 bilye çekiliyor. İkisinin de kırmızı olma olasılığı?")
            if st.button("Çözümü Göster"):
                st.success("(5/9) * (4/8) = 20/72 = 5/18")

        elif selected_q == "Soru 7":
            st.write("A torbası (3K, 2B), B torbası (4K, 5B). Bir torba seçilip bir top çekiliyor. Kırmızı olma olasılığı?")
            if st.button("Çözümü Göster"):
                st.success("P(K) = (1/2 * 3/5) + (1/2 * 4/9) = 3/10 + 2/9 = 47/90")

        elif selected_q == "Soru 8":
            st.write("**PDF Soru 22:** A(%50), B(%30), C(%20) makineleri. Hata oranları %1, %2, %3. Arızalıysa C'den olma olasılığı?")
            if st.button("Çözümü Göster"):
                p_hata = (0.50*0.01) + (0.30*0.02) + (0.20*0.03)
                ans = (0.20*0.03) / p_hata
                st.success(f"Sonuç: {ans:.4f}")

        elif selected_q == "Soru 9":
            st.write("**PDF Soru 23:** A(5S, 4M), B(9S, 6M). A'dan B'ye top atılıyor. B'den çekilenin mor olma olasılığı?")
            if st.button("Çözümü Göster"):
                res = (4/9 * 7/16) + (5/9 * 6/16)
                st.success(f"Sonuç: {res:.4f}")

        elif selected_q == "Soru 10":
            st.write("Bir hedefe yapılan atışın vurulma olasılığı 1/3'tür. 3 atışta hedefin en az bir kez vurulma olasılığı?")
            if st.button("Çözümü Göster"):
                st.success("1 - (Vuramama)^3 = 1 - (2/3)^3 = 1 - 8/27 = 19/27")