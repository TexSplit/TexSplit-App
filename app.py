import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="TexSplit - AI Renk Ayrımı", layout="wide")

st.title("TexSplit v1.0")
st.write("Tekstil desenlerini yapay zeka ile renk katmanlarına (Spot Channels) ayır.")

# --- KENAR ÇUBUĞU (AYARLAR) ---
st.sidebar.header("Ayarlar")
uploaded_file = st.sidebar.file_uploader("Deseni Yükle (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
k_colors = st.sidebar.slider("Renk Sayısı (Hedef)", min_value=2, max_value=12, value=6)

# --- ANA MOTOR ---
if uploaded_file is not None:
    # Dosyayı belleğe yükle
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Orijinal görüntüyü göster
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, channels="BGR", caption="Orijinal Desen", use_container_width=True)

    # "AYIR" BUTONU
    if st.sidebar.button("Renkleri Ayrıştır"):
        with st.spinner('Yapay zeka renkleri analiz ediyor...'):
            
            # --- RENK AYRIM ALGORİTMASI ---
            # BGR'den RGB'ye çevir
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Görüntüyü düzleştir
            Z = img_rgb.reshape((-1, 3))
            Z = np.float32(Z)

            # K-Means Çalıştır
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(Z, k_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            labels = labels.reshape(img.shape[:2])

            # --- SONUÇLARI GÖSTERME ---
            st.success("İşlem Tamamlandı! Katmanlar aşağıda.")
            
            # İndirgenmiş görseli göster
            quantized_img = centers[labels.flatten()].reshape(img_rgb.shape)
            with col2:
                st.image(quantized_img, caption=f"AI Sonucu ({k_colors} Renk)", use_container_width=True)

            st.write("---")
            st.subheader("Ayrıştırılmış Katmanlar (Baskı Kalıpları)")

            # Yan yana göstermek için kolonlar
            cols = st.columns(4) 
            
            for i in range(k_colors):
                # Maske oluştur
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask[labels == i] = 255
                
                # Rengi al (Hex kodu için)
                r, g, b = centers[i]
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                
                # İndirme işlemi için belleğe kaydet
                is_success, buffer = cv2.imencode(".png", mask)
                byte_im = buffer.tobytes()

                # Ekrana bas
                with cols[i % 4]:
                    st.color_picker(f"Renk {i+1}", hex_color, disabled=True, key=f"c_{i}")
                    st.image(mask, caption=f"Katman {i+1}", clamp=True, use_container_width=True)
                    
                    st.download_button(
                        label=f"İndir (Katman {i+1})",
                        data=byte_im,
                        file_name=f"TexSplit_Katman_{i+1}.png",
                        mime="image/png"
                    )

    else:
        st.info("Ayarları yap ve 'Renkleri Ayrıştır' butonuna bas.")

else:
    st.warning("Lütfen sol menüden bir desen yükleyin.")
