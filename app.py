import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="TexSplit - AI Renk Ayrımı", layout="wide")

st.title("TexSplit v1.1 (Şeffaf Mod)")
st.write("Tekstil desenlerini renklerine ayır ve ŞEFFAF arka planlı olarak indir.")

# --- KENAR ÇUBUĞU ---
st.sidebar.header("Ayarlar")
uploaded_file = st.sidebar.file_uploader("Deseni Yükle", type=['jpg', 'jpeg', 'png'])
k_colors = st.sidebar.slider("Renk Sayısı", 2, 12, 6)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, channels="BGR", caption="Orijinal", use_container_width=True)

    if st.sidebar.button("Renkleri Ayrıştır"):
        with st.spinner('Şeffaf katmanlar hazırlanıyor...'):
            
            # --- İŞLEM ---
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Z = img_rgb.reshape((-1, 3))
            Z = np.float32(Z)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(Z, k_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            labels = labels.reshape(img.shape[:2])
            
            st.success("Ayrıştırma Tamamlandı! Görseller artık şeffaf.")
            
            # Önizleme
            quantized_img = centers[labels.flatten()].reshape(img_rgb.shape)
            with col2:
                st.image(quantized_img, caption="AI Renk Önizlemesi", use_container_width=True)

            st.write("---")
            st.subheader("Şeffaf Katmanlar (Transparent Layers)")

            cols = st.columns(4) 
            h, w = img.shape[:2]

            for i in range(k_colors):
                # --- KRİTİK NOKTA: ŞEFFAFLIK AYARI ---
                # 4 Kanallı (RGBA) boş bir resim oluştur
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                
                # Maskeyi al (Hangi pikseller bu renge ait?)
                mask = (labels == i)
                
                # Rengi al
                r, g, b = centers[i]
                
                # RGB kanallarını boya
                rgba[mask, 0] = r # Red
                rgba[mask, 1] = g # Green
                rgba[mask, 2] = b # Blue
                
                # Alpha (Şeffaflık) kanalını ayarla
                # Maskenin olduğu yerler 255 (Görünür), olmayan yerler 0 (Görünmez)
                rgba[mask, 3] = 255 
                
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                
                # PNG olarak kodla (Şeffaflığı korur)
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                byte_im = buffer.tobytes()

                with cols[i % 4]:
                    st.color_picker(f"Renk {i+1}", hex_color, disabled=True, key=f"c_{i}")
                    # Ekranda görünmesi için
                    st.image(rgba, caption=f"Katman {i+1}", use_container_width=True)
                    
                    st.download_button(
                        label=f"İndir (Şeffaf PNG)",
                        data=byte_im,
                        file_name=f"TexSplit_Katman_{i+1}_{hex_color}.png",
                        mime="image/png"
                    )
