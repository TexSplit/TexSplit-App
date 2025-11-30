import streamlit as st
import cv2
import numpy as np
import io
import zipfile

# --- SAYFA YAPISI ---
st.set_page_config(page_title="TexSplit Pro", layout="wide")
st.title("TexSplit v1.2 (Pro Mod)")
st.write("Renk kodlu, şeffaf ve ZIP indirme özellikli sürüm.")

# --- KENAR ÇUBUĞU ---
st.sidebar.header("Ayarlar")
uploaded_file = st.sidebar.file_uploader("Deseni Yükle", type=['jpg', 'jpeg', 'png'])
k_colors = st.sidebar.slider("Renk Sayısı", 2, 12, 6)

# --- FONKSİYONLAR (Hafızada tutmak için) ---
def process_image(img, k):
    # Görüntüyü işle
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Z = img_rgb.reshape((-1, 3))
    Z = np.float32(Z)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    labels = labels.reshape(img.shape[:2])
    return labels, centers, img_rgb

def create_transparent_layer(img_shape, labels, center_color, index):
    h, w = img_shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask = (labels == index)
    
    r, g, b = center_color
    rgba[mask, 0] = r
    rgba[mask, 1] = g
    rgba[mask, 2] = b
    rgba[mask, 3] = 255 # Görünür yap
    
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

# --- ANA AKIŞ ---
if uploaded_file:
    # Dosyayı oku
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Session State (Hafıza) Kontrolü
    # Eğer yeni bir dosya yüklendiyse hafızayı temizle
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.processed = False
        st.session_state.last_uploaded = uploaded_file.name

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, channels="BGR", caption="Orijinal", use_container_width=True)

    # BUTON ve İŞLEM
    if st.sidebar.button("Renkleri Ayrıştır") or st.session_state.get('processed'):
        
        if not st.session_state.get('processed'):
            with st.spinner('Analiz yapılıyor...'):
                labels, centers, img_rgb = process_image(img, k_colors)
                # Sonuçları hafızaya at
                st.session_state.labels = labels
                st.session_state.centers = centers
                st.session_state.img_rgb = img_rgb
                st.session_state.processed = True
        
        # Hafızadan verileri çek
        labels = st.session_state.labels
        centers = st.session_state.centers
        img_rgb = st.session_state.img_rgb
        
        # Önizleme
        quantized = centers[labels.flatten()].reshape(img_rgb.shape)
        with col2:
            st.image(quantized, caption=f"AI Sonucu ({k_colors} Renk)", use_container_width=True)
        
        st.success("İşlem Tamam. Katmanlar aşağıda.")
        st.write("---")

        # --- ZIP DOSYASI HAZIRLIĞI ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            
            # Katmanları Döngüye Sok
            cols = st.columns(4)
            h, w = img.shape[:2]
            
            for i in range(len(centers)):
                r, g, b = centers[i]
                hex_code = f"{r:02x}{g:02x}{b:02x}" # Hex kodunu hesapla
                
                # Şeffaf katmanı oluştur
                layer_img = create_transparent_layer((h, w), labels, centers[i], i)
                
                # Belleğe kaydet (PNG formatında)
                is_success, buffer = cv2.imencode(".png", layer_img)
                byte_im = buffer.tobytes()
                
                # Dosya adı: Katman_1_ff0000.png
                file_name = f"Katman_{i+1}_Renk_{hex_code}.png"
                
                # ZIP'e ekle
                zf.writestr(file_name, byte_im)
                
                # Ekrana bas
                with cols[i % 4]:
                    st.color_picker(f"Renk {i+1}", f"#{hex_code}", disabled=True, key=f"c_{i}")
                    st.image(layer_img, caption=file_name, use_container_width=True)
                    
                    st.download_button(
                        label="⬇️ İndir",
                        data=byte_im,
                        file_name=file_name,
                        mime="image/png"
                    )

        # --- ZIP İNDİRME BUTONU (EN ALTTA) ---
        st.write("---")
        st.subheader("📦 Toplu İndirme")
        st.download_button(
            label="TÜM KATMANLARI ZIP OLARAK İNDİR",
            data=zip_buffer.getvalue(),
            file_name="TexSplit_Tum_Katmanlar.zip",
            mime="application/zip",
            type="primary" # Butonu vurgular
        )
