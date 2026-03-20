import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from PIL import Image

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="IA de Diagnóstico Sobre Tireoide", layout="centered")

st.title("🏥 Sistema de Triagem Inteligente")
st.markdown("""
Esta interface utiliza o modelo de **Random Forest** treinado com métricas de **Entropia, LBP e Momentos de Hu** para analisar imagens de tireoide e classificar o status.
""")

# --- CARREGAR O MODELO ---
@st.cache_resource
def carregar_modelo():
    return joblib.load('modelo_avaliacao.joblib')

modelo = carregar_modelo()

# --- FUNÇÃO DE EXTRAÇÃO (MESMA LÓGICA DO TREINO/TESTE) ---
def extrair_features_usuario(imagem_pil):
    # Converter PIL para OpenCV
    img = np.array(imagem_pil.convert('L'))
    img = cv2.resize(img, (256, 256))
    
    img_clean = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Entropia
    ent = shannon_entropy(img_clean)
    
    # LBP
    radius = 3
    n_points = 24
    lbp = local_binary_pattern(img_clean, n_points, radius, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)

    # Momentos de Hu
    _, thresh = cv2.threshold(img_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.HuMoments(cv2.moments(thresh)).flatten()
    hu_log = -np.sign(moments) * np.log10(np.abs(moments) + 1e-10)

    # Brilho e Contraste
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_c = clahe.apply(img_clean)
    media, desvio = cv2.meanStdDev(img_c)

    return np.hstack([[media[0][0], desvio[0][0], ent], hist_lbp, hu_log])

# --- INTERFACE DE UPLOAD ---
arquivo_upload = st.file_uploader("Selecione a imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])

if arquivo_upload is not None:
    # Exibir a imagem original
    img_original = Image.open(arquivo_upload)
    st.image(img_original, caption="Imagem Carregada", use_container_width=True)
    
    if st.button("Analisar Imagem"):
        with st.spinner('A IA está analisando a morfologia e textura...'):
            # Processar
            features = extrair_features_usuario(img_original)
            
            # Predição
            predicao = modelo.predict([features])[0]
            probabilidade = modelo.predict_proba([features])
            
            # Resultado
            resultado = "MALIGNO" if predicao == 1 else "BENIGNO"
            cor = "#e74c3c" if resultado == "MALIGNO" else "#27ae60"
            
            st.markdown(f"<h2 style='text-align: center; color: {cor};'>Resultado: {resultado}</h2>", unsafe_allow_html=True)
            
            # Detalhes Técnicos para o Usuário
            st.info(f"Confiança da IA: {np.max(probabilidade)*100:.2f}%")
            
            with st.expander("Ver métricas detalhadas"):
                st.write(f"**Entropia (Desordem):** {features[2]:.4f}")
                st.write(f"**Média de Brilho:** {features[0]:.2f}")
                st.write("**Histograma LBP e Momentos de Hu calculados.**")
#streamlit run app.py
st.sidebar.info("Projeto Tecnólogo em IA - Sistema de Classificação de Imagens Médicas.")