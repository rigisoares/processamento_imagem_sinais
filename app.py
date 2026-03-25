import streamlit as st
import cv2
import numpy as np
import joblib
import os
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
    # Obter o diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Caminho para o modelo no mesmo diretório
    caminho_modelo = os.path.join(script_dir, 'modelo_avaliacao.joblib')
    
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo não encontrado em: {caminho_modelo}")
    
    return joblib.load(caminho_modelo)

try:
    modelo = carregar_modelo()
    
    # Validar se o modelo tem os métodos necessários
    if not hasattr(modelo, 'predict') or not hasattr(modelo, 'predict_proba'):
        st.error("❌ Modelo inválido: não contém os métodos necessários")
        st.stop()
        
except FileNotFoundError as e:
    st.error(f"❌ {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ Erro ao carregar modelo: {str(e)}")
    st.stop()

# --- FUNÇÃO DE EXTRAÇÃO (MESMA LÓGICA DO TREINO/TESTE) ---
def extrair_features_usuario(imagem_pil):
    try:
        # Validar entrada
        if imagem_pil is None:
            raise ValueError("Imagem vazia ou inválida")
        
        # Converter PIL para OpenCV
        img = np.array(imagem_pil.convert('L'))
        
        # Validar imagem
        if img.size == 0:
            raise ValueError("Imagem com tamanho zero")
        
        # Validar se é uma imagem real (não aleatória)
        desvio_original = np.std(img)
        if desvio_original < 5:
            raise ValueError("Imagem muito uniforme - pode não ser uma imagem médica válida")
        
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

        # Combinar todas as features: [média, desvio, entropia] + hist_lbp + hu_log
        features = np.hstack([[media[0][0], desvio[0][0], ent], hist_lbp, hu_log])
        
        # Validar número de features
        if len(features) != 36:
            raise ValueError(f"Número incorreto de features. Esperado: 36, Obtido: {len(features)}")
        
        return features, ent, media[0][0], desvio[0][0]
    
    except Exception as e:
        raise RuntimeError(f"Erro ao extrair features: {str(e)}")

# --- INTERFACE DE UPLOAD ---
arquivo_upload = st.file_uploader("Selecione a imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])

if arquivo_upload is not None:
    # Exibir a imagem original
    img_original = Image.open(arquivo_upload)
    st.image(img_original, caption="Imagem Carregada", use_container_width=True)
    
    if st.button("Analisar Imagem"):
        with st.spinner('A IA está analisando a morfologia e textura...'):
            try:
                # Processar
                features, ent, media, desvio = extrair_features_usuario(img_original)
                
                # Validar dimensionalidade das features
                features_reshape = features.reshape(1, -1)
                
                # Predição
                predicao = modelo.predict(features_reshape)[0]
                probabilidade = modelo.predict_proba(features_reshape)
                
                # Resultado
                resultado = "MALIGNO" if predicao == 1 else "BENIGNO"
                confianca = np.max(probabilidade) * 100
                
                # Validar confiança mínima (70%)
                if confianca < 70:
                    cor = "#f39c12"  # Alaranjado
                    resultado = "⚠️ RESULTADO INCONCLUSIVO"
                    mensagem = f"A IA não tem confiança suficiente ({confianca:.1f}%). Recomenda-se inspeção manual."
                else:
                    cor = "#e74c3c" if predicao == 1 else "#27ae60"
                    mensagem = f"Confiança da IA: {confianca:.2f}%"
                
                st.markdown(f"<h2 style='text-align: center; color: {cor};'>Resultado: {resultado}</h2>", unsafe_allow_html=True)
                
                # Detalhes Técnicos para o Usuário
                st.info(mensagem)
                
                with st.expander("Ver métricas detalhadas"):
                    st.write(f"**Entropia (Desordem):** {ent:.4f}")
                    st.write(f"**Média de Brilho:** {media:.2f}")
                    st.write(f"**Desvio Padrão:** {desvio:.2f}")
                    st.write(f"**Total de Features Extraídas:** {len(features)}")
                    st.write("**Histograma LBP e Momentos de Hu calculados.**")
            
            except ValueError as e:
                st.error(f"❌ Erro na imagem: {str(e)}")
            except RuntimeError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Erro na análise: {str(e)}")

st.sidebar.info("Projeto Tecnólogo em IA - Sistema de Classificação de Imagens Médicas.")