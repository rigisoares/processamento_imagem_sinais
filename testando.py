import cv2
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from sklearn.metrics import classification_report

# --- 1. CONFIGURAÇÃO E CARREGAMENTO DO MODELO REFINADO ---
caminho_base = Path(r'C:\Users\User\Documents\Tecnologo IA\IMAGEM\PROJETO')
pasta_teste = caminho_base / 'ARQUIVOS TESTE'
csv_referencia_teste = pasta_teste / 'lista_teste.csv'

try:
    modelo = joblib.load('modelo_mamografia.joblib')
    print("✅ Modelo Refinado (256x256 + Entropia) carregado!")
except:
    print("❌ Erro: O arquivo 'modelo_mamografia.joblib' não foi encontrado.")
    exit()

# --- 2. FUNÇÃO DE EXTRAÇÃO REFINADA (IDÊNTICA AO TREINO) ---
def extrair_features_refinadas_teste(caminho_img):
    img = cv2.imread(str(caminho_img), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # Resolução aumentada para 256x256
    img = cv2.resize(img, (256, 256))
    
    # Filtro Bilateral
    img_clean = cv2.bilateralFilter(img, 9, 75, 75)
    
    # A. Entropia de Shannon
    ent = shannon_entropy(img_clean)
    
    # B. Textura (LBP)
    radius = 3
    n_points = 24
    lbp = local_binary_pattern(img_clean, n_points, radius, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)

    # C. Forma (Momentos de Hu)
    _, thresh = cv2.threshold(img_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.HuMoments(cv2.moments(thresh)).flatten()
    hu_log = -np.sign(moments) * np.log10(np.abs(moments) + 1e-10)

    # D. Brilho e Contraste
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_c = clahe.apply(img_clean)
    media, desvio = cv2.meanStdDev(img_c)

    # Retorna o vetor na mesma ordem exata do treino
    return np.hstack([[media[0][0], desvio[0][0], ent], hist_lbp, hu_log])

# --- 3. EXECUÇÃO DO TESTE CEGO ---
print("🔍 Iniciando teste final com alta resolução e análise de entropia...")
df_gabarito = pd.read_csv(csv_referencia_teste, sep=';', encoding='latin1')

y_real = []
y_predito = []

for _, linha in df_gabarito.iterrows():
    caminho_foto = pasta_teste / linha['Arquivo']
    features = extrair_features_refinadas_teste(caminho_foto)
    
    if features is not None:
        predicao = modelo.predict([features])[0]
        y_real.append(0 if linha['Status'] == 'benign' else 1)
        y_predito.append(predicao)

# --- 4. RELATÓRIO DE DESEMPENHO ---
print("\n" + "="*50)
print("📊 RESULTADO FINAL - REFINAMENTO (256x256 + ENTROPIA)")
print("="*50)
print(classification_report(y_real, y_predito, target_names=['Benign', 'Malignant']))