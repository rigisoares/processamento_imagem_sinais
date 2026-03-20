import cv2
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
caminho_base = Path(r'C:\Users\User\Documents\Tecnologo IA\IMAGEM\PROJETO')
pasta_treino = caminho_base / 'ARQUIVOS TREINO'
csv_referencia_treino = pasta_treino / 'lista_treino.csv'

# --- 2. FUNÇÃO DE EXTRAÇÃO REFINADA (RESOLUÇÃO 256 + ENTROPIA) ---
def extrair_features_refinadas(caminho_img):
    img = cv2.imread(str(caminho_img), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # Aumentando resolução para captar detalhes morfológicos finos
    img = cv2.resize(img, (256, 256))
    
    # Filtro Bilateral para limpeza preservando bordas
    img_clean = cv2.bilateralFilter(img, 9, 75, 75)
    
    # A. Entropia (Mede a complexidade/desordem do tecido)
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

    # D. Brilho e Contraste Local
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_c = clahe.apply(img_clean)
    media, desvio = cv2.meanStdDev(img_c)

    return np.hstack([[media[0][0], desvio[0][0], ent], hist_lbp, hu_log])

# --- 3. PROCESSAMENTO ---
print("🧠 Extraindo métricas refinadas (256x256 + Entropia)...")
df_ref = pd.read_csv(csv_referencia_treino, sep=';', encoding='latin1')

X_train = []
y_train = []

for _, linha in df_ref.iterrows():
    f = extrair_features_refinadas(pasta_treino / linha['Arquivo'])
    if f is not None:
        X_train.append(f)
        y_train.append(0 if linha['Status'] == 'benign' else 1)

# --- 4. TREINAMENTO COM AJUSTE DE PENALIDADE ---
print("🚀 Treinando modelo final...")
modelo = RandomForestClassifier(
    n_estimators=1200,   # Aumentado para maior estabilidade
    max_depth=14,        # Leve aumento para permitir aprender a nova complexidade
    min_samples_leaf=8,
    max_features='sqrt', # Força a diversidade entre as árvores
    class_weight='balanced',
    random_state=42
)
modelo.fit(X_train, y_train)

# --- 5. SALVAMENTO ---
joblib.dump(modelo, 'modelo_mamografia.joblib')
print("✅ Treino concluído com sucesso!")