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
# Usar caminho relativo ao diretório do script
script_dir = Path(__file__).parent
caminho_base = script_dir
pasta_treino = caminho_base / 'ARQUIVOS TREINO'
csv_referencia_treino = pasta_treino / 'lista_treino.csv'

# Verificar se o arquivo existe
if not csv_referencia_treino.exists():
    print(f"❌ Erro: Arquivo não encontrado: {csv_referencia_treino}")
    print(f"📁 Procurando em: {caminho_base}")
    print(f"📂 Pastas disponíveis:")
    for item in caminho_base.iterdir():
        if item.is_dir():
            print(f"   - {item.name}")
    import sys
    sys.exit(1)

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

print(f"📊 Total de arquivos na lista: {len(df_ref)}")

# Separar benignos e malignos
df_benign = df_ref[df_ref['Status'] == 'benign'].reset_index(drop=True)
df_malignant = df_ref[df_ref['Status'] == 'malignant'].reset_index(drop=True)

print(f"   - Benignos disponíveis: {len(df_benign)}")
print(f"   - Malignos disponíveis: {len(df_malignant)}")

# Usar TODOS os dados (balanceados)
print(f"\n⏳ Carregando TODOS os dados balanceados...\n")

# Combinar e embaralhar
df_train = pd.concat([df_benign, df_malignant]).sample(frac=1, random_state=42).reset_index(drop=True)

X_train = []
y_train = []
erros = 0

print(f"Processando {len(df_train)} imagens...\n")

for idx, (_, linha) in enumerate(df_train.iterrows()):
    if (idx + 1) % 200 == 0:
        print(f"  [{idx+1}/{len(df_train)}] Processadas...")
    
    f = extrair_features_refinadas(pasta_treino / linha['Arquivo'])
    if f is not None:
        X_train.append(f)
        y_train.append(0 if linha['Status'] == 'benign' else 1)
    else:
        erros += 1

print(f"\n✅ Imagens carregadas com sucesso: {len(X_train)}")
print(f"⚠️ Imagens com erro: {erros}")

if len(X_train) == 0:
    print("❌ Erro: Nenhuma imagem foi carregada!")
    import sys
    sys.exit(1)

# --- 4. TREINAMENTO COM AJUSTE DE PENALIDADE ---
print("\n🚀 Treinando modelo final...")
X_train_array = np.array(X_train)
y_train_array = np.array(y_train)

print(f"📊 Distribuição de classes:")
print(f"   - Benigno: {sum(y_train_array==0)}")
print(f"   - Maligno: {sum(y_train_array==1)}")

modelo = RandomForestClassifier(
    n_estimators=300,      # Aumentado para 300 árvores (temos muitos dados)
    max_depth=12,          # Aumentado um pouco para captar padrões complexos
    min_samples_leaf=10,   # Mantém rigor contra overfitting
    max_features='sqrt',   # Força a diversidade
    class_weight='balanced',  # Balancear automaticamente
    random_state=42,
    n_jobs=-1  # Usar todos os cores
)
modelo.fit(X_train_array, y_train_array)

# --- 5. SALVAMENTO ---
print("\n💾 Salvando modelo...")
joblib.dump(modelo, 'modelo_avaliacao.joblib')
print(f"✅ Modelo salvo: modelo_avaliacao.joblib")
print(f"📊 Acurácia no treino: {modelo.score(X_train_array, y_train_array):.2%}")
print("\n✨ Treino concluído com sucesso!")