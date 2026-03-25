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
from multiprocessing import Pool
from tqdm import tqdm

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
script_dir = Path(__file__).parent
caminho_base = script_dir
pasta_treino = caminho_base / 'ARQUIVOS TREINO'
csv_referencia_treino = pasta_treino / 'lista_treino.csv'

if not csv_referencia_treino.exists():
    print(f"❌ Erro: Arquivo não encontrado: {csv_referencia_treino}")
    import sys
    sys.exit(1)

# --- 2. FUNÇÃO DE EXTRAÇÃO OTIMIZADA ---
def extrair_features_refinadas(caminho_img):
    """Extrai features de uma imagem"""
    try:
        img = cv2.imread(str(caminho_img), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
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

        features = np.hstack([[media[0][0], desvio[0][0], ent], hist_lbp, hu_log])
        return features
    except Exception as e:
        return None

def processar_linha(args):
    """Processa uma linha do CSV em paralelo"""
    idx, linha, pasta_treino = args
    caminho_img = pasta_treino / linha['Arquivo']
    status = 0 if linha['Status'] == 'benign' else 1
    
    f = extrair_features_refinadas(caminho_img)
    if f is not None:
        return (f, status, True)
    else:
        return (None, status, False)

# --- 3. PROCESSAMENTO COM PARALELIZAÇÃO ---
if __name__ == '__main__':
    print("🧠 Extraindo métricas refinadas (com paralelização)...")
    print("📂 Carregando arquivo de referência...")

    df_ref = pd.read_csv(csv_referencia_treino, sep=';', encoding='latin1')
    print(f"📊 Total de arquivos: {len(df_ref)}")

    # Limitar a 2000 imagens para treino rápido (pode aumentar depois)
    LIMITE_IMAGENS = min(2000, len(df_ref))
    df_ref = df_ref.head(LIMITE_IMAGENS)
    print(f"🎯 Usando {len(df_ref)} imagens para treino (pode aumentar depois)")

    # Preparar argumentos para processamento paralelo
    args_list = [(idx, linha, pasta_treino) for idx, (_, linha) in enumerate(df_ref.iterrows())]

    # Processar com multiprocessing
    print("⚙️ Processando imagens em paralelo...")
    with Pool(4) as pool:  # 4 processos paralelos
        resultados = list(tqdm(pool.imap(processar_linha, args_list), total=len(args_list)))

    # Coletar resultados
    X_train = []
    y_train = []
    sucesso = sum(1 for _, _, ok in resultados if ok)

    for features, status, ok in resultados:
        if ok and features is not None:
            X_train.append(features)
            y_train.append(status)

    print(f"✅ Imagens carregadas com sucesso: {len(X_train)}")
    print(f"⚠️ Imagens com erro: {len(resultados) - sucesso}")

    if len(X_train) < 10:
        print("❌ Erro: Poucas imagens carregadas! Verifique os dados.")
        import sys
        sys.exit(1)

    # --- 4. TREINAMENTO ---
    print("\n🚀 Treinando modelo...")
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"📊 Classes - Benigno: {sum(y_train==0)}, Maligno: {sum(y_train==1)}")

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=15,
        max_features='sqrt',
        class_weight={0: 1, 1: 2},  # Penalizar falsos positivos
        random_state=42,
        n_jobs=-1  # Usar todos os cores
    )
    modelo.fit(X_train, y_train)

    # --- 5. SALVAMENTO ---
    print("\n💾 Salvando modelo...")
    joblib.dump(modelo, 'modelo_avaliacao.joblib')
    print(f"✅ Modelo salvo: modelo_avaliacao.joblib")
    print(f"📊 Acurácia no treino: {modelo.score(X_train, y_train):.2%}")
    print("\n✨ Treino concluído com sucesso!")
