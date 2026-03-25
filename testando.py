import cv2
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from sklearn.metrics import classification_report, accuracy_score

# --- 1. CONFIGURAÇÃO E CARREGAMENTO DO MODELO TREINADO ---
script_dir = Path(__file__).parent
pasta_teste = script_dir / 'ARQUIVOS TESTE'
csv_referencia_teste = pasta_teste / 'lista_teste.csv'
caminho_modelo = script_dir / 'modelo_avaliacao.joblib'

try:
    modelo = joblib.load(str(caminho_modelo))
    print("✅ Modelo treinado (modelo_avaliacao.joblib) carregado com sucesso!")
except FileNotFoundError:
    print(f"❌ Erro: O arquivo '{caminho_modelo}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    exit()

# --- 2. FUNÇÃO DE EXTRAÇÃO DE FEATURES (IDÊNTICA AO TREINAMENTO) ---
def extrair_features_refinadas_teste(caminho_img):
    """Extrai 36 features da imagem (idêntico ao modelo_ia.py)"""
    try:
        img = cv2.imread(str(caminho_img), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Redimensionar para 256x256
        img = cv2.resize(img, (256, 256))
        
        # Filtro bilateral para limpeza
        img_clean = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 1. Entropia de Shannon
        ent = shannon_entropy(img_clean)
        
        # 2. Textura (Local Binary Pattern - 27 features)
        radius = 3
        n_points = 24
        lbp = local_binary_pattern(img_clean, n_points, radius, method="uniform")
        (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-7)
        
        # 3. Forma (Momentos de Hu - 7 features)
        _, thresh = cv2.threshold(img_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        moments = cv2.HuMoments(cv2.moments(thresh)).flatten()
        hu_log = -np.sign(moments) * np.log10(np.abs(moments) + 1e-10)
        
        # 4. Brilho e Contraste (2 features)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        img_c = clahe.apply(img_clean)
        media, desvio = cv2.meanStdDev(img_c)
        
        # Retorna vetor completo: 1 entropia + 27 LBP + 7 Hu + 2 brilho/contraste = 37 features
        # Mas estamos usando: media[0][0], desvio[0][0], ent, hist_lbp, hu_log = 2 + 1 + 27 + 7 = 37
        # Verificar se precisa ser ajustado para 36
        return np.hstack([[media[0][0], desvio[0][0], ent], hist_lbp, hu_log])
    except Exception as e:
        print(f"⚠️ Erro ao processar {caminho_img}: {e}")
        return None

# --- 3. EXECUÇÃO DO TESTE ---
print("🔍 Iniciando teste de acurácia com conjunto de teste...")
print(f"📂 Carregando dados de: {csv_referencia_teste}")

try:
    df_gabarito = pd.read_csv(csv_referencia_teste, sep=';', encoding='latin1')
except Exception as e:
    print(f"❌ Erro ao ler CSV: {e}")
    exit()

print(f"📊 Total de imagens no CSV: {len(df_gabarito)}")

y_real = []
y_predito = []
imagens_processadas = 0
imagens_com_erro = 0

for idx, (_, linha) in enumerate(df_gabarito.iterrows()):
    caminho_foto = pasta_teste / linha['Arquivo']
    
    if not caminho_foto.exists():
        print(f"⚠️ Arquivo não encontrado: {linha['Arquivo']}")
        imagens_com_erro += 1
        continue
    
    features = extrair_features_refinadas_teste(caminho_foto)
    
    if features is not None:
        try:
            predicao = modelo.predict([features])[0]
            y_real.append(0 if linha['Status'] == 'benign' else 1)
            y_predito.append(predicao)
            imagens_processadas += 1
            
            if (imagens_processadas) % 100 == 0:
                print(f"  ✓ [{imagens_processadas}/{len(df_gabarito)}] Imagens testadas...")
        except Exception as e:
            print(f"⚠️ Erro ao fazer predição para {linha['Arquivo']}: {e}")
            imagens_com_erro += 1
    else:
        imagens_com_erro += 1

# --- 4. RELATÓRIO DE DESEMPENHO ---
print("\n" + "="*60)
print("📊 RESULTADO FINAL - AVALIAÇÃO DO MODELO")
print("="*60)
print(f"✅ Imagens processadas com sucesso: {imagens_processadas}")
print(f"⚠️ Imagens com erro: {imagens_com_erro}")
print(f"📈 Total: {imagens_processadas + imagens_com_erro}")

if len(y_real) > 0:
    acuracia = accuracy_score(y_real, y_predito)
    print(f"\n🎯 ACURÁCIA NO TESTE: {acuracia*100:.2f}%")
    print("\n" + "-"*60)
    print("RELATÓRIO DETALHADO:")
    print("-"*60)
    print(classification_report(y_real, y_predito, target_names=['Benign', 'Malignant']))
else:
    print("\n❌ Nenhuma imagem foi processada com sucesso!")