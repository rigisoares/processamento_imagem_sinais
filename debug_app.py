"""
Script de debug para verificar o ambiente e o modelo
"""
import os
import sys

print("=" * 60)
print("VERIFICAÇÃO DE AMBIENTE")
print("=" * 60)

# 1. Verificar diretório
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"✓ Diretório do script: {script_dir}")

# 2. Verificar arquivo do modelo
caminho_modelo = os.path.join(script_dir, 'modelo_avaliacao.joblib')
print(f"✓ Procurando modelo em: {caminho_modelo}")
print(f"✓ Arquivo existe? {os.path.exists(caminho_modelo)}")

if os.path.exists(caminho_modelo):
    tamanho = os.path.getsize(caminho_modelo)
    print(f"✓ Tamanho do arquivo: {tamanho} bytes")

# 3. Verificar dependências
print("\n" + "=" * 60)
print("VERIFICAÇÃO DE DEPENDÊNCIAS")
print("=" * 60)

deps = ['numpy', 'cv2', 'streamlit', 'joblib', 'skimage', 'PIL']
for dep in deps:
    try:
        __import__(dep)
        print(f"✓ {dep}: OK")
    except ImportError:
        print(f"✗ {dep}: NÃO INSTALADO")

# 4. Tentar carregar o modelo
print("\n" + "=" * 60)
print("VERIFICAÇÃO DO MODELO")
print("=" * 60)

try:
    import joblib
    modelo = joblib.load(caminho_modelo)
    print(f"✓ Modelo carregado com sucesso")
    print(f"✓ Tipo do modelo: {type(modelo)}")
    print(f"✓ Tem método 'predict'? {hasattr(modelo, 'predict')}")
    print(f"✓ Tem método 'predict_proba'? {hasattr(modelo, 'predict_proba')}")
    
    # Tentar pegar informações do modelo
    if hasattr(modelo, 'n_features_in_'):
        print(f"✓ Número de features esperadas: {modelo.n_features_in_}")
    
except Exception as e:
    print(f"✗ Erro ao carregar modelo: {e}")

print("\n" + "=" * 60)
print("Tudo pronto? Execute: streamlit run app.py")
print("=" * 60)
