import os
import pandas as pd
import shutil
from pathlib import Path

# --- 1-CRIAR 2 PASTAS NO DIRETÓRIO INDICADO ---
diretorio_base = Path(r'C:\Users\User\Documents\Tecnologo IA\IMAGEM\PROJETO')
pasta_treino = diretorio_base / 'ARQUIVOS TREINO'
pasta_teste = diretorio_base / 'ARQUIVOS TESTE'

def criar_pastas():
    for pasta in [pasta_treino, pasta_teste]:
        if pasta.exists():
            shutil.rmtree(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
    print("✅ Passo 1: Pastas TREINO e TESTE criadas.")

# --- 2-ANALISAR A PASTA Arquivos Ajustados ---
pasta_origem = diretorio_base / 'Arquivos Ajustados'
csv_referencia = pasta_origem / 'referencia_dataset.csv'

# Variáveis para controle dos dados selecionados
dados_treino = []
dados_teste = []

# --- 3-FUNÇÃO QUE ALEATORIAMENTE ESCOLHA E INSERE NA PASTA ---
def escolher_e_inserir(df_fonte, qtd_benign, qtd_malign, pasta_destino):
    lista_selecionados = []
    
    # Seleção aleatória de Benignos
    df_b = df_fonte[df_fonte['Diagnostico'] == 'benign'].sample(n=qtd_benign)
    # Seleção aleatória de Malignos
    df_m = df_fonte[df_fonte['Diagnostico'] == 'malignant'].sample(n=qtd_malign)
    
    df_total = pd.concat([df_b, df_m])
    
    print(f"📂 Copiando {len(df_total)} arquivos para {pasta_destino.name}...")
    
    for _, linha in df_total.iterrows():
        nome_arq = linha['Arquivo']
        status_arq = linha['Diagnostico']
        origem = pasta_origem / nome_arq
        destino = pasta_destino / nome_arq
        
        if origem.exists():
            shutil.copy2(origem, destino)
            lista_selecionados.append({'Arquivo': nome_arq, 'Status': status_arq})
            
    return lista_selecionados, df_total

# --- 4-FUNÇÃO QUE CRIE O ARQUIVO CSV DE REGISTRO NA PASTA ---
def criar_csv_registro(lista_dados, pasta_destino, nome_arquivo_csv):
    df_registro = pd.DataFrame(lista_dados)
    caminho_csv = pasta_destino / nome_arquivo_csv
    
    # Salva com as 2 colunas: Arquivo e Status
    df_registro.to_csv(caminho_csv, index=False, sep=';', encoding='latin1')
    print(f"📄 Passo 4: CSV '{nome_arquivo_csv}' criado em {pasta_destino.name}.")

# --- EXECUÇÃO DO PROCESSO ---

# Passo 1
criar_pastas()

# Passo 2
if not csv_referencia.exists():
    print(f"❌ Erro: CSV não encontrado em {csv_referencia}")
else:
    df_original = pd.read_csv(csv_referencia, sep=';', encoding='latin1')

    # Passo 3: Escolher 1814 Benignos + 1814 Malignos para TREINO
    registros_treino, df_usados_treino = escolher_e_inserir(df_original, 1814, 1814, pasta_treino)
    
    # Passo 4: Criar CSV na pasta TREINO
    criar_csv_registro(registros_treino, pasta_treino, 'lista_treino.csv')

    # Passo 5: Mesmo procedimento para TESTE (454 benignos + 454 malignos)
    # Importante: Selecionamos do que sobrou (df_original menos o que foi para treino)
    df_restante = df_original[~df_original['Arquivo'].isin(df_usados_treino['Arquivo'])]
    
    registros_teste, _ = escolher_e_inserir(df_restante, 454, 454, pasta_teste)
    criar_csv_registro(registros_teste, pasta_teste, 'lista_teste.csv')

    print("\n--- PROCESSO FINALIZADO COM SUCESSO ---")
    print(f"Total Treino: {len(registros_treino)} arquivos")
    print(f"Total Teste: {len(registros_teste)} arquivos")