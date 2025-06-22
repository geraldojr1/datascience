import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, mode

# Leitura do CSV
try:
    df_notas = pd.read_csv('notas.csv', encoding='utf-8-sig')
except FileNotFoundError:
    print("Erro: O arquivo 'notas.csv' não foi encontrado.")
    print("Por favor, verifique se o nome do arquivo está correto e se ele está na mesma pasta do seu script.")
    exit()

df_notas.columns = df_notas.columns.str.strip()

# Colunas de unidades e prova final
colunas_unidades = ['Unidade 1', 'Unidade 2', 'Unidade 3', 'Unidade 4', 'Unidade 5']
col_prova_final = 'Prova Final'

# Converter notas para numérico
for col in colunas_unidades + [col_prova_final]:
    if col in df_notas.columns:
        df_notas[col] = pd.to_numeric(df_notas[col].astype(str).str.replace(',', '.'), errors='coerce')

# Analisar por disciplina
disciplinas = df_notas['Disciplina'].unique()

# Abre o arquivo TXT para escrita (o 'with' garante que será fechado automaticamente)
with open('analise_estatistica.txt', 'w', encoding='utf-8') as f:
    for disc in disciplinas:
        # Direciona a saída para o arquivo usando o argumento 'file=f'
        f.write(f"\n\n=== Análise Estatística - Disciplina: {disc} ===\n")

        dados_disc = df_notas[df_notas['Disciplina'] == disc].copy()

        unidades_existentes = [col for col in colunas_unidades if col in dados_disc.columns and dados_disc[col].notna().any()]

        dados_disc.loc[:, 'Media_Unidades'] = dados_disc[unidades_existentes].mean(axis=1)

        if col_prova_final in dados_disc.columns and dados_disc[col_prova_final].notna().any():
            dados_disc.loc[:, 'Nota_Final_Calculada'] = (dados_disc['Media_Unidades'] + dados_disc[col_prova_final]) / 2
        else:
            dados_disc.loc[:, 'Nota_Final_Calculada'] = dados_disc['Media_Unidades']

        notas_finais = dados_disc['Nota_Final_Calculada'].dropna()

        if notas_finais.empty:
            f.write("Nenhuma nota final disponível para análise nesta disciplina.\n")
            continue

        # Estatísticas
        media = notas_finais.mean()
        mediana = notas_finais.median()
        try:
            moda_val = mode(notas_finais, keepdims=False).mode
            moda = f"{moda_val}"
        except Exception as e:
            moda = "Não definida"
        desvio = notas_finais.std()
        cv = (desvio / media) * 100 if media != 0 else np.nan
        assimetria = skew(notas_finais)
        curtose_valor = kurtosis(notas_finais)
        minimo = notas_finais.min()
        maximo = notas_finais.max()
        quartis = notas_finais.quantile([0.25, 0.5, 0.75])

        # Escreve as estatísticas no arquivo
        f.write(f"Média: {media:.2f}\n")
        f.write(f"Mediana: {mediana:.2f}\n")
        f.write(f"Moda: {moda}\n")
        f.write(f"Desvio Padrão: {desvio:.2f}\n")
        f.write(f"Coeficiente de Variação (CV): {cv:.2f}%\n")
        f.write(f"Assimetria (Skewness): {assimetria:.2f}\n")
        f.write(f"Curtose (Kurtosis): {curtose_valor:.2f}\n")
        f.write(f"Nota Mínima: {minimo:.2f}\n")
        f.write(f"Nota Máxima: {maximo:.2f}\n")
        f.write("Quartis:\n")
        f.write(f"{quartis.round(2).to_string()}\n")


        # --- Geração do Gráfico (sem alterações) ---
        plt.figure(figsize=(12, 6))
        sns.histplot(
            notas_finais,
            bins=np.arange(0, 10.5, 0.5),
            kde=True,
            color='cornflowerblue',
            edgecolor='black'
        )
        plt.title(f'Histograma de Frequência - {disc}', fontsize=16)
        plt.xlabel('Nota Final (0 a 10)', fontsize=12)
        plt.ylabel('Número de Alunos', fontsize=12)
        plt.xlim(0, 10)
        plt.xticks(np.arange(0, 11, 1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Salva o gráfico em um arquivo de imagem
        safe_disc_name = "".join([c if c.isalnum() else "_" for c in disc])
        plt.savefig(f'histograma_{safe_disc_name}.png')
        plt.close() # Fecha a figura para não consumir memória

        # Este print aparecerá no terminal para indicar o progresso
        print(f"Análise da disciplina '{disc}' salva no TXT e histograma gerado.")

print("\nAnálise concluída! Os resultados foram salvos no arquivo 'analise_estatistica.txt'.")