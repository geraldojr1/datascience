import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

def run_analysis():
    """
    Executa uma análise completa, considerando RF como reprovação e gerando um
    gráfico com o ranking de disciplinas por taxa de reprovação.
    """
    print("Iniciando análise final, incluindo RF como reprovação e o novo ranking...")

    # --- Etapa 1: Carregamento e Preparação dos Dados ---
    try:
        df = pd.read_csv('notas.csv', encoding='utf-8-sig')
        print("Arquivo 'notas.csv' carregado com sucesso.")
    except FileNotFoundError:
        print("ERRO: O arquivo 'notas.csv' não foi encontrado.")
        return

    # Limpeza e preparação dos dados
    df.columns = df.columns.str.strip()
    colunas_unidades = [f'Unidade {i}' for i in range(1, 6)]
    col_prova_final = 'Prova Final'
    col_faltas = 'Faltas'
    colunas_para_converter = colunas_unidades + [col_prova_final, col_faltas]
    for col in colunas_para_converter:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    # !!!!! MUDANÇA CRÍTICA AQUI !!!!!
    # Mapeia AM, RN e RF para os valores padrão
    mapeamento_situacao = {'AM': 'Aprovado', 'RN': 'Reprovado', 'RF': 'Reprovado'}
    df['Situação'] = df['Situação'].str.strip().replace(mapeamento_situacao)
    
    # Engenharia de Variáveis
    unidades_existentes = [col for col in colunas_unidades if col in df.columns]
    df['Media_Unidades'] = df[unidades_existentes].mean(axis=1)
    df['Nota_Final_Calculada'] = (df['Media_Unidades'] + df[col_prova_final]) / 2
    df['Nota_Final_Calculada'].fillna(df['Media_Unidades'], inplace=True)
    
    print("Mapeamento da coluna 'Situação' (incluindo RF) concluído.")
    
    # --- Redirecionamento da Saída para Arquivo TXT ---
    original_stdout = sys.stdout
    with open('resultados_analise_final.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        print("="*50 + "\n--- RESULTADOS DA ANÁLISE FINAL ---\n" + "="*50 + "\n")

        # --- Análises 1 a 4 ---
        # (O código das análises 1 a 4 permanece o mesmo e será impresso aqui)
        # ...

        # --- Análise 5 ---
        print("Análise 5: Análise de Aprovação e Reprovação\n" + "="*45)
        # 5.1 e 5.2
        crosstab_situacao = pd.crosstab(df['Disciplina'], df['Situação'], normalize='index') * 100
        media_notas_situacao = df.groupby('Situação')['Nota_Final_Calculada'].describe()
        print("\n5.1: Percentual de Sucesso por Disciplina\n" + "-"*35)
        print(crosstab_situacao.round(2))
        print("\n5.2: Média de Notas: Aprovados vs. Reprovados\n" + "-"*35)
        print(media_notas_situacao.round(2))
        
        # 5.3
        print("\n5.3: Resumo por Disciplina\n" + "-"*35)
        resumo_disciplinas = df.groupby('Disciplina').agg(
            Media_Geral_Turma=('Nota_Final_Calculada', 'mean'),
            Total_Alunos=('AlunoID', 'count')
        )
        reprovados_por_disciplina = df[df['Situação'] == 'Reprovado'].groupby('Disciplina').agg(Total_Reprovados=('AlunoID', 'count'))
        resumo_disciplinas = resumo_disciplinas.join(reprovados_por_disciplina).fillna(0)
        resumo_disciplinas['Taxa_Reprovacao_Perc'] = (resumo_disciplinas['Total_Reprovados'] / resumo_disciplinas['Total_Alunos']) * 100
        resumo_disciplinas.round(2).to_csv('resumo_disciplinas.csv', encoding='utf-8-sig')
        print(" -> O arquivo 'resumo_disciplinas.csv' foi criado com a taxa de reprovação.\n")

        # --- NOVA ANÁLISE 6: Ranking de Disciplinas com Maior Reprovação ---
        print("Análise 6: Ranking de Disciplinas por Taxa de Reprovação\n" + "="*45)
        # Ordena o dataframe pela taxa de reprovação
        ranking_reprovacao = resumo_disciplinas.sort_values(by='Taxa_Reprovacao_Perc', ascending=False)
        print("Disciplinas com as maiores taxas de reprovação (%):\n" + "-"*35)
        print(ranking_reprovacao['Taxa_Reprovacao_Perc'].to_string(float_format="%.2f%%"))
        print("\n -> Insight: Este ranking ajuda a priorizar ações pedagógicas nas disciplinas mais críticas.\n")

    # Restaura a saída padrão
    sys.stdout = original_stdout

    # --- Geração dos Gráficos ---
    print("Gerando gráficos...")
    # (Gráficos das análises anteriores)
    # ...

    # Gráfico da Análise 6 (NOVO)
    if not ranking_reprovacao.empty:
        plt.figure(figsize=(15, 10))
        sns.barplot(x=ranking_reprovacao['Taxa_Reprovacao_Perc'], y=ranking_reprovacao.index, palette='Reds_r')
        plt.title('Análise 6: Ranking de Disciplinas por Taxa de Reprovação', fontsize=16)
        plt.xlabel('Taxa de Reprovação (%)', fontsize=12)
        plt.ylabel('Disciplina', fontsize=12)
        # Adiciona o valor nas barras
        for index, value in enumerate(ranking_reprovacao['Taxa_Reprovacao_Perc']):
            plt.text(value, index, f' {value:.2f}%', va='center')
        plt.tight_layout()
        plt.savefig('analise_6_ranking_reprovacao.png')
        plt.close()

    print("\nAnálise concluída.")
    print(" -> Os resultados textuais foram salvos em 'resultados_analise_final.txt'")
    print(" -> O arquivo de resumo 'resumo_disciplinas.csv' foi atualizado.")
    print(" -> O novo gráfico com o ranking de reprovação foi salvo.")

if __name__ == '__main__':
    run_analysis()