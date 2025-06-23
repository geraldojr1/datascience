import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

def run_full_analysis():
    """
    Executa uma análise acadêmica completa e unificada, desde a preparação
    dos dados até a geração de relatórios e gráficos detalhados.
    """
    

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

    # Mapeia AM, RN e RF para os valores padrão "Aprovado" e "Reprovado"
    mapeamento_situacao = {'AM': 'Aprovado', 'RN': 'Reprovado', 'RF': 'Reprovado'}
    df['Situação'] = df['Situação'].str.strip().replace(mapeamento_situacao)
    
    # Engenharia de Variáveis
    unidades_existentes = [col for col in colunas_unidades if col in df.columns]
    df['Media_Unidades'] = df[unidades_existentes].mean(axis=1)
    df['Nota_Final_Calculada'] = (df['Media_Unidades'] + df[col_prova_final]) / 2
    df['Nota_Final_Calculada'].fillna(df['Media_Unidades'], inplace=True)
    
    print("Mapeamento da coluna 'Situação' e preparação dos dados concluídos.")
    
    # --- Redirecionamento da Saída para Arquivo TXT ---
    original_stdout = sys.stdout
    with open('resultados_analise_completa.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f # Redireciona a saída para o arquivo
        
        print("="*50 + "\n--- RESULTADOS DA ANÁLISE ACADÊMICA COMPLETA ---\n" + "="*50 + "\n")

        # --- Análises Estatísticas e Bivariadas ---

        # Análise 1: Nota Final vs. Disciplina
        print("Análise 1: Nota Final por Disciplina\n" + "-"*35)
        disciplinas = df['Disciplina'].unique()
        if len(disciplinas) > 1:
            samples = [df['Nota_Final_Calculada'][df['Disciplina'] == d].dropna() for d in disciplinas if not df['Nota_Final_Calculada'][df['Disciplina'] == d].dropna().empty]
            if len(samples) > 1:
                f_val, p_val = stats.f_oneway(*samples)
                print(f" -> Teste ANOVA: F-value = {f_val:.2f}, p-value = {p_val:.4f}")
                if p_val < 0.05:
                    print(" -> Insight: Há uma diferença estatisticamente significativa nas médias de notas entre as disciplinas.\n")
                else:
                    print(" -> Insight: Não há uma diferença estatisticamente significativa nas médias de notas entre as disciplinas.\n")

        # Análise 2: Média das Unidades vs. Prova Final
        print("Análise 2: Relação entre Média das Unidades e Prova Final\n" + "-"*35)
        correlacoes_unidades = df.groupby('Disciplina')[['Media_Unidades', col_prova_final]].corr(numeric_only=True).unstack().iloc[:, 1]
        print(" -> Coeficiente de Correlação (Pearson):\n" + correlacoes_unidades.to_string())
        print("\n -> Insight: Valores positivos altos indicam que o desempenho ao longo do semestre é um bom preditor da nota na prova final.\n")

        # Análise 3: Faltas vs. Situação (Aprovação/Reprovação)
        print("Análise 3: Relação entre Faltas e a Situação Final\n" + "-"*35)
        faltas_aprovados = df[df['Situação'] == 'Aprovado'][col_faltas].dropna()
        faltas_reprovados = df[df['Situação'] == 'Reprovado'][col_faltas].dropna()
        if len(faltas_aprovados) > 1 and len(faltas_reprovados) > 1:
            print(f" -> Média de Faltas (Grupo Aprovado): {faltas_aprovados.mean():.2f}")
            print(f" -> Média de Faltas (Grupo Reprovado): {faltas_reprovados.mean():.2f}")
            t_stat, p_val_ttest = stats.ttest_ind(faltas_aprovados, faltas_reprovados, equal_var=False)
            print(f" -> Teste T: t-statistic = {t_stat:.2f}, p-value = {p_val_ttest:.4f}")
            if p_val_ttest < 0.05:
                print(" -> Insight: A diferença na média de faltas entre os grupos Aprovado e Reprovado é estatisticamente significativa.\n")
            else:
                print(" -> Insight: Não há uma diferença estatisticamente significativa na média de faltas entre os grupos.\n")
        else:
            print(" -> Análise não executada: dados insuficientes para os grupos 'Aprovado' e 'Reprovado'.\n")
        
        # Análise 4: Faltas vs. Notas
        print("Análise 4: Relação entre Faltas e Notas Finais\n" + "-"*35)
        correlacoes_faltas = df.groupby('Disciplina')[[col_faltas, 'Nota_Final_Calculada']].corr(numeric_only=True).unstack().iloc[:, 1]
        print(" -> Coeficiente de Correlação (Pearson):\n" + correlacoes_faltas.to_string())
        print("\n -> Insight: Valores negativos indicam que um maior número de faltas está associado a notas finais menores.\n")

        # Análise 5: Análise Detalhada de Aprovação e Reprovação
        print("Análise 5: Análise de Aprovação e Reprovação\n" + "="*45)
        crosstab_situacao = pd.crosstab(df['Disciplina'], df['Situação'], normalize='index') * 100
        media_notas_situacao = df.groupby('Situação')['Nota_Final_Calculada'].describe()
        print("\n5.1: Percentual de Sucesso por Disciplina (%)\n" + "-"*35)
        print(crosstab_situacao.round(2))
        print("\n5.2: Comparativo de Notas: Aprovados vs. Reprovados\n" + "-"*35)
        print(media_notas_situacao.round(2))
        
        print("\n5.3: Criação do Resumo por Disciplina\n" + "-"*35)
        resumo_disciplinas = df.groupby('Disciplina').agg(
            Media_Geral_Turma=('Nota_Final_Calculada', 'mean'),
            Total_Alunos=('AlunoID', 'count')
        )
        reprovados_por_disciplina = df[df['Situação'] == 'Reprovado'].groupby('Disciplina').agg(Total_Reprovados=('AlunoID', 'count'))
        resumo_disciplinas = resumo_disciplinas.join(reprovados_por_disciplina).fillna(0)
        resumo_disciplinas['Taxa_Reprovacao_Perc'] = (resumo_disciplinas['Total_Reprovados'] / resumo_disciplinas['Total_Alunos']) * 100
        resumo_disciplinas.round(2).to_csv('resumo_disciplinas.csv', encoding='utf-8-sig')
        print(" -> O arquivo 'resumo_disciplinas.csv' foi criado com a taxa de reprovação de cada disciplina.\n")

        # Análise 6: Ranking de Disciplinas com Maior Reprovação
        print("Análise 6: Ranking de Disciplinas por Taxa de Reprovação\n" + "="*45)
        ranking_reprovacao = resumo_disciplinas.sort_values(by='Taxa_Reprovacao_Perc', ascending=False)
        print("Disciplinas com as maiores taxas de reprovação (%):\n" + "-"*35)
        print(ranking_reprovacao['Taxa_Reprovacao_Perc'].to_string(float_format="%.2f%%"))
        print("\n -> Insight: Este ranking ajuda a priorizar ações pedagógicas nas disciplinas mais críticas.\n")

    # Restaura a saída padrão para o terminal
    sys.stdout = original_stdout

    # --- Geração de todos os Gráficos ---
    print("Gerando gráficos...")
    # Gráfico 5.1
    if 'crosstab_situacao' in locals() and not crosstab_situacao.empty:
        crosstab_situacao.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='viridis'); plt.title('Análise 5.1: Percentual de Aprovação e Reprovação por Disciplina', fontsize=16); plt.ylabel('Percentual de Alunos (%)'); plt.xlabel('Disciplina'); plt.xticks(rotation=45, ha='right'); plt.legend(title='Situação'); plt.tight_layout(); plt.savefig('analise_5_perc_aprovacao.png'); plt.close()
    # Gráfico 6
    if 'ranking_reprovacao' in locals() and not ranking_reprovacao.empty:
        plt.figure(figsize=(15, 10)); sns.barplot(x=ranking_reprovacao['Taxa_Reprovacao_Perc'], y=ranking_reprovacao.index, palette='Reds_r'); plt.title('Análise 6: Ranking de Disciplinas por Taxa de Reprovação', fontsize=16); plt.xlabel('Taxa de Reprovação (%)', fontsize=12); plt.ylabel('Disciplina', fontsize=12)
        for index, value in enumerate(ranking_reprovacao['Taxa_Reprovacao_Perc']): plt.text(value, index, f' {value:.2f}%', va='center')
        plt.tight_layout(); plt.savefig('analise_6_ranking_reprovacao.png'); plt.close()

    print("\nAnálise completa finalizada.")
    print(" -> Os resultados textuais foram salvos em 'resultados_analise_completa.txt'")
    print(" -> O arquivo de resumo 'resumo_disciplinas.csv' foi criado/atualizado.")
    print(" -> Todos os gráficos foram salvos como arquivos .png.")

if __name__ == '__main__':
    run_full_analysis()