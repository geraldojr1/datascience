import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

def run_analysis():
    """
    Executa uma análise bivariada completa, ajustada para interpretar
    'AM' como Aprovado e 'RN' como Reprovado na coluna 'Situação'.
    """
    print("Iniciando a análise bivariada com o mapeamento de AM/RN...")

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
    # Mapeia os códigos AM/RN para os valores padrão "Aprovado" e "Reprovado"
    mapeamento_situacao = {'AM': 'Aprovado', 'RN': 'Reprovado'}
    df['Situação'] = df['Situação'].str.strip().replace(mapeamento_situacao)
    
    # Engenharia de Variáveis
    unidades_existentes = [col for col in colunas_unidades if col in df.columns]
    df['Media_Unidades'] = df[unidades_existentes].mean(axis=1)
    df['Nota_Final_Calculada'] = (df['Media_Unidades'] + df[col_prova_final]) / 2
    df['Nota_Final_Calculada'].fillna(df['Media_Unidades'], inplace=True)
    
    print("Mapeamento da coluna 'Situação' e preparação dos dados concluídos.")
    
    # --- Redirecionamento da Saída para Arquivo TXT ---
    original_stdout = sys.stdout
    with open('resultados_analise_bivariada.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        print("="*50 + "\n--- RESULTADOS DA ANÁLISE BIVARIADA (MAPEAMENTO AM/RN) ---\n" + "="*50 + "\n")

        # --- Etapa 2: Análises ---

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
        print(" -> Coeficiente de Correlação (Pearson):")
        correlacoes_unidades = df.groupby('Disciplina')[['Media_Unidades', col_prova_final]].corr(numeric_only=True).unstack().iloc[:, 1]
        print(correlacoes_unidades.to_string())
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
            print(" -> Análise não executada: não foram encontrados dados suficientes para os grupos 'Aprovado' e 'Reprovado'.")
            print(f" -> Alunos no grupo Aprovado: {len(faltas_aprovados)}")
            print(f" -> Alunos no grupo Reprovado: {len(faltas_reprovados)}\n")
        
        # Análise 4: Faltas vs. Notas
        print("Análise 4: Relação entre Faltas e Notas Finais\n" + "-"*35)
        print(" -> Coeficiente de Correlação (Pearson):")
        correlacoes_faltas = df.groupby('Disciplina')[[col_faltas, 'Nota_Final_Calculada']].corr(numeric_only=True).unstack().iloc[:, 1]
        print(correlacoes_faltas.to_string())
        print("\n -> Insight: Valores negativos indicam que um maior número de faltas está associado a notas finais menores.\n")

    # Restaura a saída padrão
    sys.stdout = original_stdout

    # --- Geração dos Gráficos ---
    print("Gerando gráficos...")
    plt.figure(figsize=(14, 7)); sns.boxplot(data=df, x='Disciplina', y='Nota_Final_Calculada', palette='viridis'); plt.title('Análise 1: Distribuição da Nota Final por Disciplina', fontsize=16); plt.tight_layout(); plt.savefig('analise_1_boxplot_notas_por_disciplina.png'); plt.close()
    lm_unidades = sns.lmplot(data=df, x='Media_Unidades', y=col_prova_final, col='Disciplina', col_wrap=3, palette='magma', height=4, line_kws={'color': 'green'}); lm_unidades.fig.suptitle('Análise 2: Média das Unidades vs. Prova Final', y=1.02); plt.savefig('analise_2_scatterplot_unidades_vs_provafinal.png'); plt.close()
    plt.figure(figsize=(10, 6)); sns.boxplot(data=df, x='Situação', y=col_faltas, palette='coolwarm'); plt.title('Análise 3: Distribuição de Faltas por Situação Final', fontsize=16); plt.xlabel('Situação Final'); plt.ylabel('Número de Faltas'); plt.tight_layout(); plt.savefig('analise_3_boxplot_faltas_por_situacao.png'); plt.close()
    lm = sns.lmplot(data=df, x=col_faltas, y='Nota_Final_Calculada', col='Disciplina', col_wrap=3, palette='plasma', height=4, line_kws={'color': 'red'}); lm.fig.suptitle('Análise 4: Relação entre Faltas e Nota Final por Disciplina', y=1.02); plt.savefig('analise_4_scatterplot_faltas_vs_nota.png'); plt.close()
    
    print("\nAnálise concluída.")
    print(" -> Os resultados textuais foram salvos em 'resultados_analise_bivariada.txt'")
    print(" -> Os gráficos foram salvos como arquivos .png.")

if __name__ == '__main__':
    run_analysis()