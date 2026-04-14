📊 Predição de Paradas Produtivas com Random Forest

Este projeto tem como objetivo prever a ocorrência de paradas produtivas em um ambiente industrial utilizando técnicas de Machine Learning, com foco em manutenção preditiva.

O modelo foi desenvolvido como parte de um Trabalho de Conclusão de Curso (TCC) em Data Science, aplicando boas práticas de engenharia de dados, balanceamento de classes e avaliação robusta de desempenho.

🚀 Tecnologias Utilizadas
- Python 3.x
- Pandas / NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib


📂 Estrutura do Projeto

📁 projeto
│-- codigo_random_forest_tcc.py

│-- Levantamento Panstone 2024-2025-TCC.xlsx


│-- 📁 saida_modelo/

│   ├── metricas_modelo.csv

│   ├── matriz_confusao.png

│   ├── curva_roc.png

│   ├── importancia_variaveis.csv

│   └── importancia_variaveis.png



⚙️ Funcionamento do Pipeline

O modelo segue um pipeline completo de Machine Learning:

1. 📥 Leitura dos dados
Importação da base Excel
Criação da variável alvo (Parada_bin)

2. 🛠 Engenharia de atributos
Extração de:
Mês, dia e dia da semana
Hora e minuto de início da falha
Remoção de variáveis irrelevantes ou com risco de data leakage

3. 🔄 Pré-processamento
Dados numéricos:
Imputação pela mediana
Dados categóricos:
Imputação por moda
One-Hot Encoding

4. ⚖️ Balanceamento de classes
Aplicação do SMOTE para tratar desbalanceamento

5. 🤖 Modelo
Algoritmo: Random Forest
Parâmetros:
n_estimators = 400
class_weight = balanced
min_samples_leaf = 2

6. 📊 Avaliação

Divisão:

80% treino / 20% teste

Métricas utilizadas:

- Acurácia
- Precisão
- Recall
- F1-score
- AUC-ROC

Validação adicional:

Cross-validation estratificada (k=3)

📈 Resultados Gerados

O modelo gera automaticamente:

📊 Métricas
metricas_modelo.csv

📉 Gráficos
Matriz de confusão

Curva ROC
Importância das variáveis

🔍 Importância das Variáveis
Ranking das variáveis mais relevantes para previsão


🎯 Objetivo do Projeto

Demonstrar como técnicas de Machine Learning podem ser aplicadas para:

- Antecipar falhas operacionais
- Apoiar decisões de manutenção
- Reduzir paradas não planejadas
- Aumentar a eficiência produtiva

👨‍💻 Autor

Bruno Maia
Engenheiro de Manutenção | Data Science


📄 Licença

Este projeto é de uso acadêmico e educacional.
