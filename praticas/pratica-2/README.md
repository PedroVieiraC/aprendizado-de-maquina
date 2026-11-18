# Prática 2: Classificadores nos dados do Kaggle (Santander)

**Notebook:** `pratica2_Pedro_Vieira.ipynb`

## Objetivo
Aplicar Árvore de Decisão, kNN e Naive Bayes à competição *Santander Customer Transaction Prediction*, comparando a avaliação local (ROC AUC) com o score do Kaggle.

## Dados (não incluídos)
`train.csv` e `test.csv` da competição: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data (200 mil linhas × 200 variáveis anônimas, cerca de 300 MB cada). As regras da competição não permitem redistribuir os dados.

## Método
- Amostragens de 5%, 20% e 25% do treino; divisão 70/30.
- Árvore de Decisão: busca de `max_depth` de 1 a 14.
- kNN com `StandardScaler`, k ∈ {5, 15, 31} (5%) e k = 21 (20%).
- Naive Bayes Gaussiano com ajuste de `var_smoothing`.
- Geração dos arquivos `submission_*.csv` para o Kaggle.

## Resultados (ROC AUC local, registrados no notebook)
| Modelo | ROC AUC local |
|---|---|
| Árvore de Decisão (padrão) | 0.5544 |
| Árvore de Decisão (`max_depth=8`) | 0.6218 |
| kNN, 5% dos dados (melhor k=31) | 0.5715 |
| kNN, 20% dos dados (k=21) | 0.6076 |
| Naive Bayes, 5% | 0.8087 |
| Naive Bayes, 20% | 0.8718 |

Os scores do Kaggle aparecem como capturas de tela no notebook. Pela conclusão do notebook, o Naive Bayes foi o melhor modelo (ROC AUC próximo de 0.88), coerente com variáveis independentes e anônimas.

## Como reproduzir
1. Baixe `train.csv` e `test.csv` da competição para esta pasta (ou, no Colab, monte o Drive e use `%cd` na pasta dos dados; as linhas estão comentadas na 1ª célula).
2. Execute em ordem. As chamadas `files.download` só funcionam no Colab.

## Dependências
pandas, numpy, scikit-learn, matplotlib
