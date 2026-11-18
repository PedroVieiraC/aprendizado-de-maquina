# Prática 3: Validação cruzada e métricas de avaliação

**Notebook:** `Prática 3 Pedro Vieira.ipynb`

## Objetivo
Comparar a avaliação *holdout* com a validação cruzada (10-fold) usando ROC AUC, com os modelos Naive Bayes, Random Forest e XGBoost, no dataset Santander do Kaggle.

## Dados (não incluídos)
`train.csv` e `test.csv` de https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data. Foi usada uma amostra de 20% do treino.

## Método
- Holdout 70/30 estratificado e `cross_validate` com `cv=10` e `scoring='roc_auc'`.
- Random Forest (100 árvores, `max_depth=5`) e XGBoost (100 árvores, `max_depth=3`, `lr=0.1`).
- XGBoost "otimizado": 300 árvores, `max_depth=5`, `lr=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.
- Submissões ao Kaggle.

## Resultados (registrados no notebook)
| Modelo | AUC Holdout | AUC CV (10-fold) |
|---|---|---|
| Naive Bayes | 0.8882 | 0.8858 |
| Random Forest | 0.7831 | 0.7773 |
| XGBoost | 0.8318 | 0.8294 |
| XGBoost otimizado | 0.8623 | 0.8627 |

Conclusão do notebook: o Naive Bayes superou o XGBoost também no Kaggle. A amostra reduzida limitou o XGBoost, que continuou inferior mesmo após o ajuste. A validação cruzada dá uma estimativa mais estável que o holdout. Os scores do Kaggle aparecem como capturas de tela.

## Como reproduzir
Baixe os dados da competição para esta pasta e execute as células em ordem (`files.download` só funciona no Colab).

## Dependências
pandas, numpy, scikit-learn, xgboost, matplotlib
