# Prática 1: k-Nearest Neighbors (kNN)

**Notebook:** `Untitled0.ipynb`

## Objetivo
Avaliar o kNN em quatro conjuntos de dados, analisando a influência do número de vizinhos (k), da métrica de distância (euclidiana × manhattan), da ponderação dos vizinhos (uniform × distance) e da normalização.

## Dados (não incluídos no repositório)
| Arquivo | Fonte |
|---|---|
| `iris.csv` | Iris, UCI: https://archive.ics.uci.edu/dataset/53/iris |
| `WineQT.csv` | Wine Quality, Kaggle: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset (derivado de https://archive.ics.uci.edu/dataset/186/wine+quality) |
| `santander_train.csv` | `train.csv` da competição Kaggle *Santander Customer Transaction Prediction*: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data |
| `evasao.csv` | Base de evasão de alunos fornecida na disciplina. **Não é publicada**: contém atributos pessoais de estudantes (sexo, etnia, idade, cidade e escola de origem). |

## Método
- `StandardScaler` (e uma variante sem normalização para a Iris), divisão treino/teste 70/30 (`random_state=42`).
- Curva de acurácia para k = 1..30; comparação de distâncias e de pesos com k fixo.
- Santander: amostragem de 10% (20.000 linhas) para viabilizar a execução.

## Resultados (registrados no notebook)
| Dataset | k | Euclidiana | Manhattan | uniform | distance |
|---|---|---|---|---|---|
| Iris (normalizada e não normalizada) | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| WineQT | 17 | 0.6356 | 0.6181 | 0.6356 | 0.6822 |
| evasao | 9 | 0.8537 | 0.8488 | 0.8537 | 0.8488 |
| Santander (amostra 10%) | 7 | 0.8995 | 0.8995 | 0.8985 | 0.8985 |

Conclusões do notebook: no Santander a acurácia fica estável em torno de 0.89–0.90 para todo k; na evasão estabiliza a partir de k≈8–9 (≈85%); no WineQT o desempenho é baixo (≈60–65%).

## Como reproduzir
1. Baixe os arquivos acima para esta pasta (renomeie o `train.csv` do Santander para `santander_train.csv`).
2. Abra o notebook (Jupyter ou Colab) e execute as células em ordem. As seções da evasão exigem o arquivo da disciplina.

## Dependências
pandas, scikit-learn, matplotlib
