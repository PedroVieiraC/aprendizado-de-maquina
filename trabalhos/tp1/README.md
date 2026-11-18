# TP1: Store Sales – Time Series Forecasting

**Notebook:** `TP1_dupla.ipynb`, feito em dupla por um colega e Pedro Vieira.

## Objetivo
Prever as vendas por família de produtos nas lojas Favorita (Equador), na competição do Kaggle *Store Sales – Time Series Forecasting*.

## Dados (não incluídos)
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data (`train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`).
O notebook lê `train_consolidated.csv` (cerca de 226 MB), um join de todas as tabelas feito pela dupla **fora deste notebook**, além do `test.csv` original.

## Método
- Ordenação temporal, tipos categóricos, limpeza de `oil_price` e `transactions`.
- Recorte para entrega parcial: datas a partir de 2016-01-01 e as 10 primeiras lojas (21.153 linhas).
- Features: ano, mês, dia, dia da semana, semana do ano, dia de pagamento (15 e 30), pós-terremoto (≥ 2016-04-16), lags de 1, 7 e 14 dias.
- Pipeline com imputação, `StandardScaler` e `OneHotEncoder`. Alvo em `log1p`.
- Avaliação com `TimeSeriesSplit(5)` usando RMSLE e MAE.
- Modelos: Dummy (média), Árvore de Decisão (`max_depth=10`), kNN (k=5) e Stacking (árvore + kNN → Ridge).

## Resultados (registrados no notebook)
| Modelo | RMSLE | MAE |
|---|---|---|
| Baseline (média) | 2.5583 | 594.97 |
| Árvore de Decisão | 0.6540 | 238.00 |
| kNN | 0.9736 | 234.94 |
| **Stacking** | **0.6029** | **210.44** |

O score do Kaggle aparece como captura de tela no notebook.

## Estado
Entrega parcial com amostra reduzida. Metade das células de código não tem saída registrada.

## Como reproduzir
1. Baixe os dados da competição e gere `train_consolidated.csv` (join de train + stores + oil + transactions + holidays_events). O script do join não está no material.
2. Coloque `train_consolidated.csv` e `test.csv` nesta pasta e execute em ordem.

## Dependências
pandas, numpy, scikit-learn (≥ 1.4, por causa do `root_mean_squared_log_error`), matplotlib
