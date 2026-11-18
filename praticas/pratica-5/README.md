# Prática 5: Rede neural do zero com NumPy

**Notebook:** `Prática_RedeNeural_Pedro_Vieira.ipynb`

## Objetivo
Implementar uma MLP (forward propagation, backpropagation e atualização de pesos) usando apenas NumPy.

## Dados
Iris, carregada por `sklearn.datasets.load_iris` (sem arquivo externo). Usa apenas as classes 0 e 1 e as 2 primeiras features.

## Método
- Rede 2 → 5 (ReLU) → 1 (sigmoide), pesos aleatórios (`seed=0`).
- Perda MSE, 10.000 épocas, taxa de aprendizado 0.1.
- Classe definida com limiar 0.5. Divisão treino/teste 80/20.

## Resultados (registrados no notebook)
- MSE de treino: 0.2534 (época 0) → 0.0986 (época 9000)
- Acurácia no teste: **0.95**
- Matriz de confusão: `[[11, 1], [0, 8]]`

## Como reproduzir
Execute o notebook; não precisa de dados externos.

## Dependências
numpy, scikit-learn, matplotlib
