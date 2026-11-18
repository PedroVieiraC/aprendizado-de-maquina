# Aprendizado de Máquina (Inteligência Computacional)

Práticas e trabalhos desenvolvidos na disciplina de Inteligência Computacional (aprendizado de máquina). Cada pasta tem seu próprio README com objetivo, dados, método, resultados e como reproduzir.

Os datasets **não** estão no repositório (tamanho, regras das competições Kaggle ou dados pessoais); cada README indica onde obtê-los.

## Práticas

| Pasta | Tema | Ferramentas |
|---|---|---|
| [praticas/pratica-1](praticas/pratica-1) | kNN: efeito de k, distância, peso e normalização (Iris, Wine, evasão, Santander) | pandas, scikit-learn |
| [praticas/pratica-2](praticas/pratica-2) | Classificadores (Árvore, kNN, Naive Bayes) no Kaggle Santander | pandas, scikit-learn |
| [praticas/pratica-3](praticas/pratica-3) | Validação cruzada e métricas (NB, Random Forest, XGBoost) | scikit-learn, xgboost |
| [praticas/pratica-5](praticas/pratica-5) | Rede neural (MLP) implementada do zero com NumPy | numpy, scikit-learn |
| [praticas/pratica-6](praticas/pratica-6) | Redes neurais convolucionais (CIFAR-10 e Cats vs Dogs) | TensorFlow/Keras |

> Não há prática 4 no material de origem.

## Trabalhos práticos (em dupla: um colega e Pedro Vieira)

| Pasta | Tema | Ferramentas |
|---|---|---|
| [trabalhos/tp1](trabalhos/tp1) | Kaggle *Store Sales – Time Series Forecasting* (regressão, stacking) | pandas, scikit-learn |
| [trabalhos/tp2](trabalhos/tp2) | Kaggle *State Farm Distracted Driver Detection* (CNN, ResNet-18, stacking) | PyTorch, torchvision |

## Ambiente

Os notebooks foram executados no Google Colab (práticas, TP1) e localmente com GPU (TP2). Para rodar localmente:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install jupyter pandas numpy matplotlib seaborn scikit-learn xgboost
# pratica-6: pip install tensorflow tensorflow-datasets
# tp2:       pip install torch torchvision pillow
```

Os dados devem ficar na mesma pasta do notebook (os caminhos são relativos).
