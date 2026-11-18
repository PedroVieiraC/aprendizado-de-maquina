# Prática 6: Redes Neurais Convolucionais (CNN)

**Notebooks:** `Cópia de CNN_CIFAR10_Prática.ipynb` e `cats_dogs_Pedro.ipynb`

## Objetivo
Classificar imagens com CNNs em Keras: multiclasse (CIFAR-10) e binária (Cats vs Dogs).

## Dados
Baixados automaticamente pelas bibliotecas, sem arquivos no repositório:
- CIFAR-10: `tensorflow.keras.datasets.cifar10` (https://www.cs.toronto.edu/~kriz/cifar.html)
- Cats vs Dogs: `tensorflow_datasets` (`cats_vs_dogs`, https://www.tensorflow.org/datasets/catalog/cats_vs_dogs), divisão 80/20

## Método
- **CIFAR-10:** CNN com Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(128) (treino interrompido), e um modelo só com Flatten + Dense(10) softmax, otimizador SGD, 10 épocas.
- **Cats vs Dogs:** imagens redimensionadas para 128×128, CNN com BatchNormalization e Dropout, otimizador Adam, perda binária, 10 épocas. O carregamento e o pré-processamento vieram do professor.

## Resultados (registrados nos notebooks)
- CIFAR-10 (modelo Flatten + Dense): acurácia no teste de **0.3813**.
- Cats vs Dogs: acurácia de **0.85** (4.652 imagens de teste; F1 de 0.85 para gato e para cachorro).

## Estado
O notebook do CIFAR-10 está **parcial**: a CNN convolucional foi interrompida (KeyboardInterrupt) e as células de avaliação dela não foram executadas. Os resultados acima são do modelo simplificado.

## Como reproduzir
Execute no Colab com GPU (Ambiente de execução → Alterar tipo → GPU). O cats_dogs instala `tensorflow-datasets` na 1ª célula.

## Dependências
tensorflow, tensorflow-datasets, numpy, matplotlib, seaborn, scikit-learn
