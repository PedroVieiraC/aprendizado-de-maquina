# TP2: State Farm Distracted Driver Detection

**Notebook:** `TP2_State_Farm_Distracted_Driver_Detection.ipynb`, feito em dupla por um colega e Pedro Vieira.

## Objetivo
Classificar imagens de motoristas em 10 comportamentos (c0 condução normal … c9 conversando com o passageiro), na competição Kaggle *State Farm Distracted Driver Detection*. A métrica é o *multi-class log loss*.

## Dados (não incluídos)
https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data (`driver_imgs_list.csv` e as pastas `imgs/train/c0..c9` e `imgs/test`, com 79.726 imagens de teste). As regras da competição não permitem redistribuir.

## Método
- Divisão por **motorista** (20 no treino e 6 na validação) para evitar vazamento entre imagens da mesma pessoa.
- **Baseline:** CNN simples com 4 blocos Conv + BatchNorm + MaxPool, Adam (`lr=0.001`), 10 épocas.
- **Solução própria:** CNN residual customizada + ResNet-18 treinadas com `GroupKFold` (10 folds, agrupando por motorista), e um meta-modelo Random Forest (stacking) sobre as probabilidades out-of-fold.
- Mapas de saliência (gradiente e oclusão) para interpretar os modelos.

## Resultados (registrados no notebook)
| Modelo | Loss treino | Acc treino | Loss validação | Acc validação |
|---|---|---|---|---|
| Baseline (SimpleCNN) | 0.0412 | 98.81% | 1.2406 | 72.08% |
| Stacking (CNN residual + ResNet-18 → RF) | 0.5303 | 99.85% | 0.7166 | 88.60% |

As submissões ao Kaggle aparecem como capturas de tela (anexos) no notebook.

## Estado
Executado localmente com GPU (RTX 2060 SUPER). A célula 14 tem um `NameError` (`model` não definido) por ordem de execução. Os pesos `.pth` (`best_baseline_model.pth`, `custom_model_fold_*.pth`, `resnet_model_fold_*.pth`) não estão no repositório, então as células que os carregam exigem treinar de novo.

## Como reproduzir
1. Baixe e extraia os dados da competição nesta pasta (`driver_imgs_list.csv` e `imgs/`).
2. Execute em ordem. O treino dos 10 folds gera os `.pth`, e o código pula folds já treinados.

## Dependências
torch, torchvision, pillow, pandas, numpy, matplotlib, scikit-learn
