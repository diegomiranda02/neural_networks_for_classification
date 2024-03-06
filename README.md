Classificador de Texto usando Redes Neurais Convolucionais (CNN)

Este repositório contém código para treinar um classificador de texto usando Redes Neurais Convolucionais (CNN) com o TensorFlow. O modelo é treinado em dados de texto previamente processados e categorizados.

Visão Geral

O código fornecido demonstra como treinar um modelo de classificação de texto usando Redes Neurais Convolucionais (CNN) com o TensorFlow. O processo inclui pré-processamento dos dados, criação do modelo CNN, treinamento e avaliação utilizando a k-fold cross-validation.

Dependências

- Python 3.x
- pandas
- numpy
- tensorflow
- scikit-learn

Observação
- Há 4 arquivos:
- classifier_base_kfold.py
- classifier_base.py
- classifier_kfold.py
- classifier.py

Os arquivos classifier_base_kfold.py e classifier_base.py estão com a implementação da rede neural do CNJ. Já os arquivos classifier_kfold.py e classifier.py já estão com a implementação feita pelo TRE-RN. O ideal é treinar os dados balanceados e não balanceados com todos os códigos e verificar as métricas. 

- O modelo será treinado usando a k-fold cross-validation com 5 folds.

- As métricas de avaliação, incluindo acurácia, precisão, recall e F1-score, serão impressas para cada fold.

- No final, a média das métricas de avaliação será calculada e exibida.

- Certifique-se de ter pré-processado seu conjunto de dados conforme necessário antes de executar o script.

- Certifique-se de ter recursos de hardware adequados para treinar o modelo, especialmente se o conjunto de dados for grande e/ou o modelo for complexo.

- Ajuste os parâmetros do modelo e do processo de treinamento conforme necessário para obter os melhores resultados para o seu conjunto de dados específico.
