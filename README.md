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

Avaliação dos resultados

Esses resultados são provenientes da avaliação de modelos de classificação de texto em conjuntos de dados balanceados e desbalanceados. Os modelos são treinados usando um conjunto de dados desbalanceado e um conjunto de dados balanceado. Aqui está uma análise dos resultados:

Conjunto de Dados Balanceado vs. Desbalanceado

Balanceado: Conjunto de dados onde o número de amostras em cada classe é aproximadamente igual. Isso ajuda a evitar viés em direção às classes majoritárias durante o treinamento do modelo.

Desbalanceado: Um conjunto de dados onde o número de amostras em uma ou mais classes é significativamente maior ou menor do que em outras classes. Isso pode levar a modelos com baixo desempenho em classes minoritárias.


Conjunto de Dados Balanceado:



Modelo Base do CNJ:

O modelo obteve uma precisão (precision) alta para a classe "aprovada", mas baixa para "aprovada_com_ressalvas" e "desaprovada".
O recall é relativamente alto para todas as classes, mas baixo para "aprovada_com_ressalvas".
O F1-score é moderado para "aprovada" e "desaprovada", mas baixo para "aprovada_com_ressalvas".
O modelo obteve uma precisão média e um F1-score moderado, mas com espaço para melhorias.

Classification Report:
                         precision    recall  f1-score   support

              aprovada       0.82      0.94      0.88        34
aprovada_com_ressalvas       0.00      0.00      0.00        22
           desaprovada       0.47      0.86      0.61        21

              accuracy                           0.65        77
             macro avg       0.43      0.60      0.50        77
          weighted avg       0.49      0.65      0.55        77

MODELO BASE DO CNJ KFOLD:


Average Accuracy across all folds: 0.5801
Average Precision across all folds: 0.6043
Average Recall across all folds: 0.5801
Average F1-Score across all folds: 0.5394



Modelo Customizado:

Este modelo melhorou as métricas de precisão, recall e F1-score para todas as classes em comparação com o modelo base do CNJ.
A precisão e o F1-score para "aprovada_com_ressalvas" e "desaprovada" foram notavelmente melhorados.
Apesar da alta precisão e F1-score para "aprovada", houve uma redução na precisão para "desaprovada", que pode ser devido a um desequilíbrio nas previsões.


Classification Report:
                         precision    recall  f1-score   support

              aprovada       0.70      0.91      0.79        34
aprovada_com_ressalvas       0.58      0.82      0.68        22
           desaprovada       1.00      0.10      0.17        21

              accuracy                           0.66        77
             macro avg       0.76      0.61      0.55        77
          weighted avg       0.75      0.66      0.59        77
          


Average Accuracy across all folds: 0.7507
Average Precision across all folds: 0.7632
Average Recall across all folds: 0.7507
Average F1-Score across all folds: 0.7519


Conjunto de Dados Desbalanceado:

Modelo Base do CNJ:

O modelo tem uma precisão muito alta para a classe "aprovada", mas baixa para "aprovada_com_ressalvas".
O recall é alto para "aprovada" e "desaprovada", mas muito baixo para "aprovada_com_ressalvas".
O F1-score é alto para "aprovada" e "desaprovada", mas muito baixo para "aprovada_com_ressalvas".
Como esperado, o modelo não generaliza bem para a classe minoritária "aprovada_com_ressalvas" devido ao desbalanceamento dos dados.


Classification Report:
                         precision    recall  f1-score   support

              aprovada       0.96      0.88      0.92        26
aprovada_com_ressalvas       0.00      0.00      0.00         7
           desaprovada       0.72      1.00      0.84        23

              accuracy                           0.82        56
             macro avg       0.56      0.63      0.59        56
          weighted avg       0.74      0.82      0.77        56
          

MODELO BASE DO CNJ KFOLD:

Average Accuracy across all folds: 0.6844
Average Precision across all folds: 0.6223
Average Recall across all folds: 0.6844
Average F1-Score across all folds: 0.6274


Modelo Customizado:

Este modelo melhorou significativamente as métricas de precisão, recall e F1-score para todas as classes em comparação com o modelo base do CNJ.
A precisão para "aprovada_com_ressalvas" foi notavelmente melhorada, mas ainda é baixa devido à escassez de dados dessa classe.
O modelo mostra uma capacidade melhorada de generalização para classes minoritárias, como "aprovada_com_ressalvas", em comparação com o modelo base do CNJ.


Classification Report:
                         precision    recall  f1-score   support

              aprovada       0.66      0.96      0.78        26
aprovada_com_ressalvas       0.00      0.00      0.00         7
           desaprovada       0.72      0.57      0.63        23

              accuracy                           0.68        56
             macro avg       0.46      0.51      0.47        56
          weighted avg       0.60      0.68      0.62        56
          
MODELO CUSTOMIZADO KFOLD:


Average Accuracy across all folds: 0.6954
Average Precision across all folds: 0.6987
Average Recall across all folds: 0.6954
Average F1-Score across all folds: 0.6780

Conclusão:
O modelo customizado superou o modelo base do CNJ em ambos os conjuntos de dados, balanceado e desbalanceado.
A balanceamento do conjunto de dados teve um impacto significativo no desempenho do modelo, especialmente para classes minoritárias.
No entanto, mesmo com o conjunto de dados balanceado, o modelo ainda enfrenta desafios na classificação da classe "aprovada_com_ressalvas" devido à sua natureza desbalanceada. Isso indica a necessidade de mais dados ou técnicas de modelagem adicionais para melhorar o desempenho nessa classe.


