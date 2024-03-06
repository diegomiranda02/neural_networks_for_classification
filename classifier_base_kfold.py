import pandas as pd
import re
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import KFold

# Arquivo onde irá ser armazenado o modelo treinado com o conjunto de dados
modelo_arquivo = '/home/sinapses-ia/projects/janus-ia/model/cnn_embedding.h5'

# Função para limpeza dos dados
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Função para criação do modelo
def create_model(input_dim, output_dim, max_length, num_classes):

    modelo = Sequential()
    modelo.add(layers.Embedding(input_dim=input_dim
                                , output_dim=output_dim
                                , input_length=max_length
                                , trainable=True))
    modelo.add(layers.Conv1D(128, 5, activation='relu'))
    modelo.add(layers.GlobalMaxPooling1D())
    modelo.add(layers.Dense(128, activation='relu'))
    modelo.add(layers.Dropout(rate=0.4))
    modelo.add(layers.Dense(64, activation='relu'))
    modelo.add(layers.Dropout(rate=0.4))
    modelo.add(layers.Dense(10, activation='relu'))
    modelo.add(layers.Dense(num_classes, activation='softmax'))
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return modelo

# Processamento e carregamento do dataset
DATASET = 'CAMINHO PARA O ARQUIVO CSV COM OS DADOS'
df = pd.read_csv(DATASET)
df['text'] = df['text'].apply(clean_text)

# Encode dos labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
labels = to_categorical(df['label_encoded'])

# Tokenizando os textos
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
max_len = max(len(s) for s in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Preparando para k-fold cross-validation
vocab_size = len(tokenizer.word_index)
num_classes = labels.shape[1]
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Lista para armazenar os resultados de cada etapa do kfold cross-validation
cv_f1_scores = []
cv_precision_scores = []
cv_recall_scores = []
cv_accuracy_scores = []

for train_idx, val_idx in kfold.split(padded_sequences):
    X_train_fold, X_val_fold = padded_sequences[train_idx], padded_sequences[val_idx]
    y_train_fold, y_val_fold = labels[train_idx], labels[val_idx]
    
    modelo = create_model(vocab_size + 1, output_dim=400, max_length=max_len, num_classes=num_classes)
    
    early_stopping = EarlyStopping(patience=4)
    model_checkpoint = ModelCheckpoint(filepath=str(modelo_arquivo), monitor='val_accuracy', save_best_only=True)

    history = modelo.fit(
        X_train_fold,
        y_train_fold,
        epochs=2,
        verbose=True,
        validation_data=(X_val_fold, y_val_fold),
        batch_size=100,
        callbacks=[model_checkpoint, early_stopping]
    )

    # Avaliar o modelo 
    scores = modelo.evaluate(X_val_fold, y_val_fold, verbose=0)

    # Predições
    predictions = modelo.predict(X_val_fold)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_val_fold, axis=1)
    
    # Calculando as métricas
    f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=1)
    precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=1)
    recall = recall_score(true_classes, predicted_classes, average='weighted', zero_division=1)
    accuracy = scores[1]

    # Adicionando as métricas às listas
    cv_f1_scores.append(f1)
    cv_precision_scores.append(precision)
    cv_recall_scores.append(recall)
    cv_accuracy_scores.append(accuracy)
    
    print(f"Fold - Loss: {scores[0]}, Accuracy: {scores[1]}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Generando o relatório
    target_names = label_encoder.classes_
    report = classification_report(true_classes, predicted_classes, target_names=target_names)
    print("Classification Report:\n", report)

# Cálculo da média das métricas
print(f"\nMédia da acurácia: {np.mean(cv_accuracy_scores):.4f}")
print(f"Média da Precision: {np.mean(cv_precision_scores):.4f}")
print(f"Média da Recall: {np.mean(cv_recall_scores):.4f}")
print(f"Média da F1-Score: {np.mean(cv_f1_scores):.4f}")
