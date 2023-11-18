import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score
from OrthogonalVectors import vectorize
import os


def encode(x):
    encoded_x = []
    for i in range(len(x)):
        sub_sequences = []
        for j in range(len(x[i])-5):
            sub_sequence = x[i][j:j+6]
            sub_sequences.append(vectorize(sub_sequence))
        encoded_x.append(sub_sequences)
    return encoded_x


def main():
    external_validation_sequence_dataset_file_path = os.path.dirname(__file__) + "\\External validation dataset.xlsx"
    external_validation_sequence_dataset = pd.read_excel(external_validation_sequence_dataset_file_path)
    x = external_validation_sequence_dataset["SEQUENCE"].tolist()
    y = external_validation_sequence_dataset["AGGREGATION"].tolist()

    x = encode(x)

    model_file_path = os.path.dirname(__file__) + "\\ANNModel.keras"

    model = keras.models.load_model(model_file_path)

    predicted_aggregations = []

    for i in range(len(x)):
        peptide = x[i]
        sequences = np.array(peptide)
        y_pred_prob = model.predict(sequences)
        y_pred = (y_pred_prob > 0.5).astype(int)
        predicted_aggregation = 1 if ([1] in y_pred.tolist()) else 0
        predicted_aggregations.append(predicted_aggregation)

    print(y)
    print(predicted_aggregations)
    accuracy = accuracy_score(predicted_aggregations, y)
    print(f"External Validation Accuracy: {accuracy}")


if __name__ == '__main__':
    main()
