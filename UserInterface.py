from tensorflow import keras
from OrthogonalVectors import vectorize
import numpy as np
import os


def encode(x):
    encoded_x = []
    for i in range(len(x)-5):
        sub_sequence = x[i:i+6]
        encoded_x.append(vectorize(sub_sequence))
    return encoded_x


def main():
    print("Welcome to Self Assembly Aggregation Predictor")
    print("Input the Amino Acid sequence of the peptide(minimum length=6):")
    sequence = input()

    x = encode(sequence)

    model_file_path = os.path.dirname(__file__) + "\\ANNModel.keras"

    model = keras.models.load_model(model_file_path)

    peptide = x
    sequences = np.array(peptide)
    y_pred_prob = model.predict(sequences, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    predicted_aggregation = 1 if ([1] in y_pred.tolist()) else 0

    if predicted_aggregation == 1:
        print("This peptide is likely to form aggregation.")
    else:
        print("This peptide is not likely to form aggregation.")

    probability = max(y_pred_prob)[0]
    print(f"The probability that this peptide forms aggregation equals to {probability}")


if __name__ == "__main__":
    main()
