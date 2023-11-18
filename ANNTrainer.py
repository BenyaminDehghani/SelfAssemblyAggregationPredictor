import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import time
from OrthogonalVectors import vectorize
import os


def encode(x):
    encoded_x = []
    for i in range(len(x)):
        encoded_x.append(vectorize(x[i]))
    return encoded_x


def main():
    training_sequence_dataset_file_path = os.path.dirname(__file__) + "\\Dataset\\Tables\\Training sequence " \
                                                                      "dataset.xlsx"
    training_dataset = pd.read_excel(training_sequence_dataset_file_path)

    x = training_dataset["SEQUENCE"].tolist()
    y = training_dataset["AGGREGATION"].tolist()

    x = encode(x)
    x = np.array(x)
    y = np.array(y)

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, )
    x_validate, x_test, y_validate, y_test = train_test_split(x_temp, y_temp, test_size=0.5)

    num_runs = 100
    accuracy_scores = []

    t1 = time()
    for i in range(num_runs):
        model = keras.Sequential()
        model.add(keras.layers.Dense(units=20, input_dim=20, kernel_initializer='glorot_uniform'))
        model.add(keras.layers.Dense(units=7, activation='linear', kernel_initializer='glorot_uniform'))
        model.add(keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=20, batch_size=6, validation_data=(x_validate, y_validate))

        plt.plot(history.history['loss'])
        plt.ylabel('binary_cross entropy')
        plt.xlabel('epochs')
        plt.show()

        y_pred_prob = model.predict(x_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        if accuracy == max(accuracy_scores):
            model.save('ANNModel.keras')
        del model

    t2 = time()

    print(f"Max Test Accuracy: {max(accuracy_scores)}")

    std_dev = np.std(accuracy_scores)
    print(f'Standard Deviation of Accuracy: {std_dev}')

    print(f'Runtime: {t2-t1} second')


if __name__ == '__main__':
    main()
