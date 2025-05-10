import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

SEQ_LENGTH = 30
FEATURES = 48
NUM_CLASSES = 2  # 0 = incorrect, 1 = correct


def load_data(csv_path):
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].values.reshape(-1, SEQ_LENGTH, FEATURES)
    y = df.iloc[:, -1].values.astype(int)
    y = to_categorical(y, num_classes=NUM_CLASSES)
    return X, y


def build_softmax_model():
    model = Sequential([
        Input(shape=(SEQ_LENGTH, FEATURES)),
        Conv1D(64, kernel_size=3, activation='relu'),
        Dropout(0.3),
        Conv1D(128, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax') 
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    X, y = load_data("squat_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_softmax_model()
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

    model.save("squat_tcn_model.h5")
    print("Model trained and saved as squat_tcn_model.h5")