import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


SEQ_LENGTH = 30
FEATURES = 170
NUM_CLASSES = 5


def load_and_merge_features():
    angles = pd.read_csv("../backend/existing_data/angles.csv")
    distances = pd.read_csv("../backend/existing_data/calculated_3d_distances.csv")
    landmarks = pd.read_csv("../backend/existing_data/landmarks.csv")
    xyz = pd.read_csv("../backend/existing_data/xyz_distances.csv")
    labels = pd.read_csv("../backend/existing_data/labels.csv")

    features = pd.concat([angles, distances.drop(columns=['vid_id', 'frame_order']),
                          landmarks.drop(columns=['vid_id', 'frame_order']),
                          xyz.drop(columns=['vid_id', 'frame_order'])], axis=1)
    return features, labels


def create_sequences(X_df, y_df, seq_len=SEQ_LENGTH):
    sequences = []
    labels = []
    label_encoder = LabelEncoder()
    label_encoder.fit(y_df['class'])

    X_df = X_df.sort_values(['vid_id', 'frame_order']).reset_index(drop=True)

    feature_cols = [col for col in X_df.columns if col not in ['vid_id', 'frame_order']]
    global FEATURES
    FEATURES = len(feature_cols)

    grouped = X_df.groupby('vid_id')

    for vid_id, group in grouped:
        group_features = group[feature_cols].values

        video_label = y_df[y_df['vid_id'] == vid_id]['class'].values[0]
        encoded_label = label_encoder.transform([video_label])[0]

        for i in range(len(group_features) - seq_len + 1):
            seq = group_features[i:i + seq_len]
            sequences.append(seq)
            labels.append(encoded_label)

    global NUM_CLASSES
    NUM_CLASSES = len(label_encoder.classes_)

    X = np.array(sequences)
    y = to_categorical(np.array(labels), num_classes=NUM_CLASSES)

    return X, y, label_encoder


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
    print("Loading and merging features.")
    X_df, y_df = load_and_merge_features()

    print("Raw label distribution.")
    print(y_df['class'].value_counts())

    X, y, label_encoder = create_sequences(X_df, y_df)
    from collections import Counter

    print("\nClass distribution after sequencing:")
    y_raw = np.argmax(y, axis=1)
    counts = Counter(y_raw)
    for idx, count in counts.items():
        print(f"{label_encoder.classes_[idx]}: {count}")

    print(f"Loaded {X.shape[0]} sequences with shape {X.shape[1:]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training model.")
    model = build_softmax_model()
    model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))
    model.save("version1_tcn_model.h5")
    print("Model saved as version1_tcn_model.h5")

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))

