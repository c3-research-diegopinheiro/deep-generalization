from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications import VGG16, ResNet50

model_configs = [
    {
        "name": "cnn",
        "batch_size": 15,
        "alpha": 1e-3,
        "epochs": 30,
        "input_shape": (200, 200, 3),
        "layers": [
            Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=(200, 200, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (5, 5), padding="same", activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid'),
        ]
    },
    {
        "name": "mlp",
        "batch_size": 15,
        "alpha": 1e-3,
        "epochs": 30,
        "input_shape": (200, 200, 3),
        "layers": [
            Flatten(input_shape=(200, 200, 3)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid'),
        ]
    },
    {
        "name": "mlp-2",
        "batch_size": 15,
        "alpha": 1e-3,
        "epochs": 30,
        "input_shape": (200, 200, 3),
        "layers": [
            Flatten(input_shape=(200, 200, 3)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid'),
        ]
    },
]

