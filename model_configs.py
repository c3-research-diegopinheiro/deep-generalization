from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications import VGG16, ResNet50

model_configs = [
    {
        "name": "simple-mlp",
        "batch_size": 15,
        "alpha": 1e-3,
        "epochs": 30,
        "input_shape": (200, 200, 3),
        "layers": [
            Flatten(input_shape=(200, 200, 3)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid'),
        ]
    },
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
        "name": "vgg16",
        "batch_size": 15,
        "alpha": 1e-3,
        "epochs": 30,
        "input_shape": (200, 200, 3),
        "layers": [
            VGG16(
                include_top=False,
                input_shape=(200, 200, 3)
            ),
            Flatten(),
            Dropout(0.5),
            Dense(1, activation='sigmoid'),
        ]
    },
    {
        "name": "resnet50",
        "batch_size": 15,
        "alpha": 1e-3,
        "epochs": 30,
        "input_shape": (200, 200, 3),
        "layers": [
            ResNet50(
                include_top=False,
                input_shape=(200, 200, 3)
            ),
            Flatten(),
            Dropout(0.5),
            Dense(1, activation='sigmoid'),
        ]
    },
    {
        "name": "simple-mlp-2",
        "batch_size": 15,
        "alpha": 1e-3,
        "epochs": 30,
        "input_shape": (200, 200, 3),
        "layers": [
            Flatten(input_shape=(200, 200, 3)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid'),
        ]
    },
    {
        "name": "simple-mlp-3",
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
]

