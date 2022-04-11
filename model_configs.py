from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
import os

model_configs = [
    {
        "name": "default_dataset_model",
        "env": os.uname().nodename,
        "batch_size": 15,
        "alpha": 1e-3,
        "epochs": 30,
        "input_shape": (200, 200, 3),
        "layers": [
            Flatten(input_shape=(200, 200, 3)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        "dataset_structure": {
            "train": {
                "noise": False,
                "amount": 0
            },
            "validation": {
                "noise": False,
                "amount": 0
            },
            "test": {
                "noise": False,
                "amount": 0
            }
        }
    },
    {
        "name": "noisy_10%_all_dataset",
        "env": os.uname().nodename,
        "batch_size": 15,
        "alpha": 1e-3,
        "epochs": 30,
        "input_shape": (200, 200, 3),
        "layers": [
            Flatten(input_shape=(200, 200, 3)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        "dataset_structure": {
            "train": {
                "noise": True,
                "amount": 0.1
            },
            "validation": {
                "noise": True,
                "amount": 0.1
            },
            "test": {
                "noise": True,
                "amount": 0.1
            }
        }
    },
    {
        "name": "noisy_10%_only_train_validation",
        "env": os.uname().nodename,
        "batch_size": 15,
        "alpha": 1e-3,
        "epochs": 30,
        "input_shape": (200, 200, 3),
        "layers": [
            Flatten(input_shape=(200, 200, 3)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        "dataset_structure": {
            "train": {
                "noise": True,
                "amount": 0.1
            },
            "validation": {
                "noise": True,
                "amount": 0.1
            },
            "test": {
                "noise": False,
                "amount": 0
            }
        },
    },
    {
        "name": "cnn_noisy_10%_only_train_validation",
        "env": os.uname().nodename,
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
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid'),
        ],
        "dataset_structure": {
            "train": {
                "noise": True,
                "amount": 0.1
            },
            "validation": {
                "noise": True,
                "amount": 0.1
            },
            "test": {
                "noise": False,
                "amount": 0
            }
        },
    },
]
