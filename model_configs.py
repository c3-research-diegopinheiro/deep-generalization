from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense
import os

model_configs = [
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
        "name": "noisy_20%_only_train_validation",
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
                "amount": 0.2
            },
            "validation": {
                "noise": True,
                "amount": 0.2
            },
            "test": {
                "noise": False,
                "amount": 0
            }
        }
    },
    {
        "name": "noisy_20%_all_dataset",
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
                "amount": 0.2
            },
            "validation": {
                "noise": True,
                "amount": 0.2
            },
            "test": {
                "noise": True,
                "amount": 0.2
            }
        }
    },
]
