from builders import dataset_builder
import itertools


def generate_dataset_structure(noises):
    train_noise, test_noise = noises

    return dict({
        "name": f"train_{train_noise}_test_{test_noise}".lower(),
        "dataset_structure": {
            "train": {
                "noise": True,
                "amount": train_noise
            },
            "validation": {
                "noise": True,
                "amount": train_noise
            },
            "test": {
                "noise": True,
                "amount": test_noise
            }
        }})


noise_product = itertools.product([0, .1, .2, .3, .4, .5, .6, .7, .8, .9], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])

datasets = [generate_dataset_structure(noises) for noises in list(noise_product)]

for dataset in datasets:
    dataset_builder.generate_dataset(dataset)
