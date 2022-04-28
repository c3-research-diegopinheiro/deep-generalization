from builders.dataset_builder import generate_dataset


for dataset_kind in ['train', 'test', 'validation']:
    for noise_amount in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
        dataset_name = f'{dataset_kind}_{noise_amount}'
        generate_dataset(dataset_name, dataset_kind, noise_amount)
