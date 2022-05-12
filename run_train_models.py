import traceback

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from builders import metrics_builder, model_builder
from utils.results_writer import ResultsWriter
from model_configs import model_configs

model_config = model_configs[3]
try:
    rw = ResultsWriter(model_config['name'])
    for noise_amount in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
        trained_model = model_builder.train_model_for_dataset(
            model_config, f'dataset/train_{noise_amount}', f'dataset/validation_{noise_amount}'
        )
        rw.write_model(trained_model, f'train_{noise_amount}')

        for noise_amount_testing in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
            test_images = ImageDataGenerator(rescale=1 / 255).flow_from_directory(
                f'DATASET/test_{noise_amount_testing}',
                target_size=(200, 200),
                shuffle=False,
                class_mode='binary',
                batch_size=model_configs[0]['batch_size'],
                save_to_dir=None)

            trained_model.evaluate(test_images)

            cm = metrics_builder.generate_confusion_matrix(trained_model, test_images, model_config['batch_size'])
            cr = metrics_builder.generate_classification_report(trained_model, test_images, model_config['batch_size'])

            rw.write_metrics_results(
                f'train_{noise_amount}_test_{noise_amount_testing}',
                noise_amount,
                noise_amount_testing,
                cr,
                cm
            )

except Exception as e:
    traceback.print_tb(e.__traceback__)
    print(e.__str__())
    print('Removing results folder for this execution')
    rw.delete_results()
