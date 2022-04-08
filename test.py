from utils.results_writer import write_metrics_results
from model_configs import model_configs
from builders import dataset_builder, metrics_builder, model_builder


for model_config in model_configs:
    dataset_builder.generate_dataset(model_config)
    batch_size = model_config['batch_size']
    alpha = model_config['alpha']
    epochs = model_config['epochs']
    input_shape = model_config['input_shape']
    layers = model_config['layers']

    history, model, train_images, validation_images, test_images = model_builder.generate_model(
        model_config['name'], input_shape, batch_size, alpha, epochs, layers
    )

    cm = metrics_builder.generate_confusion_matrix(model, test_images, batch_size)
    cr = metrics_builder.generate_classification_report(model, test_images, batch_size)

    write_metrics_results(model_config['name'], cr, cm)
