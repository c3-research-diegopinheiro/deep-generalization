import os
import shutil
import unittest
from utils.mkdir_dataset import mkdir_dataset


class UtilsTest(unittest.TestCase):

    def test_should_generate_dataset_initial_structure(self):
        dataset_test_name = 'dataset_test_name'

        mkdir_dataset(dataset_test_name)

        for dataset_type in ['train', 'validation', 'test']:
            self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/' + dataset_type), True)
            for classification in ['yes', 'no']:
                self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/' + dataset_type + '/' + classification), True)

        try:
            shutil.rmtree(f'DATASET/{dataset_test_name}')
        except OSError as e:
            print(e)

    def test_should_cant_generate_dataset_that_already_exists(self):
        dataset_test_name = 'dataset_test_name'

        mkdir_dataset(dataset_test_name)

        has_been_twice_created = mkdir_dataset(dataset_test_name)
        self.assertFalse(has_been_twice_created)

        try:
            shutil.rmtree(f'DATASET/{dataset_test_name}')
        except OSError as e:
            print(e)
