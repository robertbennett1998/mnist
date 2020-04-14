import tensorflow as tf
import os
import numpy as np
import hpo
import shutil
from tensorflow.keras import backend as K

class MnistData(hpo.Data):
    def __init__(self, cache_path, training_batch_size=100, validation_batch_size=100, test_batch_size=100):
        super().__init__()

        self._cache_path = cache_path
        self._training_data_x_path = os.path.join(self._cache_path, "_training_data_x.npy")
        self._training_data_y_path = os.path.join(self._cache_path, "_training_data_y.npy")
        self._validation_data_x_path = os.path.join(self._cache_path, "_validation_data_x.npy")
        self._validation_data_y_path = os.path.join(self._cache_path, "_validation_data_y.npy")

        self._training_batch_size = training_batch_size
        self._validation_batch_size = validation_batch_size
        self._test_batch_size = test_batch_size
        
        self._class_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self._img_width = 28
        self._img_height = 28

        self._training_image_count = 0
        self._validation_image_count = 0
        self._test_image_count = 0
        
        self._training_data = None
        self._valdiation_data = None
        self._test_data = None

    def load(self):
        def prepare_dataset(dataset, batch_size, cache=True, repeat=True, prefetch=True, shuffle=True, shuffle_seed=42, shuffle_buffer_size=1000):
            if (cache):
                if (isinstance(cache, str)):
                    print("Opening cache or creating (%s)." % (cache))
                    dataset = dataset.cache(cache)
                else:
                    print("No cache path provided. Loading into memory.")
                    dataset = dataset.cache()
            else:
                print("Not caching data. This may be slow.")

            if shuffle:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)

            if repeat:
                dataset = dataset.repeat()

            if batch_size > 0:
                dataset = dataset.batch(batch_size)

            if prefetch:
                dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            return dataset

        if not os.path.exists(self._cache_path):
            os.mkdir(self._cache_path)

        training_data_x = None
        training_data_x = None
        valdiation_data_x = None
        valdiation_data_y = None
        if not os.path.exists(self._training_data_x_path) or not os.path.exists(self._training_data_y_path) or not os.path.exists(self._validation_data_x_path) or not os.path.exists(self._validation_data_y_path):
            shutil.rmtree(self._cache_path)
            os.mkdir(self._cache_path)

            (training_data_x, training_data_y), (valdiation_data_x, valdiation_data_y) = tf.keras.datasets.mnist.load_data()
        
            training_data_x = np.array(training_data_x).astype('float32')
            training_data_y = np.array(training_data_y).astype('float32')
            valdiation_data_x = np.array(valdiation_data_x).astype('float32')
            valdiation_data_y = np.array(valdiation_data_y).astype('float32')

            np.save(self._training_data_x_path, training_data_x / 255)
            np.save(self._training_data_y_path, training_data_y / 255)
            np.save(self._validation_data_x_path, valdiation_data_x / 255)
            np.save(self._validation_data_y_path, valdiation_data_y / 255)
        else:
            training_data_x = np.load(self._training_data_x_path)
            training_data_y = np.load(self._training_data_y_path)
            valdiation_data_x = np.load(self._validation_data_x_path)
            valdiation_data_y = np.load(self._validation_data_y_path)

        self._training_image_count = training_data_x.shape[0]
        self._validation_image_count = valdiation_data_x.shape[0]

        training_data_y = [np.array(y == self._class_labels) for y in training_data_y]
        
        if K.image_data_format() == 'channels_first':
            training_data_x = training_data_x.reshape(self._training_image_count, 1, self._img_width, self._img_height)
            valdiation_data_x = valdiation_data_x.reshape(self._validation_image_count, 1, self._img_width, self._img_height)
        else:
            training_data_x = training_data_x.reshape(self._training_image_count, self._img_width, self._img_height, 1)
            valdiation_data_x = valdiation_data_x.reshape(self._validation_image_count, self._img_width, self._img_height, 1)

        self._training_data = tf.data.Dataset.from_tensor_slices((training_data_x, training_data_y))
        self._training_data = prepare_dataset(self._training_data, self._training_batch_size, os.path.join(self._cache_path, "training.tfcache"))

        valdiation_data_y = [np.array(y == self._class_labels) for y in valdiation_data_y]

        self._valdiation_data = tf.data.Dataset.from_tensor_slices((valdiation_data_x, valdiation_data_y))
        self._valdiation_data = prepare_dataset(self._valdiation_data, self._validation_batch_size, os.path.join(self._cache_path, "validation.tfcache"))

    def training_steps(self):
        return self._training_image_count // self._training_batch_size

    def validation_steps(self):
        return self._validation_image_count // self._validation_batch_size

    def test_steps(self):
        return self._test_image_count // self._test_batch_size

    def training_data(self):
        return self._training_data

    def validation_data(self):
        return self._valdiation_data

    def test_data(self):
        return self._test_data