from mnist_data import MnistData
import os
import hpo
import tensorflow as tf
import numpy as np
import hpo.strategies.genetic_algorithm
import ray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import threading

ray.init()

model_layers = [
    hpo.Layer(layer_name="input_layer_flatten", layer_type=tf.keras.layers.Flatten,
        hyperparameters=[],
        parameters=[]),

    hpo.Layer(layer_name="hidden_layer_1_dense", layer_type=tf.keras.layers.Dense,
        hyperparameters=[
            hpo.Parameter(parameter_name="units", parameter_value=16, value_range=[2**x for x in range(2, 16)], constraints=None),#range between 4 and 512
            hpo.Parameter(parameter_name="activation", parameter_value="tanh", value_range=["relu", "tanh", "sigmoid", "softmax"], constraints=None)#need to add more
        ],
        parameters=[

        ]),

    hpo.Layer(layer_name="hidden_layer_2_dropout", layer_type=tf.keras.layers.Dropout,
        hyperparameters=[
            hpo.Parameter(parameter_name="rate", parameter_value=0.5, value_range=np.arange(0.0, 0.5, 0.01).tolist(), constraints=None)
        ],
        parameters=[
            hpo.Parameter(parameter_name="seed", parameter_value=42)
        ]),

    hpo.Layer(layer_name="output_layer_dense", layer_type=tf.keras.layers.Dense,
        hyperparameters=[
            hpo.Parameter(parameter_name="activation", parameter_value="tanh", value_range=["relu", "tanh", "sigmoid", "softmax"], constraints=None)#need to add more
        ],
        parameters=[
            hpo.Parameter(parameter_name="units", parameter_value=10),
        ])]


def construct_mnist_data():
    return MnistData(os.path.join(os.getcwd(), ".cache"), training_batch_size=1000, validation_batch_size=1000)


def construct_chromosome():
    return hpo.strategies.genetic_algorithm.DefaultChromosome(model_configuration)


optimiser = hpo.Optimiser(optimiser_name="optimiser_adam", optimiser_type=tf.keras.optimizers.Adam, hyperparameters=[
    hpo.Parameter(parameter_name="learning_rate", parameter_value=0.001,
                  value_range=[1 * (10 ** n) for n in range(0, -7, -1)])
])

model_configuration = hpo.ModelConfiguration(optimiser=optimiser, layers=model_layers, loss_function="categorical_crossentropy", number_of_epochs=10)

strategy = hpo.strategies.genetic_algorithm.GeneticAlgorithm(population_size=30, max_iterations=30, chromosome_type=construct_chromosome)
strategy.mutation_strategy().mutation_probability(0.15)
strategy.survivour_selection_strategy().threshold(0.7)

hpo_instance = hpo.Hpo(model_configuration, construct_mnist_data, strategy)
best_result, results = hpo_instance.execute()

results.save(os.path.join(os.getcwd(), "results.res"))