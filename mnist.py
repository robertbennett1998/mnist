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

fig, (raw_scores_axis, average_score_axis) = plt.subplots(1, 2)
raw_scores = list()
average_scores = list()
total = 0
count = 0
def add_result_to_graph(result):
    global total
    global count
    raw_scores.append(result.score())
    total += result.score()
    count += 1
    average_scores.append(total / count)

def update_graph(i):
    raw_scores_axis.clear()
    raw_scores_axis.set_title("Score for each result.")
    raw_scores_axis.plot(raw_scores)
    average_score_axis.clear()
    average_score_axis.set_title("Average score over optimisation period.")
    average_score_axis.plot(average_scores)

optimiser = hpo.Optimiser(optimiser_name="optimiser_adam", optimiser_type=tf.keras.optimizers.Adam, hyperparameters=[
    hpo.Parameter(parameter_name="learning_rate", parameter_value=0.001,
                  value_range=[1 * (10 ** n) for n in range(0, -7, -1)])
])

model_configuration = hpo.ModelConfiguration(optimiser=optimiser, layers=model_layers, loss_function="categorical_crossentropy", number_of_epochs=10)

def construct_mnist_data():
    return MnistData(os.path.join(os.getcwd(), ".cache"), training_batch_size=1000, validation_batch_size=1000)


def construct_chromosome():
    return hpo.strategies.genetic_algorithm.DefaultChromosome(model_configuration)

best_result = None
results = None

def run():
    global best_result
    global results
    strategy = hpo.strategies.genetic_algorithm.GeneticAlgorithm(population_size=30, max_iterations=30, chromosome_type=construct_chromosome)
    strategy.mutation_strategy().mutation_probability(0.15)
    strategy.survivour_selection_strategy().threshold(0.7)

    hpo_instance = hpo.Hpo(model_configuration, construct_mnist_data, strategy)
    best_result, results = hpo_instance.execute(result_added_hook=add_result_to_graph)

hpo_thread = threading.Thread(target=run)
hpo_thread.start()

ani = animation.FuncAnimation(fig, update_graph, interval=1000)
plt.show()

hpo_thread.join()

best_result.plot_train_val_accuracy()
best_result.plot_train_val_loss()
results.save(os.path.join(os.getcwd(), "results.res"))