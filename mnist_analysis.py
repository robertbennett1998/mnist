import json
import os
import matplotlib.pyplot as plt
from hpo import *
import ray

from mnist_data import MnistData


def add_result_to_graph(result):
    pass


results = hpo_results.Results.load(os.path.join(os.getcwd(), "results.res"))
results.plot_average_score_over_optimisation_period()
results.plot_average_loss_over_optimisation_period()

best_result = results.best_result()
best_result.plot_train_val_accuracy()
best_result.plot_train_val_loss()
print("Best Model Information:")
print("\t Fitness: ", best_result.score())
print("\t Generation Number: ", best_result.meta_data()["Generation"])
print("\t Chromosome Number: ", best_result.meta_data()["Chromosome"])
print("\n\t", "Model Parameters:")
for hp in best_result.model_configuration().all_parameters():
    print("\t\t", hp.identifier(), "=", hp.value())

model = hpo_model.Model.from_model_configuration(best_result.model_configuration(), best_result.final_weights())
def construct_mnist_data():
    return MnistData(os.path.join(os.getcwd(), ".cache"), training_batch_size=1000, validation_batch_size=1000)

model.train(construct_mnist_data)