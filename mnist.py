from mnist_data import MnistData
import os
import hpo.strategies.bayesian_method
import hpo.strategies.genetic_algorithm
import mnist_models
import hpo_experiment_runner


model_configuration = hpo.ModelConfiguration(optimiser=mnist_models.optimiser, layers=mnist_models.single_dense_layer, loss_function="categorical_crossentropy", number_of_epochs=10)


def construct_mnist_data():
    return MnistData(os.path.join(os.getcwd(), ".cache"), training_batch_size=1000, validation_batch_size=1000)


def construct_chromosome():
    return hpo.strategies.genetic_algorithm.DefaultChromosome(model_configuration)


#####################################
# Bayesian Selection - Random Forest
#####################################
strategy = hpo.strategies.bayesian_method.BayesianMethod(model_configuration, 200, hpo.strategies.bayesian_method.RandomForestSurrogate())
hpo_instance = hpo.Hpo(model_configuration, construct_mnist_data, strategy)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "hpo_bayesian_random_forest.results"))

########################################
# Bayesian Selection - Gaussian Process
########################################
strategy = hpo.strategies.bayesian_method.BayesianMethod(model_configuration, 900, hpo.strategies.bayesian_method.GaussianProcessSurrogate())
hpo_instance = hpo.Hpo(model_configuration, construct_mnist_data, strategy)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "hpo_bayesian_gaussian_process.results"))

#########################################
# Genetic Algorithm - Roulette Selection
##########################################
strategy = hpo.strategies.genetic_algorithm.GeneticAlgorithm(population_size=30, max_iterations=30, chromosome_type=construct_chromosome,
                                            survivour_selection_stratergy="roulette")
strategy.mutation_strategy().mutation_probability(0.05)
strategy.survivour_selection_strategy().survivour_percentage(0.7)
hpo_instance = hpo.Hpo(model_configuration, construct_mnist_data, strategy)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "hpo_genetic_algorithm_roulette.results"))

#########################################
# Genetic Algorithm - Threshold Selection
##########################################
strategy = hpo.strategies.genetic_algorithm.GeneticAlgorithm(population_size=30, max_iterations=30, chromosome_type=construct_chromosome,
                                            survivour_selection_stratergy="threshold")
strategy.mutation_strategy().mutation_probability(0.05)
strategy.survivour_selection_strategy().threshold(0.9)
hpo_instance = hpo.Hpo(model_configuration, construct_mnist_data, strategy)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "hpo_genetic_algorithm_threshold.results"))