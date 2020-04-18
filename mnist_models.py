import tensorflow as tf
import numpy as np
import hpo.strategies.bayesian_method

optimiser = hpo.Optimiser(optimiser_name="optimiser_adam", optimiser_type=tf.keras.optimizers.Adam, hyperparameters=[
    hpo.Parameter(parameter_name="learning_rate", parameter_value=0.001,
                  value_range=[1 * (10 ** n) for n in range(0, -7, -1)])
])

single_dense_layer = [
    hpo.Layer(layer_name="input_layer_flatten", layer_type=tf.keras.layers.Flatten,
        hyperparameters=[],
        parameters=[
            hpo.Parameter(parameter_name="input_shape", parameter_value=(28, 28, 1))
        ]),

    hpo.Layer(layer_name="hidden_layer_1_dense", layer_type=tf.keras.layers.Dense,
        hyperparameters=[
            hpo.Parameter(parameter_name="units", parameter_value=16, value_range=[2**x for x in range(2, 16)]),#range between 4 and 512
            hpo.Parameter(parameter_name="activation", parameter_value="tanh", value_range=["relu", "tanh", "sigmoid", "softmax"], encode_string_values=True)#need to add more
        ],
        parameters=[

        ]),

    hpo.Layer(layer_name="hidden_layer_2_dropout", layer_type=tf.keras.layers.Dropout,
        hyperparameters=[
            hpo.Parameter(parameter_name="rate", parameter_value=0.5, value_range=np.arange(0.0, 0.5, 0.01).tolist())
        ],
        parameters=[
            hpo.Parameter(parameter_name="seed", parameter_value=42)
        ]),

    hpo.Layer(layer_name="output_layer_dense", layer_type=tf.keras.layers.Dense,
        hyperparameters=[
            hpo.Parameter(parameter_name="activation", parameter_value="tanh", value_range=["relu", "tanh", "sigmoid", "softmax"], encode_string_values=True)#need to add more
        ],
        parameters=[
            hpo.Parameter(parameter_name="units", parameter_value=10),
        ])]