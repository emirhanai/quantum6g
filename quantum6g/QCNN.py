import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AngleEmbedding, BasicEntanglerLayers
import tensorflow as tf

class QCNN:
    def __init__(
        self,
        input_shape,
        output_neurons,
        loss_function,
        epochs,
        batch_size,
        optimizer,
        n_layers,
        n_wires,
    ):
        self.input_shape = input_shape
        self.output_neurons = output_neurons
        self.loss_function = loss_function
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.n_layers = n_layers
        self.n_wires = n_wires

        self.dev = qml.device("default.mixed", wires=self.n_wires)
        self.q_circuit = qml.QNode(self.quantum_circuit, self.dev, interface="tf")

    def quantum_circuit(self, inputs, weights):
        AngleEmbedding(inputs, wires=range(self.n_wires))
        BasicEntanglerLayers(weights, wires=range(self.n_wires))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_wires)]

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Reshape((np.prod(self.input_shape),))(inputs)

        weight_shapes = {"weights": (self.n_layers, self.n_wires)}
        qlayer = qml.qnn.KerasLayer(self.q_circuit, weight_shapes, output_dim=self.n_wires)

        x = qlayer(x)
        x = tf.keras.layers.Dense(self.output_neurons, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["accuracy"])

        return model

    def benchmark(self, model, x_train, y_train, x_test, y_test):
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        accuracy = model.evaluate(x_test, y_test, verbose=0)[1]

        print("Test accuracy:", accuracy)
