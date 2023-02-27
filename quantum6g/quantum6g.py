import tensorflow as tf
import numpy as np


class Quantum6G:
    def __init__(self, num_wires=2, num_layers=4, batch_size=256, learning_rate=0.2):
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_shapes = {"weights": (self.num_layers, self.num_wires, 3)}

    def angle_embedding(self, features):
        angle = 2 * np.arcsin(np.sqrt(features))
        return np.outer(angle, [1, 1])

    def rx(self, theta, wire):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -1j*s], [-1j*s, c]])

    def rz(self, phi, wire):
        c = np.cos(phi / 2)
        s = np.sin(phi / 2)
        return np.array([[np.exp(-1j*phi/2), 0], [0, np.exp(1j*phi/2)]])

    def basic_entangler_layer(self, theta, wires):
        num_wires = len(wires)
        for i in range(num_wires - 1):
            j = i + 1
            wire_i, wire_j = wires[i], wires[j]
            gate_sequence = [
                (self.rx(np.pi/2, wire_j), wire_j),
                (self.rx(theta[i, j, 0], wire_j) @ self.rz(theta[i, j, 1], wire_i), [wire_i, wire_j]),
                (self.rx(np.pi/2, wire_i), wire_i),
                (self.rx(theta[i, j, 2], wire_i) @ self.rz(theta[i, j, 1], wire_j), [wire_i, wire_j]),
                (self.rx(np.pi/2, wire_j), wire_j),
            ]
            for gate in gate_sequence:
                yield gate

    def variational_circuit(self, features, weights):
        wires = range(self.num_wires)
        theta = np.zeros(self.weight_shapes["weights"])
        for l in range(self.num_layers):
            theta[l] = weights[l]
            gates = list(self.basic_entangler_layer(theta[l], wires))
            state = self.angle_embedding(features)
            for gate in gates:
                state = gate[0] @ state if isinstance(gate[1], int) else np.kron(gate[0], np.eye(2)) @ state.reshape(
                    [2] * self.num_wires).transpose([i for i in gate[1]]).reshape([2 ** len(gate[1]), -1])
            prob = np.real(np.diag(state @ state.conj().T))
        return prob

    def build_model(self, X_train, y_train, X_test, y_test, epochs=2):
        clayer_1 = tf.keras.layers.Dense(1, activation="relu")

        inputs = tf.keras.Input(shape=(1,))
        x = clayer_1(inputs)
        x = tf.keras.layers.Dropout(0.1)(x)

        qlayer = tf.keras.layers.Dense(self.num_wires, use_bias=False)

        x = qlayer(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(opt, loss="mse", metrics=["accuracy"])

        model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size,
                  shuffle=True, validation_data=(X_test, y_test),
                  verbose=2,
                  callbacks=[tf.keras.callbacks.ModelCheckpoint("/model/model_{epoch}.h5")])

        accuracy = model.evaluate(X_test, y_test)
        return model, accuracy
