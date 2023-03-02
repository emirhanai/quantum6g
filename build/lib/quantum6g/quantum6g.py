import tensorflow as tf
import pennylane as qml

class Quantum6G:
    def __init__(self, output_unit=1, num_layers=4, epochs=2, loss='mse', input=4, batch_size=256, learning_rate=0.2):
        self.output_unit = output_unit
        self.num_layers = num_layers
        self.epochs = epochs
        self.loss = loss
        self.input = input
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def build_model(self, X_train, y_train, X_test, y_test):
        dev = qml.device("default.qubit", wires=self.output_unit)

        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs[:2], wires=range(2))
            qml.BasicEntanglerLayers(weights, wires=range(2))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.output_unit)]

        clayer_1 = tf.keras.layers.Dense(self.output_unit, activation="relu")

        inputs = tf.keras.Input(shape=(self.input,))
        x = clayer_1(inputs)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = qml.qnn.KerasLayer(qnode, {"weights": (self.num_layers, 2)}, output_dim=self.output_unit)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x,name="Quantum_6G")
        opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(opt, loss=self.loss, metrics=["accuracy"])

        model_fit = model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                              shuffle=True, validation_data=(X_test, y_test),
                              verbose=2,
                              callbacks=[tf.keras.callbacks.ModelCheckpoint("/model/model_{epoch}.h5")])

        model.evaluate(X_test, y_test)
        return model

