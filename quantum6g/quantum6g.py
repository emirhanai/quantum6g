import tensorflow as tf
import pennylane as qml


class Quantum6G:
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(2))
        qml.BasicEntanglerLayers(weights, wires=range(2))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(2)]

    def build_model(self, X_train, y_train, X_test, y_test):
        clayer_1 = tf.keras.layers.Dense(1, activation="relu")

        inputs = tf.keras.Input(shape=(1,))
        x = clayer_1(inputs)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = qml.qnn.KerasLayer(self.qnode, {"weights": (4, 2)}, output_dim=1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        opt = tf.keras.optimizers.SGD(learning_rate=0.2)
        model.compile(opt, loss="mse", metrics=["accuracy"])

        model_fit = model.fit(X_train, y_train, epochs=2, batch_size=256,
                              shuffle=True, validation_data=(X_test, y_test),
                              verbose=2,
                              callbacks=[tf.keras.callbacks.ModelCheckpoint("/model/model_{epoch}.h5")])

        model.evaluate(X_test, y_test)
        return model
