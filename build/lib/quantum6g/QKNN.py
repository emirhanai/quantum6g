import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.optimize import GradientDescentOptimizer
import numpy as np

class Quantum6G_KNN:
    def __init__(self, n_qubits=3, n_neighbors=2):
        self.n_qubits = n_qubits
        self.n_neighbors = n_neighbors

        # define the quantum device and the quantum circuit
        self.dev = qml.device('default.mixed', wires=self.n_qubits)
        self.circuit = qml.QNode(self.qnode, self.dev)

    def qnode(self, x, weights):
        # initialize the quantum state using amplitude embedding
        AmplitudeEmbedding(x, wires=range(self.n_qubits), pad_with=0.0, normalize=True,do_queue=False)

        # apply trainable weights to the circuit using strong entangling layers
        StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

        # measure the qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def fit(self, X, y):
        # convert the labels to {-1, 1} instead of {0, 1}
        y = 2*y - 1

        # randomly initialize the weights
        self.weights = np.random.uniform(low=-np.pi, high=np.pi, size=(3, self.n_qubits, 3))

        # train the circuit using gradient descent
        opt = GradientDescentOptimizer()
        num_epochs = 200
        for i in range(num_epochs):
            for j in range(len(X)):
                X_pad = np.pad(X[j], (0, self.n_qubits - len(X[j])), mode='constant', constant_values=0)
                loss = qml.math.sum((self.circuit(X_pad, self.weights) - y[j])**2)
                self.weights = opt.step(lambda v: loss, self.weights)

    def predict(self, X,y):
        y_pred = []
        for xi in X:
            # evaluate the quantum circuit on the input feature vector
            preds = self.circuit(xi, self.weights)

            # calculate the majority class label among the k-nearest neighbors
            k_neighbors = np.argsort(preds)[:self.n_neighbors]
            y_neighbors = [y[k] for k in k_neighbors]
            y_pred.append(np.sign(np.sum(y_neighbors)))

        # convert the labels back to {0, 1}
        y_pred = [(i + 1) // 2 for i in y_pred]

        return y_pred