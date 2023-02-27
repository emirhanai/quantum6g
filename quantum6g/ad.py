import numpy as np
from quantum6g import Quantum6G

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
print("X_train shape: ",X_train.shape)
quantum_6g = Quantum6G(num_wires=4,num_layers=8,batch_size=128,learning_rate=0.002)
quantum_6g = quantum_6g.build_model(X_train, y_train, X_test, y_test,epochs=5)
print("Accuracy: {:.2f}%".format(quantum_6g[1][1] * 100))
print("Loss: {:.2f}%".format(quantum_6g[1][0] * 100))
