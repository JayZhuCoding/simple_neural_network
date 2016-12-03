import numpy as np
import matplotlib.pyplot as plt


def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, layers, activation='tanh') :
        """
        layers: A list containing the number of units in each layer.
                Should contain at least two values
        activation: The activation function to be used. Can be
                "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        self.num_layers = len(layers) - 1
        self.weights = [ np.random.randn(layers[i - 1]+1, layers[i]+1)/10 for i in range(1, len(layers) - 1) ]
        self.weights.append(np.random.randn(layers[-2]+1, layers[-1])/10)
        self.biases = []
        for i in range(1, len(layers)-1):
            self.biases.append(np.random.uniform(-1,1,layers[i]+1))
        self.biases.append(np.random.uniform(-1,1,layers[-1]))

    def forward(self, x) :
        """
        compute the activation of each layer in the network
        """
        a = [x]
        for i in range(self.num_layers) :
            #print len(np.dot(a[i], self.weights[i]))
            a.append(self.activation(np.dot(a[i], self.weights[i]) + self.biases[i]))
        return a


    def backward(self, y, a) :
        """
        compute the deltas for example i
        """
        deltas = [(y - a[-1]) * self.activation_deriv(a[-1])]
        for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
            deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
        deltas.reverse()

        return deltas

    def fit(self, X, y, learning_rate=0.2, epochs=50):
        X = np.asarray(X)
        temp = np.random.uniform(-1,1,(X.shape[0], X.shape[1]+1))
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.asarray(y)

        for k in range(epochs):
            if k%10==0 : print "***************** ", k, "epochs  ***************"
            I = np.random.permutation(X.shape[0])
            for i in I :
                a = self.forward(X[i])
                deltas = self.backward(y[i], a)
                # update the weights using the activations and deltas:
                for j in range(len(self.weights)):
                    layer = np.atleast_2d(a[j])
                    delta = np.atleast_2d(deltas[j])
                    self.weights[j] += learning_rate * layer.T.dot(delta)
                    self.biases[j] += learning_rate * deltas[j]


    def predict(self, x):
        x = np.asarray(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]) + self.biases[l])
        return a

def test_num_hidden() :

    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelBinarizer
    #get data from the original dataset
    dataset = np.genfromtxt("pendigits.tra", delimiter= ",")
    m, n = dataset.shape
    X = dataset[:,:n-1]
    y = dataset[:,n-1]
    X /= X.max()
    pl = plt.figure()
    xy = pl.add_subplot(111)
    num_hidden = np.zeros(100)
    acc = []
    for j in range(100):
        num_hidden[j] = (j+1)*4
    for i in num_hidden:
        nn = NeuralNetwork([16,i,10],'logistic')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        labels_train = LabelBinarizer().fit_transform(y_train)
        labels_test = LabelBinarizer().fit_transform(y_test)
        nn.fit(X_train,labels_train,epochs=100)
        predictions = []
        for i in range(X_test.shape[0]) :
            o = nn.predict(X_test[i])
            predictions.append(np.argmax(o))
        acc.append(accuracy_score(y_test, predictions))
        #plot out accuracy
        print accuracy_score(y_test, predictions)
    xy.plot(num_hidden,acc,label = "activation = logistic")
    xy.set_xlabel('Number of hidden units')
    xy.set_ylabel('Accuracy')
    xy.set_xlim(num_hidden.min(),num_hidden.max())
    plt.grid('on')
    plt.show()


if __name__=='__main__' :
    test_num_hidden()








    """network = NeuralNetwork([2,2,1], 'logistic')
    network.fit(X,y, epochs=100)
    for i in range(len(X)) :
        print i, y[i], network.predict(X[i])"""
