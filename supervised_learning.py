from numpy import array
from numpy.linalg import inv
from random import randint


class LinearRegression:
    """
    Linear Regression with Gradient Descent to minimize cost function

    Attributes
    ----------
    m: int
        number of training examples
    n: int
        number of features e.g x1, x2, x3
    features: 2d m*n list
        features of training set
    outputs: m-sized list
        outputs of training set
    parameters: (n)-sized list
        parameters of linear regression

    Methods
    -------
    predict(features)
        predicts output for given features based on calculated parameters
    """

    def __init__(self, features: list, outputs: list, learning_rate=0.01, learning_algorithm='bgd'):
        """
        Parameters
        ----------
        features: 2d m*n list
            features of training set
        outputs: m-sized list
            outputs of training set
        learning_rate: float between 0 and 1, optional
            rate of minizing cost function
        learning_algorithm: string
            learning algorithm type (bgd|sgd|ma)
        """
        self.features = features
        for feature in self.features:
            feature.append(1)  # constant feature
        self.outputs = outputs
        self.learning_rate = learning_rate

        self.m = len(features)
        self.n = len(features[0])
        self.parameters = [0]*(self.n)
        assert learning_algorithm in ('bgd', 'sgd', 'ma')
        if learning_algorithm == 'bgd':
            self.__batch_gradient_descent()
        elif learning_algorithm == 'sgd':
            self.__stochastic_gradient_descent()
        elif learning_algorithm == 'ma':
            self.__matrix_approach()
        print("Learned parameters:", self.parameters)

    def __batch_gradient_descent(self):
        """Alters parameters to minimize cost"""
        for _ in range(int(1e3)):
            total_cost = 0
            for j in range(self.n):
                marginal_cost = 0  # derivative of cost function
                for i in range(self.m):
                    marginal_cost += (self.predict(
                        self.features[i]) - self.outputs[i])*self.features[i][j]
                    print(marginal_cost)
                self.parameters[j] -= self.learning_rate * marginal_cost
                total_cost += abs(marginal_cost)
            # break when local minimum is found
            if total_cost < 1e-5:
                break

    def __stochastic_gradient_descent(self):
        """Alters parameters to minimize cost"""
        for _ in range(int(1e3)):
            for i in range(self.m):
                for j in range(self.n):
                    marginal_cost = (self.predict(
                        self.features[i]) - self.outputs[i])*self.features[i][j]
                    self.parameters[j] -= self.learning_rate * marginal_cost
            # TODO: implement termination condition

    def __matrix_approach(self):
        """Matrix approach to simple linear regression"""
        X = array(self.features)
        y = array(self.outputs)
        # linear least squares
        self.parameters = list(inv(X.T.dot(X)).dot(X.T).dot(y))

    def predict(self, features: list) -> float:
        """
        Predicts output based on features and calculated parameters

        Parameters
        ----------
        features: list
            list of features

        Returns
        -------
        float
            predicted output
        """
        prediction = 0
        x = features.copy()
        if len(x) == len(self.parameters)-1:
            x.append(1)
        for j in range(len(x)):
            prediction += self.parameters[j] * x[j]

        return prediction


if __name__ == "__main__":
    # inputs/features x is a m*n array
    # output/target value y is an array with size of n
    x, y = [], []

    def sample_function(x1, x2):
        return x1 + 2*x2 + 3

    for _ in range(10):
        x1 = randint(-9999, 9999)
        x2 = randint(-9999, 9999)
        x.append([x1, x2])
        # y = x1 + 2 * x2 + 3
        y.append(sample_function(x1, x2))

    regression = LinearRegression(
        x, y, learning_rate=0.01, learning_algorithm='bgd')
    x1 = randint(-9999, 9999)
    print("x1:", x1)
    x2 = randint(-9999, 9999)
    print("x2:", x2)
    print("Expected output:", sample_function(x1, x2))
    # expected result is y = 5 + 2 * 6 + 3 = 20
    print("Predicted output:", regression.predict([x1, x2]))
