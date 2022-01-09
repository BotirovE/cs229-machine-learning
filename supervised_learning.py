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
    parameters: (n+1)-sized list
        parameters of linear regression

    Methods
    -------
    predict(features)
        predicts output for given features based on calculated parameters
    """

    def __init__(self, features: list, outputs: list, learning_rate=0.01, gd_type='batch'):
        """
        Parameters
        ----------
        features: 2d m*n list
            features of training set
        outputs: m-sized list
            outputs of training set
        learning_rate: float between 0 and 1, optional
            rate of minizing cost function
        gd_type: string
            gradient descent type (batch|stochastic)
        """
        self.features = features
        self.outputs = outputs
        self.learning_rate = learning_rate

        self.m = len(features)
        self.n = len(features[0])
        self.parameters = [0]*(self.n+1)  # last parameter for constant feature
        assert gd_type in ('batch', 'stochastic')
        if gd_type == 'batch':
            self.__batch_gradient_descent
        elif gd_type == 'stochastic':
            self.__stochastic_gradient_descent
        print("Learned parameters:", self.parameters)

    def __batch_gradient_descent(self):
        """Alters parameters to minimize cost"""
        for _ in range(1e7):
            J_derivative_sum = 0
            for j in range(self.n):
                J_derivative = 0  # derivative of cost function
                for i in range(self.m):
                    J_derivative += (self.predict(
                        self.features[i]) - self.outputs[i])*self.features[i][j]
                self.parameters[j] -= self.learning_rate * J_derivative
                J_derivative_sum += abs(J_derivative)
            # parameter of constant feature
            J_derivative = sum([(self.predict(self.features[i]) - self.outputs[i])
                                for i in range(self.m)])
            self.parameters[self.n] -= self.learning_rate * J_derivative
            J_derivative_sum += abs(J_derivative)
            # break when local minimum is found
            if J_derivative_sum < 1e-10:
                break

    def __stochastic_gradient_descent(self):
        """Alters parameters to minimize cost"""
        for _ in range(1e7):
            for i in range(self.m):
                for j in range(self.n):
                    J_derivative = (self.predict(
                        self.features[i]) - self.outputs[i])*self.features[i][j]
                    self.parameters[j] -= self.learning_rate * J_derivative
                # parameter of constant feature
                J_derivative = self.predict(self.features[i]) - self.outputs[i]
                self.parameters[self.n] -= self.learning_rate * J_derivative
            # TODO: implement termination condition

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
        prediction = self.parameters[-1]  # parameter of constant feature
        for j in range(len(features)):
            prediction += self.parameters[j] * features[j]

        return prediction


if __name__ == "__main__":
    # inputs/features x is a m*n array
    x = [[1, 2],
         [2, 3],
         [3, 4],
         [4, 5]]
    # output/target value y is an array with size of n
    # y = x1 + 2 * x2 + 3
    y = [1+2*2+3,
         2+3*2+3,
         3+4*2+3,
         4+5*2+3]
    regression = LinearRegression(x, y)
    # expected result is y = 5 + 2 * 6 + 3 = 20
    print(regression.predict([5, 6]))
