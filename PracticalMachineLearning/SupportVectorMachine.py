import matplotlib.pyplot as plot
from matplotlib import style
import numpy as np

style.use('ggplot')


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plot.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # training
    def fit(self, data):
        self.data = data
        # dict of mag(w) to [w, b]
        opt_dict = {}
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        # used to get the max and min of the data
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # step sizes to approach the decision boundary
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # small steps: more expensive to computer
                      self.max_feature_value * 0.001,
                      ]

        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   (self.max_feature_value * b_range_multiple),
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print('optimized a step')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            # ||w|| : [w, b]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi*(np.dot(self.w, xi)+self.b))

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):
            print(x, w, b, v)
            return (-w[0] * x - b + v) / w[1]

        data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyperplane_x_min = data_range[0]
        hyperplane_x_max = data_range[1]

        # w.x+b = 1
        # positive support vector hyperplane
        positive_support_vector1 = hyperplane(hyperplane_x_min, self.w, self.b, 1)
        positive_support_vector2 = hyperplane(hyperplane_x_max, self.w, self.b, 1)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max],
                     [positive_support_vector1, positive_support_vector2], 'k')

        # w.x+b = -1
        # negative support vector hyperplane
        negative_support_vector1 = hyperplane(hyperplane_x_min, self.w, self.b, -1)
        negative_support_vector2 = hyperplane(hyperplane_x_max, self.w, self.b, -1)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max],
                     [negative_support_vector1, negative_support_vector2], 'k')

        # w.x+b = 0
        # decision boundary hyperplane
        decision_boudary1 = hyperplane(hyperplane_x_min, self.w, self.b, 0)
        decision_boudary2 = hyperplane(hyperplane_x_max, self.w, self.b, 0)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max],
                     [decision_boudary1, decision_boudary2], '--y')

        plot.show()


data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3], ])}

svm = SupportVectorMachine()
svm.fit(data=data_dict)
svm.visualize()
