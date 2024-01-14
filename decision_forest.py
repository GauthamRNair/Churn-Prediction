from abc import ABC

import keras
from keras import ops

from decision_tree import NeuralDecisionTree


class NeuralDecisionForest(keras.Model, ABC):
    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        super().__init__()
        self.ensemble = []
        self.num_classes = num_classes

        for _ in range(num_trees):
            self.ensemble.append(
                NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
            )

    def call(self, inputs):
        batch_sz = ops.shape(inputs)[0]
        outputs = ops.zeros([batch_sz, self.num_classes])

        for tree in self.ensemble:
            outputs += tree(inputs)

        outputs /= len(self.ensemble)
        return outputs