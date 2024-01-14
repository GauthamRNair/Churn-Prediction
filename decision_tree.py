from abc import ABC

import keras
from keras import layers
from keras import ops
import numpy as np


class NeuralDecisionTree(keras.Model, ABC):
    def __init__(self, depth, num_features, used_features_rate, num_classes):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_classes = num_classes

        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indices = np.random.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = ops.convert_to_tensor(
            one_hot[sampled_feature_indices], dtype="float32"
        )

        self.pi = self.add_weight(
            initializer="random_normal",
            shape=(self.num_leaves, self.num_classes),
            dtype="float32",
            trainable=True,
        )

        self.decision_fn = layers.Dense(
            units=self.num_leaves, activation="sigmoid", name="decision"
        )

    def call(self, features):
        batch_sz = ops.shape(features)[0]

        features = ops.matmul(
            features, ops.transpose(self.used_features_mask)
        )  # [batch_size, num_used_features]

        decisions = ops.expand_dims(
            self.decision_fn(features), axis=2
        )  # [batch_size, num_leaves, 1]

        decisions = layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )  # [batch_size, num_leaves, 2]

        mu = ops.ones([batch_sz, 1, 1])

        begin_idx = 1
        end_idx = 2

        for level in range(self.depth):
            mu = ops.reshape(mu, [batch_sz, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = ops.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[
                              :, begin_idx:end_idx, :
                              ]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = ops.reshape(mu, [batch_sz, self.num_leaves])  # [batch_size, num_leaves]
        probabilities = keras.activations.softmax(self.pi)  # [num_leaves, num_classes]
        outputs = ops.matmul(mu, probabilities)  # [batch_size, num_classes]
        return outputs
