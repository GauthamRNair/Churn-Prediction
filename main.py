import keras
from keras import layers
from keras.layers import StringLookup
from tensorflow import data as tf_data
import pandas as pd
import math
from decision_tree import NeuralDecisionTree
from decision_forest import NeuralDecisionForest


# All the features in the dataset
CSV_HEADER = ["State",
              "Account length",
              "Area code",
              "International plan",
              "Voice mail plan",
              "Number vmail messages",
              "Total day minutes",
              "Total day calls",
              "Total day charge",
              "Total eve minutes",
              "Total eve calls",
              "Total eve charge",
              "Total night minutes",
              "Total night calls",
              "Total night charge",
              "Total intl minutes",
              "Total intl calls",
              "Total intl charge",
              "Customer service calls",
              "Churn"]

# The numeric features in the dataset
NUMERIC_FEATURE_NAMES = ["Account length",
                         "Number vmail messages",
                         "Total day minutes",
                         "Total day calls",
                         "Total day charge",
                         "Total eve minutes",
                         "Total eve calls",
                         "Total eve charge",
                         "Total night minutes",
                         "Total night calls",
                         "Total night charge",
                         "Total intl minutes",
                         "Total intl calls",
                         "Total intl charge",
                         "Customer service calls"]


# All categorical features with vocabulary
train_data = pd.read_csv("data/churn-bigml-80.csv", header=None, names=CSV_HEADER)
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "State": sorted(list(train_data["State"].unique())),
    "Area code": sorted(list(train_data["Area code"].unique())),
    "International plan": sorted(list(train_data["International plan"].unique())),
    "Voice mail plan": sorted(list(train_data["Voice mail plan"].unique())),
}
# All the categorical feature names
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())

# All the features
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

# Default values for each column in the dataset
COLUMN_DEFAULTS = [
    [0.0] if feature_name in NUMERIC_FEATURE_NAMES else ["NA"]
    for feature_name in CSV_HEADER
]

# The target feature name
TARGET_FEATURE_NAME = "Churn"
TARGET_LABELS = ["False", "True"]
target_label_lookup = StringLookup(
    vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0
)

# Setting up dictionary of categorical features {feature_name: StringLookup()}
lookup_dict = {}
for feature_name in CATEGORICAL_FEATURE_NAMES:
    lookup_dict[feature_name] = StringLookup(
        vocabulary=[str(s) for s in CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]],
        mask_token=None,
        num_oov_indices=0,
    )


# Encoding categorical features into index values
def encode_categorical(batch_x, batch_y):
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        batch_x[feature_name] = lookup_dict[feature_name](batch_x[feature_name])
    return batch_x, batch_y


def get_dataset_from_csv(csv_file_path, shuffle=False, batch_sz=128):
    dataset = (tf_data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_sz,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        shuffle=shuffle)
               .map(lambda features, target: (features, target_label_lookup(target)))
               .map(encode_categorical))

    return dataset.cache()


def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype="float32"
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype="int32"
            )

    return inputs


def encode_inputs(inputs):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            value_index = inputs[feature_name]
            embedding_dims = int(math.sqrt(lookup_dict[feature_name].vocabulary_size()))
            embedding_encoder = layers.Embedding(
                input_dim=len(vocabulary),
                output_dim=embedding_dims
            )
            encoded_feature = embedding_encoder(value_index)
        else:
            encoded_feature = inputs[feature_name]
            if inputs[feature_name].shape[-1] is None:
                encoded_feature = keras.ops.expand_dims(encoded_feature, -1)

        encoded_features.append(encoded_feature)

    encoded_features = layers.concatenate(encoded_features)
    return encoded_features


learning_rate = 0.01
batch_size = 265
num_epochs = 100


def run_experiment(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print("Start training the model...")
    train_dataset = get_dataset_from_csv("data/churn-bigml-80.csv", shuffle=True, batch_sz=batch_size)

    model.fit(train_dataset, epochs=num_epochs)
    print("Model training finished")

    print("Evaluating model performance...")
    test_dataset = get_dataset_from_csv("data/churn-bigml-20.csv", batch_sz=batch_size)

    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")


num_trees = 10
depth = 10
used_features_rate = 1.0
num_classes = len(TARGET_LABELS)


def create_tree_model():
    inputs = create_model_inputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]

    tree = NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
    outputs = tree(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_forest_model():
    inputs = create_model_inputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]

    forest = NeuralDecisionForest(num_trees, depth, num_features, used_features_rate, num_classes)
    outputs = forest(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


forest_model = create_forest_model()
run_experiment(forest_model)
