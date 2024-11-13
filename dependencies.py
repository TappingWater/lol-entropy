import pandas as pd
import numpy as np
from graphviz import Digraph


def get_info_gain(target_column_name, data_frame):
    """
    Compute the Information Gain for each input feature relative to the target variable.
    This algorithm assumes it the dataset is a binary classification problem.

    Parameters:
    - target_column_name: Name of the column with our target variables
    - data_frame: This will be the input dataset

    Returns:
    - dict: key: Name of input feature value: Information gain  for input feature
    """
    # We first extract the unique values in our target column
    # Since this algorithm is designed for a binary classification problem it assumes there will be
    # 2 unique values, in addition we also take the occurrence of each target variable
    target_labels, counts = np.unique(data_frame[target_column_name], return_counts=True)
    if data_frame[target_column_name].nunique() != 2:
        raise AssertionError(f"Not a binary class problem. Has the following target values: {target_labels}")
    entropy = compute_binary_entropy(counts[0], counts[1])
    # This is the entropy of the dataset
    print(f"Parent entropy: {entropy}")
    info_gain_map = {}
    # Iterate through each feature
    for input_feature_x in data_frame.columns.tolist():
        if input_feature_x != target_column_name:
            print(f"\n Computing entropy for input feature: {input_feature_x}")
            # Create crosstab between the feature and the target
            crosstab = pd.crosstab(data_frame[input_feature_x], data_frame[target_column_name])
            info_gain = 0
            # Iterate over each unique value and calculate the entropy one by one
            for unique_value, row in crosstab.iterrows():
                class_0_count = row[target_labels[0]]
                class_1_count = row[target_labels[1]]
                print(
                    f"For {unique_value}:\n Num of positive examples: {
                      class_0_count} \n Num of negative examples: {class_1_count}"
                )
                child_entropy = compute_binary_entropy(class_0_count, class_1_count)
                print(f"Entropy for label, {unique_value} : {child_entropy}")
                weighted_entropy = 0
                if child_entropy != 0:
                    weighted_entropy = child_entropy * (class_0_count + class_1_count) / (counts[0] + counts[1])
                info_gain += weighted_entropy
                print(
                    f"Weighted entropy for label, {
                      unique_value} : {weighted_entropy}"
                )
            info_gain = entropy - info_gain
            info_gain_map[input_feature_x] = info_gain
    return info_gain_map


def compute_binary_entropy(num_instances_class_0, num_instances_class_1):
    """
    Compute the entropy given the number of positive examples to negative examples

    Parameters:
    - num_instances_class_0: number of instances in class 0
    - num_instances_class_1: number of instances in class 1

    Return:
    - The entropy for a 2 class problem
    """
    # In the case we have a 0 we know for sure that the probability should be 0
    # This also avoid NaN errors
    if num_instances_class_0 == 0 or num_instances_class_1 == 0:
        return 0
    probability_class_0 = num_instances_class_0 / (num_instances_class_0 + num_instances_class_1)
    probability_class_1 = num_instances_class_1 / (num_instances_class_0 + num_instances_class_1)
    return -(probability_class_0 * np.log2(probability_class_0) + probability_class_1 * np.log2(probability_class_1))


class Node:
    def __init__(
        self,
        label,
        class_a_samples,
        class_b_samples,
        info_gain=None,
        children=None,
        class_a_label="Class A",
        class_b_label="Class B",
    ):
        """
        Initializes a node in the decision tree.

        Parameters:
        - label (str): The label or decision at this node.
        - class_a_samples (int): Number of samples for class A at this node.
        - class_b_samples (int): Number of samples for class B at this node.
        - info_gain (float): Information Gain at this node.
        - children (list of tuples): List of tuples (edge_label, child_node).
        - class_a_label (str): The label for class A.
        - class_b_label (str): The label for class B
        """
        self.label = label
        self.class_a_samples = class_a_samples
        self.class_b_samples = class_b_samples
        self.info_gain = info_gain
        self.children = children if children is not None else []
        self.class_a_label = class_a_label
        self.class_b_label = class_b_label

    def add_child(self, edge_label, child_node):
        """
        Adds a child to the node.

        Parameters:
        - edge_label (str): Label on the edge leading to the child.
        - child_node (Node): The child node to add.
        """
        self.children.append((edge_label, child_node))


def traverse(node, dot, parent_id=None, edge_label=""):
    """
    Traverse the tree node by node and build the graph in Graphviz by recursively visiting each node once.
    """
    node_id = str(id(node))
    # Create the label for the node with dynamic class labels
    node_label = f"{node.label}\n{node.class_a_label}: {
        node.class_a_samples} {node.class_b_label}: {node.class_b_samples}"
    if node.info_gain is not None:
        node_label += f"\nInfo Gain: {node.info_gain:.4f}"
    # Add the node to the graph
    dot.node(node_id, label=node_label, shape="box")
    # If this is not the root node, connect it to its parent
    if parent_id is not None:
        dot.edge(parent_id, node_id, label=edge_label)
    # Recursively add child nodes
    for child_edge_label, child_node in node.children:
        traverse(child_node, dot, parent_id=node_id, edge_label=child_edge_label)


def render_tree(root_node, output_filename="decision_tree"):
    """
    Traverses the tree starting from the root node and renders it using Graphviz.

    Parameters:
    - root_node (Node): The root node of the decision tree.
    - output_filename (str): The filename for the output image.

    Returns:
    - Saves the rendered image as an output file.
    """
    dot = Digraph(comment="Information Gain Decision Tree")
    # Start traversal from the root node
    traverse(root_node, dot)
    # Render the graph to a file
    dot.render(filename=output_filename, format="png", cleanup=True)
    print(f"Decision tree rendered and saved as {output_filename}.png")


def build_decision_tree(data, target_column_name, features, parent_node=None, class_labels=None):
    """
    Recursively builds a decision tree based on information gain.

    Parameters:
    - data: The dataset (pandas DataFrame).
    - target_column_name: The name of the target variable column.
    - features: List of features to consider for splitting.
    - parent_node: The parent node in the tree (used in recursion).
    - class_labels: Optional tuple containing labels for the binary classes (class_a, class_b).

    Returns:
    - Node: The root node of the decision tree.
    """
    # Determine the labels for the classes
    target_values = data[target_column_name]
    unique_classes = target_values.unique()
    # Assign class labels dynamically, if not provided
    if class_labels is None:
        class_a_label, class_b_label = unique_classes[0], unique_classes[1]
        class_labels = (class_a_label, class_b_label)
    else:
        class_a_label, class_b_label = class_labels
    # Count the number of samples in each class
    class_a_samples = (target_values == class_a_label).sum()
    class_b_samples = (target_values == class_b_label).sum()
    # If all samples are in one class, create a leaf node
    # If all samples are in one class, create a leaf node
    if len(unique_classes) == 1:
        if unique_classes[0] == class_a_label:
            return Node(
                label="Leaf Node:",
                class_a_samples=class_a_samples,
                class_b_samples=0,
                class_a_label=class_a_label,
                class_b_label=class_b_label,
            )
        else:
            return Node(
                label="Leaf Node:",
                class_a_samples=0,
                class_b_samples=class_b_samples,
                class_a_label=class_a_label,
                class_b_label=class_b_label,
            )
    if len(features) == 0:
        class_a_samples = (target_values == class_a_label).sum()
        class_b_samples = (target_values == class_b_label).sum()
        return Node(
            label="Leaf Node:",
            class_a_samples=class_a_samples,
            class_b_samples=class_b_samples,
            class_a_label=class_a_label,
            class_b_label=class_b_label,
        )
    # Compute information gain for all features
    info_gain_map = get_info_gain(target_column_name, data[features + [target_column_name]])
    # Select the feature with the highest information gain
    best_feature = max(info_gain_map, key=info_gain_map.get)
    best_info_gain = info_gain_map[best_feature]
    # Create the root node with the best feature
    root = Node(
        label=best_feature,
        class_a_samples=class_a_samples,
        class_b_samples=class_b_samples,
        info_gain=best_info_gain,
        class_a_label=class_a_label,
        class_b_label=class_b_label,
    )
    # Remove the best feature from the list of features
    remaining_features = [feat for feat in features if feat != best_feature]
    # Get unique values of the best feature
    feature_values = data[best_feature].unique()
    # For each value of the best feature, create child nodes
    for value in feature_values:
        subset = data[data[best_feature] == value]
        # Recursively build the subtree for this subset
        child_node = build_decision_tree(
            subset, target_column_name, remaining_features, parent_node=root, class_labels=class_labels
        )
        # Add the child node to the root node
        root.add_child(edge_label=str(value), child_node=child_node)
    return root


def initialize_parameters(layer_sizes, seed=13):
    """
    Initialize weights and paramters

    Parameters:
    - layer_sizes: Array of ordered layers  with their respective sizes. for example, passing [4, 5] means 2 hidden layers
    with 4 and 5 neurons respectively

    Returns:
    - A python dictionary with the initialized parameters and biases for each layer
    """
    np.random.seed(seed)
    parameters = {}
    num_layers = len(layer_sizes)
    for i in range(1, num_layers):
        parameters[f"W{i}"] = np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i - 1])
        # Initialize biases to zeros
        parameters[f"b{i}"] = np.zeros((1, layer_sizes[i]))
    return parameters


# Define activation functions
def relu(linear_output):
    return np.maximum(0, linear_output)


def relu_derivative(linear_output):
    return np.where(linear_output > 0, 1, 0)


def softmax(linear_output):
    exp_values = np.exp(linear_output - np.max(linear_output, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def forward_propagation(input_data, parameters):
    """
    Propagate input data forward through the network.

    Parameters:
    - input_data: The input features.
    - parameters: Dictionary containing the network parameters.

    Returns:
    - activation: The output of the last layer (predictions).
    - forward_propagation_cache: List of caches containing intermediate computations.
    """
    forward_propagation_cache = []
    activation = input_data
    num_layers = len(parameters) // 2
    for layer_index in range(1, num_layers + 1):
        activation_prev = activation
        weights = parameters[f"W{layer_index}"]
        biases = parameters[f"b{layer_index}"]
        linear_output = np.dot(activation_prev, weights) + biases
        if layer_index == num_layers:
            activation = softmax(linear_output)
        else:
            activation = relu(linear_output)
        forward_propagation_cache.append((activation_prev, weights, biases, linear_output))
    return activation, forward_propagation_cache


def compute_loss(expected_labels, activation_layer):
    """
    Computes the cross-entropy loss.

    Parameters:
        expected_labels: True labels (one-hot encoded).
        activation_layer: Predicted probabilities from the model.

    Returns:
        loss: Cross-entropy loss.
    """
    m = expected_labels.shape[0]
    # Clip activation_layer to prevent log(0)
    activation_layer = np.clip(activation_layer, 1e-10, 1.0)
    loss = -np.sum(expected_labels * np.log(activation_layer)) / m
    return loss


def backward_propagation(activation_layer, expected_labels, forward_propagation_cache):
    """
    Implements backward propagation for the neural network.

    Parameters:
    - activation_layer: Output from forward propagation (predictions).
    - expected_labels: True labels (one-hot encoded).
    - forward_propagation_cache: List of caches from forward propagation.

    Returns:
    - grads: Dictionary with gradients for each parameter.
    """
    grads = {}
    num_layers = len(forward_propagation_cache)
    m = expected_labels.shape[0]
    expected_labels = expected_labels.reshape(activation_layer.shape)
    delta_linear_output = activation_layer - expected_labels
    for layer_index in reversed(range(num_layers)):
        activation_prev, weights, biases, linear_output = forward_propagation_cache[layer_index]
        # Compute gradients for the current layer
        delta_weights = np.dot(activation_prev.T, delta_linear_output) / m
        delta_biases = np.sum(delta_linear_output, axis=0, keepdims=True) / m
        # Store gradients
        grads[f"dW{layer_index + 1}"] = delta_weights
        grads[f"db{layer_index + 1}"] = delta_biases
        if layer_index > 0:
            # Compute delta_linear_output for the previous layer
            delta_activation_prev = np.dot(delta_linear_output, weights.T)
            linear_output_prev = forward_propagation_cache[layer_index - 1][3]
            delta_linear_output = delta_activation_prev * relu_derivative(linear_output_prev)
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent.

    Parameters:
    - parameters: Dictionary containing parameters
    - grads: Dictionary containing gradients
    - learning_rate: Learning rate for gradient descent

    Returns:
    parameters: Updated parameters after backward propagation
    """
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters


def train(training_data, expected_labels, layer_sizes, epochs, learning_rate):
    """
    Trains the neural network.

    Parameters:
    - training_data: Input features for training.
    - expected_labels: True labels (one-hot encoded).
    - layer_sizes: List containing the size of each layer.
    - epochs: Number of epochs to train.
    - learning_rate: Learning rate for gradient descent.

    Returns:
    - parameters: Trained parameters.
    - loss_history: List of loss values during training.
    """
    parameters = initialize_parameters(layer_sizes)
    loss_history = []
    for epoch in range(epochs):
        # Forward propagation
        activation_layer, forward_propagation_cache = forward_propagation(training_data, parameters)
        # Compute loss
        loss = compute_loss(expected_labels, activation_layer)
        loss_history.append(loss)
        # Backward propagation
        grads = backward_propagation(activation_layer, expected_labels, forward_propagation_cache)
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    return parameters, loss_history


def predict(input_data, parameters):
    """
    Predicts the class labels for given input data.

    Parameters:
    - input_data: Input features.
    - parameters: Trained parameters.

    Returns:
    - predictions: Predicted class labels.
    """
    activation_layer, _ = forward_propagation(input_data, parameters)
    predictions = np.argmax(activation_layer, axis=1)
    return predictions


def compute_feature_importance(parameters):
    """
    Computes the importance of each feature based on the weights of the first layer.

    Parameters:
    - parameters: Dictionary containing the network parameters.

    Returns:
    - feature_importance: Array of feature importance scores.
    """
    weights_first_layer = parameters["W1"]
    feature_importance = np.mean(np.abs(weights_first_layer), axis=1)
    return feature_importance


def get_least_important_feature(feature_importance, feature_names):
    """
    Identifies the least important feature.

    Parameters:
        feature_importance: Array of feature importance scores.
        feature_names: List of feature names.

    Returns:
        least_important_feature: Name of the least important feature.
    """
    min_index = np.argmin(feature_importance)
    least_important_feature = feature_names[min_index]
    return least_important_feature


def drop_feature(input_data, feature_names, feature_to_drop):
    """
    Drops the specified feature from the dataset.

    Parameters:
    - input_data: Input data as a NumPy array.
    - feature_names: List of feature names.
    - feature_to_drop: Name of the feature to drop.

    Returns:
    - transformed_data: Dataset with the least important feature column removed.
    - feature_names_new: New feature names.
    """
    # Find the index of the feature to drop
    drop_index = feature_names.index(feature_to_drop)
    # Remove the feature from the data
    transformed_data = np.delete(input_data, drop_index, axis=1)
    # Remove the feature name from the list
    feature_names_new = feature_names[:drop_index] + feature_names[drop_index + 1 :]
    return transformed_data, feature_names_new

def backward_feature_elimination(input_data, expected_labels, feature_names, layer_sizes, epochs, learning_rate, min_features=1):
    """
    Performs backward feature elimination on the dataset.

    Parameters:
        input_data: Input data as a NumPy array.
        expected_labels: True Labels (one-hot encoded).
        feature_names: List of feature names.
        layer_sizes: List containing the size of each layer.
        epochs: Number of epochs to train.
        learning_rate: Learning rate for gradient descent.
        min_features: Minimum number of features to retain.

    Returns:
        selected_features: List of selected features after elimination.
        performance_history: List of tuples (number of features, accuracy).
    """
    performance_history = []
    current_dataset = input_data.copy()
    feature_names_current = feature_names.copy()
    while len(feature_names_current) >= min_features:
        # Update layer sizes to match the current number of features
        current_layer_sizes = layer_sizes.copy()
        current_layer_sizes[0] = current_dataset.shape[1]
        # If we remove a feature we need to retrain it again with the new input layer which will have less neurons
        parameters, _ = train(current_dataset, expected_labels, current_layer_sizes, epochs, learning_rate)
        # Evaluate the model
        predictions = predict(current_dataset, parameters)
        true_labels = np.argmax(expected_labels, axis=1)
        accuracy = np.mean(predictions == true_labels)
        performance_history.append((len(feature_names_current), accuracy))
        # Compute feature importance
        feature_importance = compute_feature_importance(parameters)
        # Identify and drop the least important feature
        least_important_feature = get_least_important_feature(feature_importance, feature_names_current)
        print(f"Dropping feature: {least_important_feature} with accuracy: {accuracy:.4f}")
        # Drop the feature from the dataset
        current_dataset, feature_names_current = drop_feature(current_dataset, feature_names_current, least_important_feature)
    return feature_names_current, performance_history