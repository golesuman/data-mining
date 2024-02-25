import numpy as np
import pandas as pd


class Node:
    def __init__(self, attribute=None, value=None, results=None, children=None):
        self.attribute = attribute  # Index of the attribute to split on
        self.value = value  # Value of the attribute
        self.results = results  # Results for this node (if leaf node)
        self.children = children  # Children nodes (dictionary)


def entropy(data):
    # Calculate the entropy of a dataset
    results = data.iloc[:, -1].value_counts()
    entropy = 0
    total = len(data)

    for result in results:
        p = results[result] / total
        entropy -= p * np.log2(p)

    return entropy


def split_data(data, attribute, value):
    # Split dataset based on an attribute and its value
    return data[data[attribute] == value].reset_index(drop=True)


def information_gain(data, attribute):
    # Calculate the information gain of an attribute
    total_entropy = entropy(data)
    values = data[attribute].unique()
    subset_entropy = 0

    for value in values:
        subset = split_data(data, attribute, value)
        weight = len(subset) / len(data)
        subset_entropy += weight * entropy(subset)

    return total_entropy - subset_entropy


def build_tree(data, features):
    # Recursively build the decision tree
    if len(data) == 0:
        return Node()

    max_gain = 0
    best_attribute = None

    for feature in features:
        gain = information_gain(data, feature)
        if gain > max_gain:
            max_gain = gain
            best_attribute = feature

    if max_gain == 0:
        return Node(results=data.iloc[:, -1].value_counts().idxmax())

    remaining_features = [f for f in features if f != best_attribute]
    node = Node(attribute=best_attribute, children={})

    for value in data[best_attribute].unique():
        subset = split_data(data, best_attribute, value)
        node.children[value] = build_tree(subset, remaining_features)

    return node


def print_tree(node, indent=""):
    if node.results is not None:
        print(indent + "Predict:", node.results)
    else:
        print(indent + node.attribute + " ?")
        for value, child_node in node.children.items():
            print(indent + "  " + value + " -->", end=" ")
            print_tree(child_node, indent + "    ")


# Example Usage
if __name__ == "__main__":
    # Sample dataset
    data = pd.DataFrame(
        {
            "Outlook": [
                "Sunny",
                "Sunny",
                "Overcast",
                "Rain",
                "Rain",
                "Rain",
                "Overcast",
                "Sunny",
                "Sunny",
                "Rain",
            ],
            "Temperature": [
                "Hot",
                "Hot",
                "Hot",
                "Mild",
                "Cool",
                "Cool",
                "Cool",
                "Mild",
                "Cool",
                "Mild",
            ],
            "Humidity": [
                "High",
                "High",
                "High",
                "High",
                "Normal",
                "Normal",
                "Normal",
                "High",
                "Normal",
                "Normal",
            ],
            "Wind": [
                "Weak",
                "Strong",
                "Weak",
                "Weak",
                "Weak",
                "Strong",
                "Strong",
                "Weak",
                "Weak",
                "Weak",
            ],
            "PlayTennis": [
                "No",
                "No",
                "Yes",
                "Yes",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
            ],
        }
    )

    features = data.columns[:-1]

    # Build the tree
    decision_tree = build_tree(data, features)

    # Print the tree
    print("Decision Tree:")
    print_tree(decision_tree)
