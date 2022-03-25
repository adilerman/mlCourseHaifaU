import pandas as pd

TRAIN = 8040
VALIDATION = 10050
TEST = 12563


def load_data(path):
    return pd.read_csv(path)


def gini_impurity(a, b):
    return 1 - (a / (a + b)) ** 2 - (b / (a + b)) ** 2


class Node:
    def __init__(self, data, target_column, depth, min_samples_leaf, min_samples_split, max_depth):
        self.data = data
        self.target = target_column
        self.target_values = sorted(self.data[self.target].unique())
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.depth = depth
        self.left = None
        self.right = None
        feature, numeric_value, feature_values = self.select_feature(self.data)
        # create split left = true
        left_data = self.data.loc[
            self.data[feature] < numeric_value if numeric_value else self.data[feature] == feature_values[1]]
        right_data = self.data.loc[
            self.data[feature] >= numeric_value if numeric_value else self.data[feature] == feature_values[0]]

        left_result = left_data.loc[left_data[target_column] == self.target_values[1]].__len__() / left_data.__len__()
        self.left_prediction = self.target_values[1] if left_result > 0.5 else self.target_values[1]

        right_result = right_data.loc[
                           right_data[target_column] == self.target_values[1]].__len__() / right_data.__len__()
        self.right_prediction = self.target_values[1] if right_result > 0.5 else self.target_values[1]

        a = 'stop for debug'
        # test if going to create leaf
        if depth < max_depth and min_samples_leaf < left_data.__len__() and left_result < 1 and left_result > 0:  # here add pure leaf split
            self.left = Node(left_data, target_column, depth + 1, min_samples_leaf, min_samples_split, max_depth)
        if depth < max_depth and min_samples_leaf < right_data.__len__() and right_result < 1 and right_result > 0:  # here add pure leaf split
            self.left = Node(right_data, target_column, depth + 1, min_samples_leaf, min_samples_split, max_depth)

        a = 'stop for debug'

    def calculate_gini(self, feature, data):
        feature_values = sorted(data[feature].unique())
        if feature_values.__len__() > 2:
            # for non binary data types
            return self.calculate_numerical_gini(feature, data, feature_values)
        feature_gini = 0
        ser = data.groupby([feature, self.target]).size()
        for feature_value in feature_values:
            num_records = ser.get((feature_value, self.target_values[0]), 0) + ser.get((feature_value, self.target_values[1]), 0)
            feature_gini += (num_records / data.__len__()) * gini_impurity(
                ser.get((feature_value, self.target_values[0]), 0), ser.get((feature_value, self.target_values[1]), 0))
        return feature_gini, None, feature_values

    def select_feature(self, data):
        # selects best feature using gini
        # if the feature is numerical (regression based) -> returns the value for the split
        feature_gini_results_dict = {}
        for feature in data.columns.tolist():
            if feature == self.target:
                continue
            feature_gini_results_dict[feature] = self.calculate_gini(feature, data)
        selected = min(feature_gini_results_dict.items(), key=lambda x: x[1])
        best_feature, regression_value, feature_values = selected[0], selected[1][1], selected[1][2]
        return best_feature, regression_value, feature_values

    def calculate_numerical_gini(self, feature, data, feature_values):
        averages = [(a + b) / 2 for a, b in zip(feature_values, feature_values[1:])]
        gini_dict = {}
        for avg in averages:
            new_col_name = f'avg_{avg}'
            gini_dict[avg] = self.calculate_gini(new_col_name, pd.concat([data[self.target], data[feature] < avg],
                                                                         axis=1).reset_index(drop=True).rename(
                columns={feature: new_col_name}))
        best_value = min(gini_dict.items(), key=lambda x: x[1])
        return best_value[1][0], best_value[0], best_value[1][2]


class Tree:
    def __init__(self, data_path, target_column, max_depth, min_samples_leaf, min_samples_split):
        self.target = target_column
        self.data = load_data(data_path)
        self.target_values = self.data[self.target].unique()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.root = Node(self.data, target_column, 0, min_samples_leaf, min_samples_split, max_depth)
        x = 3


if __name__ == '__main__':
    tree = Tree('tutData.csv', 'loves_movies', 3, 1, 1)
    print(tree.data.head())
    print(tree.data.columns.tolist())
    x = 4

