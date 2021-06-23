import numpy as np
import pandas as pd
from random import randrange
import random
from scipy.stats import chi2
from pprint import pprint
# for understanding of the data
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# data = pd.read_csv("surfinexamplefromclass.csv")


# load data
data = pd.read_csv("DefaultOfCreditCardClients.csv")


# arrange data
# understand data - by graphs
# X1 - amount of the given credit
# X1 = data['X1']

# sns.kdeplot(data['X1'], shade=True)
# sns.distplot(X1, hist=True, kde=False)
# plt.show()

# X5 - age
# X5 = data['X5']
#
# sns.distplot(X5, hist=True, kde=False)
# plt.show()

# X6-X11 - history of past payment
# X6
# X6 = data['X6']
#
# sns.distplot(X6, hist=True, kde=False)
# plt.show()
# # X7
# X7 = data['X7']
#
# sns.distplot(X7, hist=True, kde=False)
# plt.show()
# # X8
# X8 = data['X8']
#
# sns.distplot(X8, hist=True, kde=False)
# plt.show()
# # X9
# X9 = data['X9']
#
# sns.distplot(X9, hist=True, kde=False)
# plt.show()
# # X10
# X10 = data['X10']
#
# sns.distplot(X10, hist=True, kde=False)
# plt.show()
# # X11
# X11 = data['X11']
#
# sns.distplot(X11, hist=True, kde=False)
# plt.show()

# X12-X17 - amount of bill statement
# X12
# X12 = data['X12']
#
# sns.distplot(X12, hist=True, kde=False)
# plt.show()
# # X13
# X13 = data['X13']
#
# sns.distplot(X13, hist=True, kde=False)
# plt.show()
# # X14
# X14 = data['X14']
#
# sns.distplot(X14, hist=True, kde=False)
# plt.show()
# # X15
# X15 = data['X15']
#
# sns.distplot(X15, hist=True, kde=False)
# plt.show()
# # X16
# X16 = data['X16']
#
# sns.distplot(X16, hist=True, kde=False)
# plt.show()
# # X17
# X17 = data['X17']
#
# sns.distplot(X17, hist=True, kde=False)
# plt.show()

# X18-X23 - amount of previous payment
# X18
# X18 = data['X18']
#
# sns.distplot(X18, hist=True, kde=False)
# plt.show()
# # X19
# X19 = data['X19']
#
# sns.distplot(X19, hist=True, kde=False)
# plt.show()
# # X20
# X20 = data['X20']
#
# sns.distplot(X20, hist=True, kde=False)
# plt.show()
# # X21
# X21 = data['X21']
#
# sns.distplot(X21, hist=True, kde=False)
# plt.show()
# # X22
# X22 = data['X22']
#
# sns.distplot(X22, hist=True, kde=False)
# plt.show()
# # X23
# X23 = data['X23']
#
# sns.distplot(X23, hist=True, kde=False)
# plt.show()

# mean, max, min
# print(data.astype(float).mean(axis=0))
# print(data.min())
# print(data.max())
# print(data)

def arrange_data(arr_as_df):  # divide to buckets by the graphs I analyzed
    arr_as_df = arr_as_df.astype(int)  # change string type to int for all data
    good_data = arr_as_df
    labels_X1 = [0, 1, 2]
    bins_X1 = [-1, 150000, 400000, 2000000]
    good_data['X1'] = pd.cut(good_data['X1'], bins=bins_X1, labels=labels_X1)

    # X5
    labels_X5 = [0, 1, 2, 3, 4]
    bins_X5 = [0, 30, 40, 50, 60, 120]
    good_data['X5'] = pd.cut(good_data['X5'], bins=bins_X5, labels=labels_X5)

    # X6-11
    labels_X6_11 = [-2, -1, 0, 1, 2, 3]
    bins_X6_11 = [-3, -2, -1, 0, 1, 2, 10]
    good_data['X6'] = pd.cut(good_data['X6'], bins=bins_X6_11, labels=labels_X6_11)
    good_data['X7'] = pd.cut(good_data['X7'], bins=bins_X6_11, labels=labels_X6_11)
    good_data['X8'] = pd.cut(good_data['X8'], bins=bins_X6_11, labels=labels_X6_11)
    good_data['X9'] = pd.cut(good_data['X9'], bins=bins_X6_11, labels=labels_X6_11)
    good_data['X10'] = pd.cut(good_data['X10'], bins=bins_X6_11, labels=labels_X6_11)
    good_data['X11'] = pd.cut(good_data['X11'], bins=bins_X6_11, labels=labels_X6_11)

    # X12-17
    labels_X12_17 = [0, 1, 2, 3, 4]
    bins_X12_17 = [-2000000, 0, 50000, 100000, 200000, 2000000]
    good_data['X12'] = pd.cut(good_data['X12'], bins=bins_X12_17, labels=labels_X12_17)
    good_data['X13'] = pd.cut(good_data['X13'], bins=bins_X12_17, labels=labels_X12_17)
    good_data['X14'] = pd.cut(good_data['X14'], bins=bins_X12_17, labels=labels_X12_17)
    good_data['X15'] = pd.cut(good_data['X15'], bins=bins_X12_17, labels=labels_X12_17)
    good_data['X16'] = pd.cut(good_data['X16'], bins=bins_X12_17, labels=labels_X12_17)
    good_data['X17'] = pd.cut(good_data['X17'], bins=bins_X12_17, labels=labels_X12_17)

    # X18-23
    labels_X18_23 = [0, 1, 2, 3]
    bins_X18_23 = [-1, 10000, 40000, 100000, 2000000]
    good_data['X18'] = pd.cut(good_data['X18'], bins=bins_X18_23, labels=labels_X18_23)
    good_data['X19'] = pd.cut(good_data['X19'], bins=bins_X18_23, labels=labels_X18_23)
    good_data['X20'] = pd.cut(good_data['X20'], bins=bins_X18_23, labels=labels_X18_23)
    good_data['X21'] = pd.cut(good_data['X21'], bins=bins_X18_23, labels=labels_X18_23)
    good_data['X22'] = pd.cut(good_data['X22'], bins=bins_X18_23, labels=labels_X18_23)
    good_data['X23'] = pd.cut(good_data['X23'], bins=bins_X18_23, labels=labels_X18_23)
    return good_data


del data['Unnamed: 0']  # remove ID col
data = data.iloc[1:]  # remove the names of variables
data = arrange_data(data)  # set correct Data


def train_test_split(df, train_size):  # gets Data frame and % of train size and divide the df to train&test
    test_size = round(train_size * len(df))
    indices = df.index.tolist()
    train_indices = random.sample(population=indices, k=test_size)
    train_data_frame = df.loc[train_indices]
    test_data_frame = df.drop(train_indices)
    return train_data_frame, test_data_frame


def is_pure(arr):  # check if all target values are similar
    target_col = arr[:, -1]
    unique_targets = np.unique(target_col)
    if len(unique_targets) == 1:
        return True
    else:
        return False


def classify(arr):  # check data classification by majority of targets
    target_col = arr[:, -1]
    unique_targets, counts_unique_targets = np.unique(target_col, return_counts=True)
    index = counts_unique_targets.argmax()  # the index of the target that appears the most
    classification = unique_targets[index]
    return classification


def optional_splits(arr):  # gives the options to split each feature
    options = {}
    n_rows, n_cols = arr.shape
    for col_index in range(n_cols - 1):  # excluding the last column which is the target
        options[col_index] = []
        values = arr[:, col_index]
        unique_values = np.unique(values)
        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2  # the middle between the actual values
                options[col_index].append(potential_split)
    return options


def split_data(arr, split_column):  # split data of specific column by value
    split_column_values = arr[:, split_column]
    data_splits = []
    unique_values = np.unique(split_column_values)
    for i in unique_values:
        data_splits.append(arr[split_column_values == i])
    return data_splits


def entropy(arr):  # calculate entropy
    target_col = arr[:, -1]
    targets, target_count = np.unique(target_col, return_counts=True)
    entropy_value = - np.sum([(target_count[i] / np.sum(target_count)) *
                              np.log2(target_count[i] / np.sum(target_count))
                              for i in range(len(targets))])
    return entropy_value


def gain(arr, attr):  # calculate gain from current entropy
    current_entropy = entropy(arr)
    values, feature_count = np.unique(arr[:, attr], return_counts=True)
    weighted_entropy = np.sum([(feature_count[i] / np.sum(feature_count)) *
                               (entropy(arr[arr[:, attr] == values[i]]))
                               for i in range(len(values))])
    feature_gain = current_entropy - weighted_entropy
    return feature_gain


def importance(arr):  # choose the attribute with the best gain
    best_gain = 0
    chosen_attribute = ''
    for a in range(len(arr[0]) - 1):
        a_gain = gain(arr, a)
        if a_gain > best_gain:
            best_gain = a_gain
            chosen_attribute = a
    if best_gain == 0 and chosen_attribute == '':  # if all gains are 0 - choose attribute randomly
        # a combo with the new attribute may help
        chosen_attribute = randrange(len(arr[0]))
    return chosen_attribute


def decision_tree_algorithm(arr, cols, counter=0):  # makes a tree
    num_rows, num_cols = arr.shape
    data_cols_in_array = num_cols - 1
    relevant_array = arr[:, 0: data_cols_in_array]
    if (is_pure(arr)) or (num_rows == 1) or (num_cols == 2) or (relevant_array == relevant_array[0]).all():
        # if all targets are the same / only 1 example / only 1 attribute / all examples are identical beside Y
        classification = classify(arr)
        return classification  # base case
    else:
        counter += 1
        split_column = importance(arr)
        feature_name = cols[split_column]
        while feature_name == 'Y':  # make sure the chosen attribute isn't the target
            split_column = importance(arr)
            feature_name = cols[split_column]
        feature_values = np.unique(arr[:, split_column])
        root = "{} : {}".format(feature_name, feature_values)
        root = root.replace('\\', "")
        sub_tree = {root: []}
        data_values = split_data(arr, split_column)
        index = 0
        new_col = cols.drop(feature_name)
        for v in feature_values:  # for each value build a subtree
            new_data = np.delete(data_values[index], split_column, axis=1)
            new_tree = decision_tree_algorithm(new_data, new_col, counter)
            sub_tree[root].append(new_tree)
            index += 1
        return sub_tree


global COLUMN_HEADERS
COLUMN_HEADERS = data.columns


def classify_example(example, dt):  # make a classification for 1 example over a decision tree
    root = list(dt.keys())[0]
    feature_name, dots, values_s = root.split(' ', 2)
    values = values_s.strip('][').split(' ')
    correct_values = []
    for i in range(0, len(values)):
        if values[i] == '':
            continue
        correct_values.append(int(values[i]))
    answer = ''
    index = 0
    for v in correct_values:
        if example[feature_name] == v:
            answer = dt[root][index]
            break
        index += 1
    if answer == '':  # if the tree didn't have the exact value as a bucket
        answer = classify(data[[feature_name, 'Y']].values)  # classify by the major Y of this feature
    if not isinstance(answer, dict):  # got to classification
        return answer
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)  # go deeper in the tree


def calculate_accuracy(df, dt):  # accuracy of test df over decision tree
    df["classification"] = df.apply(classify_example, axis=1, args=(dt,))  # add col of classification to df
    df["classification_correct"] = df["classification"] == df["Y"]  # check which ones are correct
    accuracy = df["classification_correct"].mean()
    return accuracy


def cross_validation_split(df, folds):  # divide data to K randomly
    data_split = []
    df_copy = df
    fold_size = int(df_copy.shape[0] / folds)
    for i in range(folds):  # create each fold
        fold = []
        while len(fold) < fold_size:  # add data to the fold
            r = randrange(df_copy.shape[0])  # takes data randomly
            index = df_copy.index[r]
            fold.append(df_copy.loc[index].values.tolist())  # add tha data to the fold
            df_copy = df_copy.drop(index)  # remove it from the arr
        data_split.append(np.asarray(fold))
    return data_split  # list of all folds - each fold as array


# ------------------------------ tree_error ------------------------------
def tree_error(k):  # return the error rate of a tree, using k-folds cross validation
    if k <= 1:
        print('k<=1 -> the train data is all data: there is no data to test error rate')
    else:
        arr = cross_validation_split(data, k)
        result = []
        for i in range(k):  # decide on train and test data
            r = list(range(k))
            r.pop(i)
            for j in r:
                if j == r[0]:
                    cv = arr[j]
                else:
                    cv = np.concatenate((cv, arr[j]), axis=0)
            decision_tree = decision_tree_algorithm(cv, COLUMN_HEADERS)
            cols = COLUMN_HEADERS.tolist()
            cv_train_df = pd.DataFrame(data=cv, columns=cols)
            pruned_tree = post_pruning(decision_tree, cv_train_df)
            # calculate accuracy
            df_test = pd.DataFrame(data=arr[i], columns=COLUMN_HEADERS)
            accuracy = calculate_accuracy(df_test, pruned_tree)
            result.append(accuracy)
        average_result = sum(result) / len(result)
        error_rate = 1 - average_result
        print(error_rate)


# ------------------------------ build tree ------------------------------
def build_tree(k):  # build tree with k% of the data as train and print the tree and error rate
    train_df, test_df = train_test_split(data, train_size=k)
    tree = decision_tree_algorithm(train_df.values, COLUMN_HEADERS)
    pruned_tree = post_pruning(tree, train_df)
    if k == 1:
        pprint(pruned_tree, width=400)  # prints the tree nicely (:
        print('k=1 -> the train data is all data: there is no data to test error rate')
    else:
        accuracy = calculate_accuracy(test_df, pruned_tree)
        pprint(pruned_tree, width=400)  # prints the tree nicely (:
        error_rate = 1 - accuracy
        print('error rate = ', error_rate)


# ------------------------------ will default ------------------------------
def will_default(arr):  # gives a prediction if will default or not
    tree = decision_tree_algorithm(data.values, COLUMN_HEADERS)  # tree with all data
    pruned_tree = post_pruning(tree, data)
    cols = COLUMN_HEADERS.delete(23)
    cols = cols.tolist()
    if isinstance(arr, list):
        test = pd.DataFrame(data=[arr])
        test.columns = cols
    else:
        test = pd.DataFrame(data=arr, columns=cols)  # the arr given
    arranged_test = arrange_data(test)
    arranged_test["classification"] = arranged_test.apply(classify_example, axis=1, args=(pruned_tree,))
    print(arranged_test["classification"])


def post_pruning(decision_tree, train):  # prune tree by chi square test
    root = list(decision_tree.keys())[0]
    kids = decision_tree[root]
    are_leafs = []
    for k in kids:  # check if all kids are leafs
        if not isinstance(k, dict):
            are_leafs.append(True)
        else:
            are_leafs.append(False)
    if all(flag == True for flag in are_leafs):  # if we got to leaf
        return do_the_chi_square(decision_tree, train)  # base case
    else:
        index = 0
        for subtree in kids:
            if isinstance(subtree, dict):
                kids[index] = post_pruning(subtree, train)
            index += 1
        dt = {root: kids}
        are_leafs2 = []
        for k in kids:  # check if all kids are leafs
            if not isinstance(k, dict):
                are_leafs2.append(True)
            else:
                are_leafs2.append(False)
        if all(flag == True for flag in are_leafs2):
            return do_the_chi_square(decision_tree, train)
        else:
            return dt


def do_the_chi_square(decision_tree, train):  # chi square test
    root = list(decision_tree.keys())[0]
    feature_name, dots, values_s = root.split(' ', 2)
    values = values_s.strip('][').split(' ')
    correct_values = []
    for i in range(0, len(values)):
        if values[i] == '':
            continue
        correct_values.append(int(values[i]))
    delta = 0
    p = int(train[train['Y'] == 1].count()[0])
    n = int(train[train['Y'] == 0].count()[0])
    df = len(correct_values) - 1  # (num of buckets of feature - 1)*(number of targets -1) = ()*1
    for v in correct_values:
        pk = int(train[(train['Y'] == 1) & (train[feature_name] == v)].count()[0])
        nk = int(train[(train['Y'] == 0) & (train[feature_name] == v)].count()[0])
        estimate_pk = float(p * ((pk + nk) / (p + n)))
        estimate_nk = float(n * ((pk + nk) / (p + n)))
        if estimate_pk == 0 or estimate_nk == 0:
            delta += 1000000
        else:
            delta = delta + ((pk - estimate_pk) ** 2) / estimate_pk + ((nk - estimate_nk) ** 2) / estimate_nk
    chi_square = chi2.ppf(0.95, df)  # ---------------------------------------
    if delta < chi_square:  # this leaf is just a noise
        leaf = data['Y'].value_counts().index[0]
        return leaf
    else:
        return decision_tree


# ------------------------ activate functions ----------------------------------------
# build_tree(0.6)
# tree_error(3)
# check_arr = np.array([['20000', '2', '2', '1', '32', '2', '2', '-1', '-1', '-2', '-2', '3913', '3102', '13559', '0',
#                        '14948', '0', '0', '689', '0', '0', '0', '0'],
#                       ['120000', '2', '2', '2', '14', '-1', '2', '0', '1', '0', '2', '2682', '13559', '2682', '3272',
#                        '3455', '3261', '0', '1000', '1000', '17000', '1000', '2000'],
#                       ['90000', '2', '2', '2', '80', '1', '2', '-1', '0', '1', '0', '29239', '14027', '13559', '14331',
#                        '689', '15549', '1518', '1500', '1000', '1000', '1000', '50']
#                       ])
# check_list = ['90000', '2', '2', '2', '80', '1', '2', '-1', '0', '1', '0', '29239', '14027', '13559', '14331', '689',
#               '15549', '1518', '1500', '1000', '1000', '1000', '50']
# check_list_int = [90000, 2, 2, 2, 80, 1, 2, -1, 0, 1, 0, 29239, 14027, 13559, 14331, 689,
#               15549, 1518, 1500, 1000, 1000, 1000, 50]
# check_arr_int = np.array([[20000, 2, 2, 1, 32, 2, 2, -1, -1, -2, -2, 3913, 3102, 13559, 0,
#                        14948, 0, 0, 689, 0, 0, 0, 0],
#                       [120000, 2, 2, 2, 14, -1, 2, 0, 1, 0, 2, 2682, 13559, 2682, 3272,
#                        3455, 3261, 0, 1000, 1000, 17000, 1000, 2000],
#                       [90000, 2, 2, 2, 80, 1, 2, -1, 0, 1, 0, 29239, 14027, 13559, 14331,
#                        689, 15549, 1518, 1500, 1000, 1000, 1000, 50]
#                       ])
# will_default(check_list_int)

