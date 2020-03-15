import argparse
import os
import numpy as np
import random


def read_file(file_path):
    if not os.path.exists(file_path):
        raise Exception("File not found: " + file_path)

    X = []
    y = []
    is_header_skipped = False
    with open(file_path, "r") as f:
        for line in f:
            if not is_header_skipped:
                is_header_skipped = True
                continue

            cols = line.split(",")
            cols = [col.strip() for col in cols]
            X.append(np.array(cols[:-1], np.float))
            y.append(np.array(cols[-1:], np.float))

    return np.array(X), np.array(y)


class Perceptron:
    def __init__(self):
        pass

    def learn(self, X, y):
        m = X.shape[0]  # size of training sequence
        d = X.shape[1]  # dimension of training instance

        # To account for bias term, increasing dimension of weight vector by 1,
        # and adding extra feature with value 1 in vector X of domain set
        w = np.zeros((1, d + 1), dtype=float)
        X = np.append(X, np.ones((m, 1), dtype=float), axis=1)

        # Treating class 0 as -1 and class 1 as +1
        y = np.where(y == 0, -1.0, y)

        X = np.transpose(X)
        y = np.transpose(y)

        t = 1
        should_terminate = False
        while not should_terminate:
            h_x = np.dot(w, X)
            y_dot_w_x = np.multiply(y, h_x)

            should_terminate = True
            for i in range(m):
                if y_dot_w_x[0][i] <= 0:
                    should_terminate = False
                    w = np.add(w, np.multiply(y[0][i], X[:, i]))
                    break

            if t == 100000:
                should_terminate = True
            t += 1

        print("Iterations took: {0}".format(str(t)))

        return w

    def calculate_risk(self, X, y, w):
        m = X.shape[0]

        # To account for bias term, adding extra feature with value 1 in vector X of domain set
        X = np.append(X, np.ones((m, 1), dtype=float), axis=1)
        # Treating class 0 as -1 and class 1 as +1
        y = np.where(y == 0, -1, y)

        X = np.transpose(X)
        y = np.transpose(y)

        h_x = np.dot(w, X)
        h_x = np.where(h_x > 0, 1, -1)
        mistakes = np.where(h_x != y)
        empirical_risk = float(mistakes[0].shape[0]) / m

        return empirical_risk

    def format_erm_output(self, weight_vector, error):
        separation = "-".join("" for i in range(40))
        return "{0}\nFinal output\nError: {1}\nWeight Vector: {2}\n{3}".format(separation, str(error),
                                                                               str(weight_vector),
                                                                               separation)

    def collect_fold(self, folds, accumulator):
        if len(accumulator) > 0:
            training_seq_list = [inst[0] for inst in accumulator]
            label_seq_list = [inst[1] for inst in accumulator]

            folds.append({"X": training_seq_list, "y": label_seq_list})


    def generate_folds(self, X, y, number_of_folds):
        d = X.shape[1]
        X = X.tolist()
        y = y.tolist()
        instance_sequence = [(X[index], y[index]) for index in range(len(X))]
        random.shuffle(instance_sequence)

        folds = []
        accumulator = []
        batch_size = len(X) // number_of_folds
        if len(X) % number_of_folds > 0:
            batch_size += 1

        for i in range(len(instance_sequence)):
            accumulator.append((X[i], y[i]))
            i += 1

            if len(accumulator) == batch_size:
                self.collect_fold(folds, accumulator)
                accumulator = []

        # Adding residual if any left
        self.collect_fold(folds, accumulator)

        return folds


    def calculate_kfold_errors(self, output_set):
        individual_errors = []
        mean_error = 0.0

        for output in output_set:
            weight_vector = output["w"]
            test_X = output["test_X"]
            test_y = output["test_y"]

            error = self.calculate_risk(test_X, test_y, weight_vector)
            individual_errors.append(error)
            mean_error += error

        mean_error = mean_error / len(individual_errors)

        return mean_error, individual_errors

    def learn_in_kfolds(self, X, y, number_of_folds):
        feature_len = X.shape[1]
        folds = self.generate_folds(X, y, number_of_folds)

        output_set = []

        for test_fold_no in range(len(folds)):
            print("Processing: {0}".format(str(test_fold_no)))

            X = []
            y = []
            test_X = []
            test_y = []
            for i in range(len(folds)):
                if i == test_fold_no:
                    test_X = folds[i]['X']
                    test_y = folds[i]['y']
                else:
                    X = X + folds[i]['X']
                    y = y + folds[i]['y']

            train_len = len(X)
            test_len = len(test_X)
            X = np.array(X).reshape(train_len, feature_len)
            y = np.array(y).reshape(train_len, 1)
            test_X = np.array(test_X).reshape(test_len, feature_len)
            test_y = np.array(test_y).reshape(test_len, 1)

            perceptron = Perceptron()
            weight_vector = self.learn(X, y)
            output_set.append({"w": weight_vector, "test_X": test_X, "test_y": test_y})

        return output_set

    def format_cross_val_output(self, individual_errors, mean_error):
        return "Final output: Mean Error: {0},\nIndividual Errors: {1}".format(
            str(mean_error),
            str(individual_errors)
        )



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='perceptron')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--mode', type=str, help='Mode of learner, pass erm or crossvalidation')

    args = parser.parse_args()

    if not args.dataset:
        parser.error('please specify --dataset with corresponding path to dataset')

    if not args.mode:
        parser.error('please specify --mode with corresponding type of learning')

    X, y = read_file(args.dataset)
    print("Read Training sequence and label set: {0} {1}".format(str(X.shape), str(y.shape)))
    learner = Perceptron()
    if args.mode == "erm":
        w = learner.learn(X, y)
        error = learner.calculate_risk(X, y, w)
        print(learner.format_erm_output(w, error))
    else:
        output_set = learner.learn_in_kfolds(X, y, 10)
        mean_error, individual_erros = learner.calculate_kfold_errors(output_set)
        print(learner.format_cross_val_output(individual_erros, mean_error))
