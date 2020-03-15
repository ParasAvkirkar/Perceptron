import argparse
import os
import numpy as np
import math
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


class AdaBoost:
    def __init__(self):
        pass

    def predict_by_stump(self, feature, decision_stump):
        return 1.0 if feature < decision_stump else -1.0

    def weak_learner(self, X, y, distribution_vector):
        m = X.shape[0]
        d = X.shape[1]

        F_star = float('inf')
        j_star = -1
        theta_star = -1
        for j in range(d):
            features = X[:, j]
            feature_y_d_list = [{"x": features[i], "y": y[i, 0], "d": distribution_vector[i,0]} for i in range(m)]

            feature_y_d_list = sorted(feature_y_d_list, key=lambda criteria: criteria["x"])
            F = sum([feature_y_d_list[i]["d"] if feature_y_d_list[i]["y"] == 1.0 else 0.0 for i in range(m)])
            if F < F_star:
                F_star = F
                theta_star = feature_y_d_list[0]["x"] - 1
                j_star = j

            # add dummy suffix to reduce if-else case in taking average when encountered last element
            feature_y_d_list = feature_y_d_list + [{"x": feature_y_d_list[m-1]["x"] + 1, "y": -1.0, "d": -1.0}]
            for i in range(m):
                F = F - feature_y_d_list[i]["y"] * feature_y_d_list[i]["d"]
                if F < F_star and feature_y_d_list[i]['x'] != feature_y_d_list[i+1]['x']:
                    F_star = F
                    theta_star = (feature_y_d_list[i]["x"] + feature_y_d_list[i+1]["x"]) * 0.5
                    j_star = j

        return j_star, theta_star


    def learn(self, X, y, T):
        m = X.shape[0]  # size of training sequence
        d = X.shape[1]  # dimension of training instance

        # Treating class 0 as -1 and class 1 as +1
        y = np.where(y == 0, -1.0, y)

        distribution_vector = np.array([1.0/m for i in range(m)]).reshape(m, 1)
        hypothesis_params_by_iterations = []
        for t in range(T):
            print("Processing round: {0}".format(str(t + 1)))
            j_star, theta_star = self.weak_learner(X, y, distribution_vector)
            epsilon = 0.0
            predictions = []
            for i in range(m):
                prediction = self.predict_by_stump(feature=X[i,j_star], decision_stump=theta_star)
                predictions.append(prediction)
                epsilon += distribution_vector[i, 0] * (1.0 if prediction != y[i, 0] else 0.0)

            w_t = 0.5 * math.log((1.0/epsilon) - 1)

            prediction_vector = np.array(predictions).reshape(m, 1)

            temp_distribution_vector = np.multiply(-1.0 * w_t, np.multiply(y, prediction_vector))
            temp_distribution_vector = np.exp(temp_distribution_vector)
            temp_distribution_vector = np.multiply(distribution_vector, temp_distribution_vector)
            distribution_vector = np.multiply(1.0/np.sum(temp_distribution_vector), temp_distribution_vector)

            print("Round stat: {0}, weight: {1}, j_star: {2}, theta_star: {3}".format(str(t), str(w_t), str(j_star), str(theta_star)))
            hypothesis_params_by_iterations.append({"w_t": w_t, "j_star": j_star, "theta_star": theta_star})

        return hypothesis_params_by_iterations

    def calculate_risk(self, X, y, hypothesis_params):
        m = X.shape[0]
        error = 0.0
        y = np.where(y == 0, -1.0, 1.0)
        for i in range(m):
            weighted_sum = 0.0
            for params in hypothesis_params:
                w_t = params['w_t']
                theta_star = params['theta_star']
                j_star = params['j_star']

                feature_value = X[i, j_star]
                weighted_sum += abs(w_t) * (self.predict_by_stump(feature=feature_value, decision_stump=theta_star))

            prediction = 1.0 if weighted_sum > 0 else -1.0
            error += 1.0 if prediction != y[i,0] else 0.0

        return error/m

    def format_erm_output(self, hypothesis_params, error):
        return "Final Output Error: {0}".format(str(error))

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

            perceptron = AdaBoost()
            weight_vector = perceptron.learn(X, y)
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
    learner = AdaBoost()
    if args.mode == "erm":
        hypothesis_params = learner.learn(X, y, 5)
        error = learner.calculate_risk(X, y, hypothesis_params)
        print(learner.format_erm_output(hypothesis_params, error))
    else:
        output_set = learner.learn_in_kfolds(X, y, 10)
        mean_error, individual_errors = learner.calculate_kfold_errors(output_set)
        print(learner.format_cross_val_output(individual_errors, mean_error))
