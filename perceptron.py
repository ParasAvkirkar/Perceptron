import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='perceptron')
	parser.add_argument('--dataset', type=str, help='Path to dataset')
	parser.add_argument('--mode', type=str, help='Mode of learner, pass erm or crossvalidation')

	args = parser.parse_args()

	if not args.dataset:
		parser.error('please specify --dataset with corresponding path to dataset')

	if not args.mode:
		parser.error('please specify --mode with corresponding type of learning')		

	# print("Arguments specified: {0} {1}".format(str(args.dataset), str(args.mode)))