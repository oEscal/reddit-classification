import argparse
import pickle
import time

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from reddit_classification.graphics import plot_accuracy_over_var
from reddit_classification.data_treatment import read_data
from reddit_classification.utils import convert_input


MIN_ALPHA = 1e-10
MAX_ALPHA = 100


def get_data(number_classes):
	if number_classes:
		X, y = read_data(f"data/pruned_{number_classes}_entries.tsv")
	else:
		X, y = read_data()

	# 60-20-20 division
	train_split_index = int(len(X) * 0.6)
	cv_split_index = train_split_index + int(len(X) * 0.2)

	X_train, X_cv, X_test = X[:train_split_index], X[train_split_index:cv_split_index], X[cv_split_index:]
	y_train, y_cv, y_test = y[:train_split_index], y[train_split_index:cv_split_index], y[cv_split_index:]

	label_encoder = LabelEncoder()
	label_encoder.fit(y_train)
	y_train = label_encoder.transform(y_train)
	y_cv = label_encoder.transform(y_cv)
	y_test = label_encoder.transform(y_test)

	return X_train, X_cv, X_test, y_train, y_cv, y_test


def train(number_classes=50, retrain=0):
	"""
	from https://www.kaggle.com/mswarbrickjones/starter-naive-bayes-benchmark-p-at-5-0-88
	"""
	X_train, X_cv, X_test, y_train, y_cv, y_test = get_data(number_classes)

	if retrain:
		X_train = np.concatenate((X_train, X_cv), axis=0)
		X_cv = X_test
		y_train = np.concatenate((y_train, y_cv), axis=0)
		y_cv = y_test

	# extract features
	X_train, X_cv, token_frequencies = convert_input(X_train, X_cv, limit=1000, min_df=1)

	# save token frequencies
	with open(f"reddit_bayes/token_frequencies/token_frequencies_{number_classes}", 'wb') as file:
		pickle.dump(token_frequencies, file)

	# train
	initial_alpha = MIN_ALPHA if not retrain else retrain
	while initial_alpha <= (MAX_ALPHA if not retrain else retrain):
		for i in [1, 5]:
			alpha = initial_alpha * i

			initial_time = time.time()
			model = MultinomialNB(alpha=alpha)
			model.fit(X_train, y_train)

			# save model
			with open(f"reddit_bayes/models/model_{number_classes}_{format(alpha, '.10g')}{'_retrain' if retrain else ''}",
			          'wb') as file:
				pickle.dump(model, file)

			# with open(f"reddit_bayes/models/model_{number_classes}_{alpha}", 'rb') as file:
			# 	model = pickle.load(file)

			y_pred_proba = model.predict_proba(X_cv)
			y_pred = np.argmax(y_pred_proba, axis=1)

			print(f"precision for alpha {format(alpha, '.10g')} = {np.mean(y_cv == y_pred)}")

			print(time.time() - initial_time)

			if retrain:
				break
		initial_alpha *= 10


def interpretation(number_classes=50, retrain=0):
	X_train, X_cv, X_test, y_train, y_cv, y_test = get_data(number_classes)

	# load token frequencies
	with open(f"reddit_bayes/token_frequencies/token_frequencies_{number_classes}", 'rb') as file:
		token_frequencies = pickle.load(file)
		X_train = token_frequencies.fit_transform(X_train).todense()
		X_cv = token_frequencies.transform(X_cv).todense()
		X_test = token_frequencies.transform(X_test).todense()

	# normalize
	train_max = np.max(X_train)
	X_train = X_train / train_max
	X_cv = X_cv / train_max
	X_test = X_test / train_max

	if retrain:
		with open(f"reddit_bayes/models/model_{number_classes}_{format(retrain, '.10g')}_retrain", 'rb') as file:
			model: MultinomialNB = pickle.load(file)

		y_pred = model.predict(X_test)
		report = classification_report(y_test, y_pred, output_dict=True)

		print(r"\begin{tabular}{l c c c c}")
		print(r"Class & Accuracy & Recall & Precision & F1 Score\\ \hline")
		for key in report:
			if key.isdigit():
				accuracy = accuracy_score(y_test[y_test == int(key)], y_pred[y_test == int(key)])
				print(f"{key} & {accuracy:.3} & {report[key]['recall']:.3} & "
				      f"{report[key]['precision']:.3} & {report[key]['f1-score']:.3}\\\\")
		print(r"\hline" + f"\nMacro Average & {report['accuracy']:.3} & {report['macro avg']['recall']:.3} & "
		                  f"{report['macro avg']['precision']:.3} & {report['macro avg']['f1-score']:.3}\\\\")
		print(r"\end{tabular}")
	else:
		accuracy_over_alpha_train = {}
		accuracy_over_alpha_cv = {}
		initial_alpha = MIN_ALPHA
		while initial_alpha <= MAX_ALPHA:
			for i in [1, 5]:
				alpha = initial_alpha * i

				with open(f"reddit_bayes/models/model_{number_classes}_{format(alpha, '.10g')}", 'rb') as file:
					model: MultinomialNB = pickle.load(file)

				accuracy_over_alpha_train[alpha] = model.score(X_train, y_train)
				accuracy_over_alpha_cv[alpha] = model.score(X_cv, y_cv)

			initial_alpha *= 10

		plot_accuracy_over_var(list(accuracy_over_alpha_cv.keys()), list(accuracy_over_alpha_cv.values()),
		                       "alpha", "graphics", hold_on=True)
		plot_accuracy_over_var(list(accuracy_over_alpha_train.keys()), list(accuracy_over_alpha_train.values()),
		                       "alpha", "graphics", max_value=False, legend=["Cross Validation score", "Training score"])


def main(args):
	number_classes = args.number_classes

	if args.train:
		train(number_classes, retrain=args.retrain)
	if args.interpretation:
		interpretation(number_classes, retrain=args.retrain)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--train", action="store_true")
	parser.add_argument("--interpretation", action="store_true")
	parser.add_argument("--number_classes", type=int, default=50)
	parser.add_argument("--retrain", type=float, default=0)

	args = parser.parse_args()
	main(args)
