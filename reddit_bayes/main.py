import pickle
import time

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB


from reddit_classification.utils import convert_input
from reddit_classification.data_treatment import read_data


def main():
	initial_time = time.time()
	number_classes = 0
	if number_classes:
		X, y = read_data(f"data/pruned_{number_classes}_entries.tsv")
	else:
		X, y = read_data()

	train_split_index = int(len(X) * 0.8)

	X_train, X_test = X[:train_split_index], X[train_split_index:]
	y_train, y_test = y[:train_split_index], y[train_split_index:]

	label_encoder = LabelEncoder()
	label_encoder.fit(y_train)
	y_train = label_encoder.transform(y_train)
	y_test = label_encoder.transform(y_test)

	# extract features
	X_train, X_test, token_frequencies = convert_input(X_train, X_test)

	# save token frequencies
	with open(f"reddit_bayes/token_frequencies/token_frequencies_{number_classes}", 'wb') as file:
		pickle.dump(token_frequencies, file)

	# train
	model = MultinomialNB(alpha=0.1)
	model.fit(X_train, y_train)

	# save model
	with open(f"reddit_bayes/models/model_{number_classes}", 'wb') as file:
		pickle.dump(model, file)

	y_pred_proba = model.predict_proba(X_test)
	y_pred = np.argmax(y_pred_proba, axis=1)

	def precision_at_k(y_true, y_pred, k=5):
		y_true = np.array(y_true)
		y_pred = np.array(y_pred)
		y_pred = np.argsort(y_pred, axis=1)
		y_pred = y_pred[:, ::-1][:, :k]
		arr = [y in s for y, s in zip(y_true, y_pred)]
		return np.mean(arr)

	print('precision@1 =', np.mean(y_test == y_pred))
	print('precision@3 =', precision_at_k(y_test, y_pred_proba, 3))
	print('precision@5 =', precision_at_k(y_test, y_pred_proba, 20))

	print(time.time() - initial_time)


if __name__ == "__main__":
	main()
