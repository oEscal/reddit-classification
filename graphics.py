import matplotlib.pyplot as plt
from utils import occurrences_per_label
from data_treatment import read_data


def full_data_distribution(labels, counters):
	fig = plt.figure()
	ax = fig.add_axes([0, 0, 1, 1])
	ax.bar(labels, counters)
	plt.show()


def main():
	x, y = read_data()

	occurrences_per_label_dict = occurrences_per_label(y)
	full_data_distribution(occurrences_per_label_dict.keys()[:50], occurrences_per_label_dict.values()[:50])


if __name__ == '__main__':
	main()
