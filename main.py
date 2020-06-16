from utils import read_data, shuffle_split_data
from classifier import pick_best_model, predict


def main():
	x, y = read_data()
	x, _, y, _ = shuffle_split_data(x, y, 0.9999, True)  # Pruning data just for testing
	config, best_model, tokenizer = pick_best_model('modelo', x, y)
	
	print(predict('modelo', ['EZPIZI']))


if __name__ == '__main__':
	main()
