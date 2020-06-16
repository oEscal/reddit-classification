from data_treatment import read_data
from classifier import pick_best_model


def main():
	x, y = read_data('data/pruned_entries.tsv')
	identifier = "modelo_1"
	config, best_model, tokenizer = pick_best_model(identifier, x, y)


if __name__ == '__main__':
	main()
