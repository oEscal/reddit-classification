from .classifier import pick_best_model
from .data_treatment import read_data


def main(input_path="data", output_path="models"):
    entries = 50
    x, y = read_data(f"{input_path}/pruned_{entries}_entries.tsv")
    identifier = f"{entries}_entries_model_last_same_neurons"
    config, best_model, tokenizer, data = pick_best_model(identifier, x, y, output_path, neurons=200)


if __name__ == '__main__':
    main()
