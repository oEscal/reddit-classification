from .classifier import pick_best_model
from .data_treatment import read_data


def main(input_path="data", output_path="models"):
    x, y = read_data(f"{input_path}/pruned_5_entries.tsv")
    print(len(x))
    return
    identifier = "5_entries_model"

    _, config, best_model, tokenizer = pick_best_model(identifier, x, y, output_path)


if __name__ == '__main__':
    main()
