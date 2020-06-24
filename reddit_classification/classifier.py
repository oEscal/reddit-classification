from keras import Sequential, layers, regularizers, callbacks
from .utils import save_model, save_logs, tokenize, convert_input, shuffle_split_data, best_neuron_number, \
    sentiment_analysis, save_tokenizer, get_tokenizer, save_confusion_matrix
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from .graphics import plot_accuracy_function, merge_accuracy_fold
from sklearn.metrics import confusion_matrix
import statistics


def create_model(input_dim, labels_size, n_samples, regularization_factor=0.001, neurons_=100):
    model = Sequential()

    neurons = max(neurons_, best_neuron_number(input_dim, labels_size, n_samples) * 2)

    model.add(layers.Dense(neurons, activation="relu",
                           input_dim=input_dim, kernel_regularizer=regularizers.l2(l=regularization_factor)))

    model.add(
        layers.Dense(neurons // 2, activation="relu", kernel_regularizer=regularizers.l2(l=regularization_factor)))

    model.add(layers.Dense(labels_size, activation='softmax'))

    model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def pick_best_model(identifier, x, y, output_path, epochs=200, batch_size=500, save_logs_status=True, neurons=100,
                    regularization_factor=0.001):
    labels_size = len(set(y))

    status, data = get_tokenizer(labels_size)

    if not status:
        x_train, x_test, y_train, y_test = shuffle_split_data(x, y, 0.2, False)

        x_train_sentiments = [sentiment_analysis(text) for text in x_train]
        x_test_sentiments = [sentiment_analysis(text) for text in x_test]

        x_train = [tokenize(text) for text in x_train]
        x_test = [tokenize(text) for text in x_test]

        x_train, x_test, tf = convert_input(x_train, x_test)

        x_train = np.append(x_train, np.asarray(x_train_sentiments).reshape(-1, 1), axis=1)
        x_test = np.append(x_test, np.asarray(x_test_sentiments).reshape(-1, 1), axis=1)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        label_enncoder = LabelEncoder()
        y_train_integer_encoded = label_enncoder.fit_transform(y_train).reshape(-1, 1)
        y_test_integer_encoded = label_enncoder.transform(y_test).reshape(-1, 1)

        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = onehot_encoder.fit_transform(y_train_integer_encoded)
        y_test = onehot_encoder.transform(y_test_integer_encoded)

        new_data = {
            'tf': tf,
            'onehot_encoder': onehot_encoder,
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test
        }

        save_tokenizer(labels_size, new_data)

    else:
        print("Cached tokenizer")
        tf, x_train, x_test, y_train, y_test, onehot_encoder = data['tf'], data['x_train'], data['x_test'], \
                                                               data['y_train'], data['y_test'], data['onehot_encoder']

    vocab_size = len(tf.get_feature_names())

    n_samples = x_train.shape[0]
    kf = KFold(n_splits=5)

    fold_var = 1

    early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss', patience=5)

    f_name = f"_last_same_n_neurons_regularization_factor_{regularization_factor}_new"

    train_acc = []
    cv_acc = []
    all_data = []

    for train_index, val_index in kf.split(np.zeros(n_samples), y_train):
        temp_x = np.append(x_train[train_index], x_train[val_index], 0)
        temp_y = np.append(y_train[train_index], y_train[val_index], 0)

        model = create_model(input_dim=vocab_size + 1, labels_size=labels_size, n_samples=n_samples, neurons_=neurons)

        history = model.fit(temp_x, temp_y, validation_split=0.2, verbose=1, shuffle=False,
                            callbacks=[early_stopping_callback],
                            epochs=epochs, batch_size=batch_size)

        plot_accuracy_function(history.history, f'{labels_size}_{fold_var}_fold.png', f'Model accuracy fold {fold_var}',
                               path=f'graphics/accuracy_function/{neurons}_neurons_{labels_size}{f_name}')

        train_acc.append(model.evaluate(x_train[train_index], y_train[train_index])[-1])
        cv_acc.append(model.evaluate(x_train[val_index], y_train[val_index])[-1])

        all_data.append((history.history['accuracy'], history.history['val_accuracy']))

        fold_var += 1

    model = create_model(input_dim=vocab_size + 1, labels_size=labels_size, n_samples=n_samples, neurons_=neurons)

    history = model.fit(np.append(x_train, x_test, 0), np.append(y_train, y_test, 0), validation_split=0.2,
                        verbose=1,
                        shuffle=False,
                        callbacks=[early_stopping_callback],
                        epochs=epochs, batch_size=batch_size)

    plot_accuracy_function(history.history, f'{labels_size}_final_png', f'Model accuracy final retrain',
                           path=f'graphics/accuracy_function/{neurons}_neurons_{labels_size}{f_name}', l='test')

    merge_accuracy_fold(all_data, path=f'graphics/accuracy_function/{neurons}_neurons_{labels_size}{f_name}')

    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'max_len': vocab_size
    }

    if save_logs_status:
        data = {
            'data_size': len(x),
            'train_acc': model.evaluate(x_train, y_train)[-1],
            'test_acc': model.evaluate(x_test, y_test)[-1],
            'mean_train_acc': float(statistics.mean(train_acc)),
            'mean_val_acc': float(statistics.mean(cv_acc))
        }
        save_logs(f"{neurons}_neurons_{labels_size}{f_name}", data)

    c_matrix = confusion_matrix(onehot_encoder.inverse_transform(model.predict(x_test)),
                                onehot_encoder.inverse_transform(y_test))

    save_confusion_matrix(c_matrix, f"{neurons}_neurons_{labels_size}{f_name}")

    save_model(identifier, config, model.model.get_weights(), tf, output_path)

    return config, model, tf
