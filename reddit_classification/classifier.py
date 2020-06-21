from keras import regularizers
from tensorflow.keras import Sequential, layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from .utils import read_model, save_model, save_logs, tokenize, convert_input, shuffle_split_data
from matplotlib import pyplot as plt


def create_model(input_dim, labels_size, regularization_factor=0.001):
    model = Sequential()
    model.add(layers.Dense(100, activation="relu", input_dim=input_dim,
                           kernel_regularizer=regularizers.l2(l=regularization_factor)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(100, activation="relu", kernel_regularizer=regularizers.l2(l=regularization_factor)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(labels_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def pick_best_model(identifier, x, y, output_path, epochs=50, batch_size=500, save_logs_status=True):
    x = [tokenize(text) for text in x]

    x_train, x_test, y_train, y_test = shuffle_split_data(x, y, 0.2, True)
    x_train, x_test, tf = convert_input(x_train, x_test)
    vocab_size = len(tf.get_feature_names())
    labels_size = len(set(y))

    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=True,
                            validation_split=0.2, input_dim=vocab_size, labels_size=labels_size)

    history = model.fit(x_train, y_train)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'max_len': vocab_size
    }

    if save_logs_status:
        data = {
            'data_size': len(x),
            'train_acc': model.score(x_train, y_train),
            'test_acc': model.score(x_test, y_test)
        }
        save_logs(identifier, data)

    save_model(identifier, config, model, tf, output_path)

    return history, config, model, tf


def predict(identifier, x):
    config, model, tf = read_model(identifier)

    classifier = KerasClassifier(build_fn=create_model, epochs=config['epochs'], batch_size=config['batch_size'])

    classifier.model = model
    x = [tokenize(text) for text in x]
    x = tf.transform(x).todense()

    return model.predict(x)
