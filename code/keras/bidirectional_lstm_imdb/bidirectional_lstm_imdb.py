"""
Title: Bidirectional LSTM on IMDB
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2020/05/03 (git commit: dad05fa)
Last modified: 2022/11/28 (kevincoakley)
Description: Train a 2-layer bidirectional LSTM on the IMDB movie review sentiment classification dataset.
"""
"""
## Setup
"""

import argparse, csv, os, random, sys, yaml
from datetime import datetime 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

script_version = "1.5.2"
max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

def bidirectional_lstm_imdb(run_number, deterministic=False, seed_val=1, run_name="", save_model=False):

    if deterministic or seed_val != 1:
        """
        ## Configure Tensorflow for fixed seed runs
        """
        major, minor, revision = tf.version.VERSION.split('.')

        if int(major) >= 2 and int(minor) >= 7:
            # Sets all random seeds for the program (Python, NumPy, and TensorFlow).
            # Supported in TF 2.7.0+
            tf.keras.utils.set_random_seed(seed_val)
            print("Setting random seed using tf.keras.utils.set_random_seed()")
        else:
            # for TF < 2.7
            random.seed(seed_val)
            np.random.seed(seed_val)
            tf.random.set_seed(seed_val)
            print("Setting random seeds manually")
        # Configures TensorFlow ops to run deterministically to enable reproducible 
        # results with GPUs (Supported in TF 2.8.0+)
        if int(major) >= 2 and int(minor) >= 8:
            tf.config.experimental.enable_op_determinism()
            print("TF config: enabled op determinism")

    """
    ## Build the model
    """

    epochs = 2

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()

    """
    ## Load the IMDB movie review sentiment data
    """

    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
        num_words=max_features
    )
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    # Use pad_sequence to standardize sequence length:
    # this will truncate sequences longer than 200 words and zero-pad sequences shorter than 200 words.
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    """
    ## Train and evaluate the model

    You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/bidirectional-lstm-imdb)
    and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/bidirectional_lstm_imdb).
    """

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))

    """
    ## Evaluate the trained model
    """
    score = model.evaluate(x_val, y_val, verbose=0)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    """
    ## Save the model
    """
    if save_model:
        base_name = os.path.basename(sys.argv[0]).split('.')[0]

        if len(run_name) >= 1:
            base_name = base_name + "_" + run_name

        y_predicted = model.predict(x_val)
        np.save(base_name + "_predict_" + str(run_number) + ".npy", y_predicted)
        model.save(base_name + "_model_" + str(run_number) + ".h5")


    return score[0], score[1], epochs


def get_system_info():
    if os.path.exists("system_info.py"):
        import system_info
        sysinfo = system_info.get_system_info()
        base_name = os.path.basename(sys.argv[0]).split('.')[0]    

        with open("%s_system_info.yaml" % base_name, "w") as system_info_file:
            yaml.dump(sysinfo, system_info_file, default_flow_style=False)

        return sysinfo
    else:
        return None


def save_score(test_loss, test_accuracy, epochs, deterministic, seed_val, run_name=""):
    csv_file = os.path.basename(sys.argv[0]).split('.')[0] + ".csv"
    write_header = False

    # If determistic is false and the seed value is 1 then the
    # seed value is totally random and we don't know what it is.
    if deterministic == False and seed_val == 1:
        seed_val = "random"

    if not os.path.isfile(csv_file):
        write_header = True
      
    with open(csv_file, "a") as csvfile:
        fieldnames = ["run_name", "script_version", "date_time", "python_version", "tensorflow_version",
        "tensorflow_compiler_version", "epochs", "random_seed", "test_loss", "test_accuracy"]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow({"run_name": run_name, "script_version": script_version, "date_time": datetime.now(), "python_version": sys.version.replace("\n", ""), "tensorflow_version": tf.version.VERSION,
        "tensorflow_compiler_version": tf.version.COMPILER_VERSION, "epochs": epochs, "random_seed": seed_val, "test_loss": test_loss, "test_accuracy": test_accuracy})


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        help="Run in deterministic mode",
        action='store_true'
    )

    parser.add_argument(
        "--random-seed-val",
        dest="random_seed_val",
        help="Pick a random int for the seed value every run and record it in the csv file",
        action="store_true"
    )

    parser.add_argument(
        "--seed-val",
        dest="seed_val",
        help="Set the seed value",
        type=int,
        default=1
    )

    parser.add_argument(
        "--num-runs",
        dest="num_runs",
        help="Number of training runs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--run-name",
        dest="run_name",
        help="Name of training run",
        default="",
    )

    parser.add_argument(
        "--save-model",
        dest="save_model",
        help="Save the model",
        action="store_true"
    )

    return parser.parse_args(args)


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])

    system_info = get_system_info()
    seed_val = args.seed_val

    for x in range(args.num_runs):

        if args.random_seed_val:
            seed_val = random.randint(0, 2**32 - 1)

        print("\nBidirectional LSTM on IMDB Count: %s of %s [%s]\n======================\n" % (str(x + 1), args.num_runs, seed_val))
        test_loss, test_accuracy, epochs = bidirectional_lstm_imdb(x + 1, deterministic=args.deterministic, seed_val=seed_val, run_name=args.run_name, save_model=args.save_model)
        save_score(test_loss, test_accuracy, epochs, deterministic=args.deterministic, seed_val=seed_val, run_name=args.run_name)
