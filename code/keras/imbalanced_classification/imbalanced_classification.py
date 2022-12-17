"""
Title: Imbalanced classification: credit card fraud detection
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/05/28
Last modified: 2020/04/17 (git commit: d8288ba)
Last modified: 2022/12/17 (kevincoakley)
Description: Demonstration of how to handle highly imbalanced classification problems.
"""
"""
## Introduction

This example looks at the
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud/)
dataset to demonstrate how
to train a classification model on data with highly imbalanced classes.
"""

"""
## First, vectorize the CSV data
"""

import argparse, csv, os, random, sys, yaml
from tokenize import String
from datetime import datetime
import numpy as np
import tensorflow as tf

script_version = "1.5.0"


def imbalanced_classification(run_number, deterministic=False, seed_val=1, run_name="", save_model=False):

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

    # Get the real data from https://www.kaggle.com/mlg-ulb/creditcardfraud/
    fname = "creditcard.csv"

    all_features = []
    all_targets = []
    with open(fname) as f:
        for i, line in enumerate(f):
            if i == 0:
                print("HEADER:", line.strip())
                continue  # Skip header
            fields = line.strip().split(",")
            all_features.append([float(v.replace('"', "")) for v in fields[:-1]])
            all_targets.append([int(fields[-1].replace('"', ""))])
            if i == 1:
                print("EXAMPLE FEATURES:", all_features[-1])

    features = np.array(all_features, dtype="float32")
    targets = np.array(all_targets, dtype="uint8")
    print("features.shape:", features.shape)
    print("targets.shape:", targets.shape)

    """
    ## Prepare a validation set
    """

    num_val_samples = int(len(features) * 0.2)
    train_features = features[:-num_val_samples]
    train_targets = targets[:-num_val_samples]
    val_features = features[-num_val_samples:]
    val_targets = targets[-num_val_samples:]

    print("Number of training samples:", len(train_features))
    print("Number of validation samples:", len(val_features))

    """
    ## Analyze class imbalance in the targets
    """

    counts = np.bincount(train_targets[:, 0])
    print(
        "Number of positive samples in training data: {} ({:.2f}% of total)".format(
            counts[1], 100 * float(counts[1]) / len(train_targets)
        )
    )

    weight_for_0 = 1.0 / counts[0]
    weight_for_1 = 1.0 / counts[1]

    """
    ## Normalize the data using training set statistics
    """

    mean = np.mean(train_features, axis=0)
    train_features -= mean
    val_features -= mean
    std = np.std(train_features, axis=0)
    train_features /= std
    val_features /= std

    """
    ## Build a binary classification model
    """

    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Dense(
                256, activation="relu", input_shape=(train_features.shape[-1],)
            ),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.summary()

    """
    ## Train the model with `class_weight` argument
    """

    model.compile(
        optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=["accuracy"]
    )

    #callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.h5")]
    class_weight = {0: weight_for_0, 1: weight_for_1}

    epochs = 30

    model.fit(
        train_features,
        train_targets,
        batch_size=2048,
        epochs=epochs,
        verbose=2,
        #callbacks=callbacks,
        validation_data=(val_features, val_targets),
        class_weight=class_weight,
    )

    score = model.evaluate(val_features, val_targets, verbose=0)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    """
    ## Save the model
    """
    if save_model:
        base_name = os.path.basename(sys.argv[0]).split('.')[0]

        if len(run_name) >= 1:
            base_name = base_name + "_" + run_name

        features_predicted = model.predict(val_features)
        np.save(base_name + "_predict_" + str(run_number) + ".npy", features_predicted)
        model.save(base_name + "_model_" + str(run_number) + ".h5")

    return score[0], score[1], epochs


def get_system_info():
    if os.path.exists("system_info.py"):
        import system_info
        sysinfo = system_info.get_system_info()

        with open("imbalanced_classification_system_info.yaml", "w") as system_info_file:
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

        print("\nImbalanced Classification Count: %s of %s [%s]\n======================\n" % (str(x + 1), args.num_runs, seed_val))
        test_loss, test_accuracy, epochs = imbalanced_classification(x + 1, deterministic=args.deterministic, seed_val=seed_val, run_name=args.run_name, save_model=args.save_model)
        save_score(test_loss, test_accuracy, epochs, deterministic=args.deterministic, seed_val=seed_val, run_name=args.run_name)

"""
## Conclusions

At the end of training, out of 56,961 validation transactions, we are:

- Correctly identifying 66 of them as fraudulent
- Missing 9 fraudulent transactions
- At the cost of incorrectly flagging 441 legitimate transactions

In the real world, one would put an even higher weight on class 1,
so as to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets  declined in an online purchase -- this is why.

Example available on HuggingFace.

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/🤗%20Model-Imbalanced%20Classification-black.svg)](https://huggingface.co/keras-io/imbalanced_classification) | [![Generic badge](https://img.shields.io/badge/🤗%20Spaces-Imbalanced%20Classification-black.svg)](https://huggingface.co/spaces/keras-io/Credit_Card_Fraud_Detection) |

"""