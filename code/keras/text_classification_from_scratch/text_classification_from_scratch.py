"""
Title: Text classification from scratch
Authors: Mark Omernick, Francois Chollet
Date created: 2019/11/06
Last modified: 2020/05/17 (git commit: e40fab4)
Last modified: 2023/1/1 (kevincoakley)
Description: Text sentiment classification starting from raw text files.
Accelerator: GPU
"""
"""
## Introduction

This example shows how to do text classification starting from raw text (as
a set of text files on disk). We demonstrate the workflow on the IMDB sentiment
classification dataset (unprocessed version). We use the `TextVectorization` layer for
 word splitting & indexing.
"""

"""
## Setup
"""

import tensorflow as tf
import numpy as np

script_version = "1.0.0"

"""
## Load the data: IMDB movie review sentiment classification

Let's download the data and inspect its structure.
"""

"""shell
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
"""

"""
The `aclImdb` folder contains a `train` and `test` subfolder:
"""

"""shell
ls aclImdb
"""

"""shell
ls aclImdb/test
"""

"""shell
ls aclImdb/train
"""

"""
The `aclImdb/train/pos` and `aclImdb/train/neg` folders contain text files, each of
 which represents one review (either positive or negative):
"""

"""shell
cat aclImdb/train/pos/6248_7.txt
"""

"""
We are only interested in the `pos` and `neg` subfolders, so let's delete the rest:
"""

"""shell
rm -r aclImdb/train/unsup
"""

import argparse, csv, os, random, sys, yaml
from datetime import datetime 

def text_classification_from_scratch(run_number, deterministic=False, seed_val=1, run_name="", save_model=False):

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
            print("Enabled op determinism")



    """
    You can use the utility `tf.keras.utils.text_dataset_from_directory` to
    generate a labeled `tf.data.Dataset` object from a set of text files on disk filed
    into class-specific folders.

    Let's use it to generate the training, validation, and test datasets. The validation
    and training datasets are generated from two subsets of the `train` directory, with 20%
    of samples going to the validation dataset and 80% going to the training dataset.

    Having a validation dataset in addition to the test dataset is useful for tuning
    hyperparameters, such as the model architecture, for which the test dataset should not
    be used.

    Before putting the model out into the real world however, it should be retrained using all
    available training data (without creating a validation dataset), so its performance is maximized.

    When using the `validation_split` & `subset` arguments, make sure to either specify a
    random seed, or to pass `shuffle=False`, so that the validation & training splits you
    get have no overlap.

    """

    batch_size = 32
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        "aclImdb/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=1337,
    )
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        "aclImdb/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=1337,
    )
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        "aclImdb/test", batch_size=batch_size
    )

    print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
    print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
    print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")

    """
    ## Prepare the data

    In particular, we remove `<br />` tags.
    """

    from tensorflow.keras.layers import TextVectorization
    import string
    import re

    # Having looked at our data above, we see that the raw text contains HTML break
    # tags of the form '<br />'. These tags will not be removed by the default
    # standardizer (which doesn't strip HTML). Because of this, we will need to
    # create a custom standardization function.
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
        return tf.strings.regex_replace(
            stripped_html, f"[{re.escape(string.punctuation)}]", ""
        )


    # Model constants.
    max_features = 20000
    embedding_dim = 128
    sequence_length = 500

    # Now that we have our custom standardization, we can instantiate our text
    # vectorization layer. We are using this layer to normalize, split, and map
    # strings to integers, so we set our 'output_mode' to 'int'.
    # Note that we're using the default split function,
    # and the custom standardization defined above.
    # We also set an explicit maximum sequence length, since the CNNs later in our
    # model won't support ragged sequences.
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    # Now that the vocab layer has been created, call `adapt` on a text-only
    # dataset to create the vocabulary. You don't have to batch, but for very large
    # datasets this means you're not keeping spare copies of the dataset in memory.

    # Let's make a text-only dataset (no labels):
    text_ds = raw_train_ds.map(lambda x, y: x)
    # Let's call `adapt`:
    vectorize_layer.adapt(text_ds)

    """
    ## Two options to vectorize the data

    There are 2 ways we can use our text vectorization layer:

    **Option 1: Make it part of the model**, so as to obtain a model that processes raw
    strings, like this:
    """

    """

    ```python
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
    x = vectorize_layer(text_input)
    x = layers.Embedding(max_features + 1, embedding_dim)(x)
    ...
    ```

    **Option 2: Apply it to the text dataset** to obtain a dataset of word indices, then
    feed it into a model that expects integer sequences as inputs.

    An important difference between the two is that option 2 enables you to do
    **asynchronous CPU processing and buffering** of your data when training on GPU.
    So if you're training the model on GPU, you probably want to go with this option to get
    the best performance. This is what we will do below.

    If we were to export our model to production, we'd ship a model that accepts raw
    strings as input, like in the code snippet for option 1 above. This can be done after
    training. We do this in the last section.


    """


    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label


    # Vectorize the data.
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    # Do async prefetching / buffering of the data for best performance on GPU.
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    test_ds = test_ds.cache().prefetch(buffer_size=10)

    """
    ## Build a model

    We choose a simple 1D convnet starting with an `Embedding` layer.
    """

    from tensorflow.keras import layers

    # A integer input for vocab indices.
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(max_features, embedding_dim)(inputs)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    """
    ## Train the model
    """

    epochs = 3

    # Fit the model using the train and test datasets.
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    """
    ## Evaluate the trained model
    """
    score = model.evaluate(val_ds, verbose=0)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    """
    ## Save the model
    """
    if save_model:
        base_name = os.path.basename(sys.argv[0]).split('.')[0]

        if len(run_name) >= 1:
            base_name = base_name + "_" + run_name

        y_predicted = model.predict(val_ds)
        np.save(base_name + "_predict_" + str(run_number) + ".npy", y_predicted)
        model.save(base_name + "_model_" + str(run_number) + ".h5")

    return score[0], score[1], epochs


def get_system_info():
    if os.path.exists("system_info.py"):
        base_name = os.path.basename(sys.argv[0]).split('.')[0]        

        import system_info
        sysinfo = system_info.get_system_info()

        with open("%s.yaml" % base_name, "w") as system_info_file:
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

        print("\Text Classification From Scratch Count: %s of %s [%s]\n======================\n" % (str(x + 1), args.num_runs, seed_val))
        test_loss, test_accuracy, epochs = text_classification_from_scratch(x + 1, deterministic=args.deterministic, seed_val=seed_val, run_name=args.run_name, save_model=args.save_model)
        save_score(test_loss, test_accuracy, epochs, deterministic=args.deterministic, seed_val=seed_val, run_name=args.run_name)
