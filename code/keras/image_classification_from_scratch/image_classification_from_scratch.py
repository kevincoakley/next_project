"""
Title: Image classification from scratch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/27
Last modified: 2020/04/28 (git commit: 52d879a)
Last modified: 2022/11/28 (kevincoakley)
Description: Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.
"""
"""
## Introduction

This example shows how to do image classification from scratch, starting from JPEG
image files on disk, without leveraging pre-trained weights or a pre-made Keras
Application model. We demonstrate the workflow on the Kaggle Cats vs Dogs binary
 classification dataset.

We use the `image_dataset_from_directory` utility to generate the datasets, and
we use Keras image preprocessing layers for image standardization and data augmentation.
"""

"""
## Setup
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

script_version = "1.0.0"

"""
## Load the data: the Cats vs Dogs dataset

### Raw data download

First, let's download the 786M ZIP archive of the raw data:
"""

"""shell
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
"""

"""shell
unzip -q kagglecatsanddogs_5340.zip
ls
"""

"""
Now we have a `PetImages` folder which contain two subfolders, `Cat` and `Dog`. Each
 subfolder contains image files for each category.
"""

"""shell
ls PetImages
"""
import argparse, csv, os, random, sys, yaml
from datetime import datetime 
import numpy as np

def image_classification_from_scratch(run_number, deterministic=False, seed_val=1, run_name="", save_model=False):

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
    ### Filter out corrupted images

    When working with lots of real-world image data, corrupted images are a common
    occurence. Let's filter out badly-encoded images that do not feature the string "JFIF"
    in their header.
    """

    import os

    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)

    """
    ## Generate a `Dataset`
    """

    image_size = (180, 180)
    batch_size = 128

    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    """
    ## Using image data augmentation

    When you don't have a large image dataset, it's a good practice to artificially
    introduce sample diversity by applying random yet realistic transformations to the
    training images, such as random horizontal flipping or small random rotations. This
    helps expose the model to different aspects of the training data while slowing down
    overfitting.
    """

    # data augmentation commented out because it introduces non-determinism
    data_augmentation = keras.Sequential(
        [
            #layers.RandomFlip("horizontal"),
            #layers.RandomRotation(0.1),
        ]
    )

    """
    ## Standardizing the data

    Our image are already in a standard size (180x180), as they are being yielded as
    contiguous `float32` batches by our dataset. However, their RGB channel values are in
    the `[0, 255]` range. This is not ideal for a neural network;
    in general you should seek to make your input values small. Here, we will
    standardize values to be in the `[0, 1]` by using a `Rescaling` layer at the start of
    our model.
    """

    """
    ## Two options to preprocess the data

    There are two ways you could be using the `data_augmentation` preprocessor:

    **Option 1: Make it part of the model**, like this:

    ```python
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    ...  # Rest of the model
    ```

    With this option, your data augmentation will happen *on device*, synchronously
    with the rest of the model execution, meaning that it will benefit from GPU
    acceleration.

    Note that data augmentation is inactive at test time, so the input samples will only be
    augmented during `fit()`, not when calling `evaluate()` or `predict()`.

    If you're training on GPU, this may be a good option.

    **Option 2: apply it to the dataset**, so as to obtain a dataset that yields batches of
    augmented images, like this:

    ```python
    augmented_train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y))
    ```

    With this option, your data augmentation will happen **on CPU**, asynchronously, and will
    be buffered before going into the model.

    If you're training on CPU, this is the better option, since it makes data augmentation
    asynchronous and non-blocking.

    In our case, we'll go with the second option. If you're not sure
    which one to pick, this second option (asynchronous preprocessing) is always a solid choice.
    """

    """
    ## Configure the dataset for performance

    Let's apply data augmentation to our training dataset,
    and let's make sure to use buffered prefetching so we can yield data from disk without
    having I/O becoming blocking:

    """

    # Apply `data_augmentation` to the training images.
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    """
    ## Build a model

    We'll build a small version of the Xception network. We haven't particularly tried to
    optimize the architecture; if you want to do a systematic search for the best model
    configuration, consider using
    [KerasTuner](https://github.com/keras-team/keras-tuner).

    Note that:

    - We start the model with the `data_augmentation` preprocessor, followed by a
    `Rescaling` layer.
    - We include a `Dropout` layer before the final classification layer.
    """


    def make_model(input_shape, num_classes):
        inputs = keras.Input(shape=input_shape)

        # Entry block
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes

        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units, activation=activation)(x)
        return keras.Model(inputs, outputs)


    model = make_model(input_shape=image_size + (3,), num_classes=2)

    """
    ## Train the model
    """

    epochs = 25

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]
    # jit_compile=True causes a crash 
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,  # Enable XLA compilation for faster training
    )
    # Comment out the callback because we don't need to save the model checkpoint
    # for each epoch.
    model.fit(
        train_ds,
        epochs=epochs,
        #callbacks=callbacks,
        validation_data=val_ds,
    )

    """
    We get to ~96% validation accuracy after training for 25 epochs on the full dataset.
    """

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

        print("\nImage Classification From Scratch Count: %s of %s [%s]\n======================\n" % (str(x + 1), args.num_runs, seed_val))
        test_loss, test_accuracy, epochs = image_classification_from_scratch(x + 1, deterministic=args.deterministic, seed_val=seed_val, run_name=args.run_name, save_model=args.save_model)
        save_score(test_loss, test_accuracy, epochs, deterministic=args.deterministic, seed_val=seed_val, run_name=args.run_name)
