import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    MaxPooling2D,
    Rescaling,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os


class ImageClassifier:
    def __init__(
        self, data_dir, img_height=224, img_width=224, batch_size=32, num_epochs=15
    ):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_ds, self.val_ds = self.load_data()
        self.class_names = self.train_ds.class_names
        self.model = self.build_model()

    def load_data(self):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds

    def build_model(self):
        data_augmentation = keras.Sequential(
            [
                RandomFlip(
                    "horizontal", input_shape=(self.img_height, self.img_width, 3)
                ),
                RandomRotation(0.1),
                RandomZoom(0.1),
            ]
        )

        num_classes = len(self.class_names)

        model = Sequential(
            [
                data_augmentation,
                Rescaling(1.0 / 255),
                Conv2D(16, 3, padding="same", activation="relu"),
                MaxPooling2D(),
                Conv2D(32, 3, padding="same", activation="relu"),
                MaxPooling2D(),
                Conv2D(64, 3, padding="same", activation="relu"),
                MaxPooling2D(),
                Dropout(0.2),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(num_classes),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def train_model(self):
        history = self.model.fit(
            self.train_ds, validation_data=self.val_ds, epochs=self.num_epochs
        )
        return history

    def save_model(self, file_path):
        self.model.save(file_path)


def main():
    print("Build the model")
    model = ImageClassifier("Dataset\Data for test")

    print("Train the model")
    model.train_model()

    print("Save the trained model")
    model.save_model("model2.h5")


if __name__ == "__main__":
    main()
