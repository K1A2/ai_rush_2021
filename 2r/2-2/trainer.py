from evaluator import F1Score
from model import BaseLine
import tensorflow as tf
from preprocessor import Dataset


class Trainer:
    def __init__(self):
        model = BaseLine()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy', F1Score()])
        self._model = model
        self._dataset = Dataset()

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def dataset(self) -> Dataset:
        return self._dataset
