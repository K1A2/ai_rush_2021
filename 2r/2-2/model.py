import tensorflow as tf
import tensorflow.keras as keras


class BaseLine(tf.keras.Model):
    def __init__(self):
        super(BaseLine, self).__init__()

        self.input_kind_now = keras.Input(shape=(28,), dtype='float32', name='input_kind_now')

        self.input_kind_next = keras.Input(shape=(28,), dtype='float32', name='input_kind_next')

        self.input_features = keras.Input(shape=(1,), dtype='float32', name='input_features')

        self.main_output = keras.layers.Dense(1, activation='sigmoid', name='main_output')

    def get_config(self):
        return {}

    def call(self, inputs, training=None, mask=None):
        link_no = tf.cast(inputs[0], dtype=tf.float32)
        self.input_kind_now = self.input_kind_now(link_no)

        link_ne = tf.cast(inputs[1], dtype=tf.float32)
        self.input_kind_next = self.input_kind_next(link_ne)

        link_fe = tf.cast(inputs[2], dtype=tf.float32)
        self.input_features = self.input_features(link_fe)

        input_kind_now_embedd = keras.layers.Embedding(210, 20, input_length=31)(self.input_kind_now)
        x1 = keras.layers.Flatten()(input_kind_now_embedd)
        x1 = keras.layers.Dense(15, activation='relu')(x1)
        x1 = keras.layers.Dense(15, activation='relu')(x1)

        input_kind_next_embedd = keras.layers.Embedding(210, 20, input_length=31)(self.input_kind_next)
        x2 = keras.layers.Flatten()(input_kind_next_embedd)
        x2 = keras.layers.Dense(15, activation='relu')(x2)
        x2 = keras.layers.Dense(15, activation='relu')(x2)

        x3 = keras.layers.Dense(15, activation='relu')(self.input_features)

        x = keras.layers.concatenate([x1, x2, x3])
        x = keras.layers.Dense(8, activation='relu')(x)

        return self.main_output(x)
