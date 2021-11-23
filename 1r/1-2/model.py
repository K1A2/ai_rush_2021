import tensorflow as tf
import nsml
from tensorflow.python.client import device_lib

class CustomModelSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CustomModelSaveCallback).__init__()
        self.rmse_max = 52
        self.count = 1

    def on_epoch_end(self, epoch, logs=None):
        rmse = logs['root_mean_squared_error']
        if self.rmse_max > rmse:
            nsml.report(step=self.count, scope=locals(), summary=True, rmse=rmse)
            nsml.save('best{0}'.format(self.count))
            # nsml.save('best')
            print('saved model! RMSE: %.6f' % (rmse))
            self.rmse_max = rmse
            self.count += 1

class BaselineModel(tf.keras.Model):
    def __init__(self, optimizer, loss, metrics, epochs, batch_size):
        super(BaselineModel, self).__init__()

        # model
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Conv1D(kernel_size=2, filters=32, input_shape=(2, 8)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=0.1, return_sequences=True),
            tf.keras.layers.LSTM(22, dropout=0.1, recurrent_dropout=0.1, return_sequences=True),
            tf.keras.layers.LSTM(22, dropout=0.1, recurrent_dropout=0.1, return_sequences=True),
            tf.keras.layers.LSTM(12),
            tf.keras.layers.Dense(units=8, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='relu')

            # # tf.keras.layers.Conv1D(kernel_size=2, filters=9, input_shape=(10, 9), activation='relu'),
            # tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.1, input_shape=(10, 9)),
            # tf.keras.layers.LSTM(86, return_sequences=True, recurrent_dropout=0.1),
            # tf.keras.layers.LSTM(48, return_sequences=True, recurrent_dropout=0.1),
            # tf.keras.layers.LSTM(10),
            # tf.keras.layers.Dense(8, activation='relu'),
            # tf.keras.layers.Dense(1, activation='relu')
        ])
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # training variables
        self.epochs = epochs
        self.batch_size = batch_size

    def get_available_gpus(self):
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def train(self, train_data):
        train_data_batch = train_data.batch(self.batch_size)
        self.fit(train_data_batch, epochs=self.epochs, shuffle=False, callbacks=[CustomModelSaveCallback()])
        print(self.dense.summary())
        # Group training_data into batches

        # , callbacks = [
            # tf.keras.callbacks.EarlyStopping(monitor='root_mean_squared_error', mode='min', verbose=1, patience=2)]
    
    def call(self, inputs):
        return self.dense(inputs)
