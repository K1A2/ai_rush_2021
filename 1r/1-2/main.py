import argparse
import os
import tensorflow as tf
import pickle

from pandas.core.frame import DataFrame

from trainer import Trainer
from model import BaselineModel
from tensorflow.python.client import device_lib

import nsml

def bind_model(trainer: Trainer, main_args):
    def save(dirname, *args):
        trainer.model.save(os.path.join(dirname, 'model'))
        with open(os.path.join(dirname, "preprocessor.pckl"), "wb") as f:
            pickle.dump(trainer.preprocessor, f)

    def load(dirname, *args):
        trainer.model = tf.keras.models.load_model(os.path.join(dirname, 'model'))
        with open(os.path.join(dirname, "preprocessor.pckl"), "rb") as f:
            trainer.preprocessor = pickle.load(f)

    def infer(test_data : DataFrame):
        test_data = trainer.preprocessor.preprocess_test_data(test_data).batch(1)
        prediction = trainer.model.predict(test_data).flatten().tolist()
        # print(prediction)
        # prediction = trainer.preprocessor.normalizer.get_lable(prediction)
        # single prediction for last next duration
        print(prediction)
        return round(prediction[len(prediction) - 1])

    nsml.bind(save = save, load = load, infer = infer)

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'XLA_GPU']

# class Load():
#     def __init__(self, model):
#         self.model = model
# load = Load(None)
# def load_model_custom(dirname, *args):
#     model = tf.keras.models.load_model(os.path.join(dirname, 'model'))
#     load.model = model

def main():
    args = argparse.ArgumentParser()

    # RESERVED FOR NSML
    args.add_argument('--mode', type=str, default='train', help='nsml submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    args.add_argument('--ephoc', type=int, default=3)
    args.add_argument('--batch', type=int, default=32)
    args.add_argument('--prodir', type=str, default='q')
    args.add_argument('--load_model', type=int, default=0)
    args.add_argument('--session_name', type=str, default='')
    args.add_argument('--check_name', type=str, default='')
    config = args.parse_args()

    # ADD MORE ARGS IF NECESSARY
    # NSML - Bind Model
    # Building model and preprocessor
    model = BaselineModel(optimizer=tf.keras.optimizers.Adam(),
                          loss=tf.losses.MeanSquaredError(),
                          metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()],
                          epochs=config.ephoc, batch_size=config.batch)

    trainer = Trainer(model)

    # nsml.bind() should be called before nsml.paused()
    bind_model(trainer, config)
    if config.pause:
        nsml.paused(scope=locals())

    # Train Model
    if config.mode == "train":
        if config.load_model == 1:
            point = config.check_name
            if point == '0':
                nsml.load(checkpoint='best', session='KR95422/airush2021-1-2a/' + config.session_name)
                nsml.save('best')
            else:
                start, end = tuple(map(int, point.split('_')))
                for i in range(start, end + 1):
                    nsml.load(checkpoint='best' + str(i), session='KR95422/airush2021-1-2a/' + config.session_name)
                    nsml.save('best' + str(i))
        else:
            print("Training model...")
            print(device_lib.list_local_devices())
            # nsml.load(checkpoint='best', session='KR95422/airush2021-1-2a/23', load_fn=load_model_custom)
            # model.set_weights(load.model.get_weights())
            trainer.train()
    

if __name__ == '__main__':
    main()