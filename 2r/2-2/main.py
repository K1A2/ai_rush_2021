import argparse
import pickle

import tensorflow as tf
import os
import subprocess

from evaluator import F1Score
from trainer import Trainer

import nsml
from nsml import DATASET_PATH

import tensorflow.keras as keras


def bind_model(trainer: Trainer):
    train_networks_dir = os.path.join(DATASET_PATH, 'train', 'train_data')
    if os.path.exists(train_networks_dir):
        print("train_data 가 접근 가능하면, bind 시점에 전처리하고 메모리에 올려둔다")
        preprop_networks_df = trainer.dataset.preprocess_networks(train_networks_dir)

    def save(dirname, *args):
        # if os.path.exists(train_networks_dir):
        #     print("train_data 가 접근 가능하면, bind 시점에 전처리하고 메모리에 올려둔다")
        #     preprop_networks_df = trainer.dataset.preprocess_networks_2()

        # 모델 저장 할때마다, preprop networks 를 포함시킨다. infer 할때 모델과 같이 쓸 수 있다
        networks_df_pickle = os.path.join(dirname, 'networks.pkl')
        if not os.path.exists(networks_df_pickle):
            print(f"networks 전처리테이블을 모델과 함께 저장합니다: {networks_df_pickle}")
            # preprop_networks_df.to_pickle(networks_df_pickle)
            with open(networks_df_pickle, 'wb') as f:
                pickle.dump(preprop_networks_df, f, pickle.HIGHEST_PROTOCOL)

        os.makedirs(dirname, exist_ok=True)
        trainer.model.save(os.path.join(dirname, 'model'))
        print("NSML save called. dirname contents")
        subprocess.check_call(['ls', '-lR', dirname])

    def load(dirname, *args):
        networks_df_pickle = os.path.join(dirname, 'networks.pkl')
        with open(networks_df_pickle, 'rb') as f:
            trainer.dataset.set_preprop_networks(pickle.load(f))

        print("NSML load called. dirname contents")
        subprocess.check_call(['ls', '-lR', dirname])
        trainer.model = tf.keras.models.load_model(
            os.path.join(dirname, 'model'),
            custom_objects={'F1Score': F1Score()}
        )
        print(f"model load weights done {trainer.model}")

    def infer(data_basedir, **kwargs):
        tf_ds = trainer.dataset.preprocess(os.path.join(data_basedir, 'test', 'test_data', 'test_data.csv'))
        pred = trainer.model.predict(tf_ds).flatten().tolist()
        return [round(x) for x in pred]

    nsml.bind(save=save, load=load, infer=infer)


class NsmlSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.f1_min = 0
        self.count = 1

    def on_epoch_end(self, epoch, logs=None):
        f1 = logs['f1_score']
        if self.f1_min < f1:
            nsml.report(step=self.count, scope=locals(), summary=True, f1=f1)
            nsml.save('best{0}'.format(self.count))
            # nsml.save('best')
            print('saved model! RMSE: %.6f' % (f1))
            self.f1_min = f1
            self.count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--pause', default=0)
    parser.add_argument('--ephoc', default=0, type=int)
    args = parser.parse_args()

    trainer = Trainer()
    bind_model(trainer)

    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == 'train':
        print('train')
        dataset_basedir = DATASET_PATH
        ds = trainer.dataset.preprocess(
            f'{dataset_basedir}/train/train_data/train_data.csv',
            f'{dataset_basedir}/train/train_label')

        def model_1():
            input_link_next_kind_road = keras.Input(shape=(10,), dtype='float32', name='input_link_next_kind_road')
            input_link_next_kind_road_embedd = keras.layers.Embedding(10, 4, input_length=1)(input_link_next_kind_road)
            flat4 = keras.layers.Flatten()(input_link_next_kind_road_embedd)

            input_link_next_level_road = keras.Input(shape=(11,), dtype='float32', name='input_link_next_level_road')
            input_link_next_level_road_embedd = keras.layers.Embedding(11, 4, input_length=1)(input_link_next_level_road)
            flat5 = keras.layers.Flatten()(input_link_next_level_road_embedd)

            input_link_next_kind = keras.Input(shape=(21,), dtype='float32', name='input_link_next_kind')
            input_link_next_kind_embedd = keras.layers.Embedding(21, 4, input_length=1)(input_link_next_kind)
            flat6 = keras.layers.Flatten()(input_link_next_kind_embedd)

            layer_nodes_next = keras.layers.concatenate([flat4, flat5, flat6])
            layer_nodes_next = keras.layers.Dense(64, activation='relu')(layer_nodes_next)
            # layer_nodes_next = keras.layers.Dense(32, activation='relu')(layer_nodes_next)

            # input_node_type = keras.Input(shape=(7,), dtype='float32', name='input_node_type')
            # input_node_type_embedd = keras.layers.Embedding(7, 4, input_length=1)(input_node_type)
            # flat7 = keras.layers.Flatten()(input_node_type_embedd)
            # layer_nodes_type = keras.layers.Dense(64, activation='relu')(flat7)

            input_features = keras.Input(shape=(1,), dtype='float32', name='input_features')

            input_angles = keras.Input(shape=(20,), dtype='float32', name='input_angles')
            input_angles_embedd = keras.layers.Embedding(21, 4, input_length=1)(input_angles)
            flat7 = keras.layers.Flatten()(input_angles_embedd)
            flat7 = keras.layers.Dense(16, activation='relu')(flat7)

            # layer_embedded = keras.layers.concatenate([layer_nodes_now, layer_nodes_next])
            # layer_embedded = keras.layers.Dense(64, activation='relu')(layer_embedded)
            # layer_embedded = keras.layers.Dense(64, activation='relu')(layer_nodes_next)
            # layer_embedded = keras.layers.Dense(32, activation='relu')(layer_embedded)

            layer_last = layer_nodes_next
            layer_last = keras.layers.Dense(64, activation='relu')(layer_last)
            layer_last = keras.layers.Dense(32, activation='relu')(layer_last)
            layer_last = keras.layers.Dropout(0.1)(layer_last)
            layer_last = keras.layers.Dense(16, activation='relu')(layer_last)
            layer_last2 = keras.layers.concatenate([layer_last, flat7, input_features])
            layer_last2 = keras.layers.Dense(32, activation='relu')(layer_last2)
            layer_last2 = keras.layers.Dense(16, activation='relu')(layer_last2)
            layer_last2 = keras.layers.Dropout(0.1)(layer_last2)
            layer_last2 = keras.layers.Dense(8, activation='relu')(layer_last2)
            layer_last2 = keras.layers.Dense(1, activation='sigmoid')(layer_last2)

            # model = keras.Model([input_link_now_kind_road, input_link_now_level_road, input_link_now_kind,
            #                      input_link_next_kind_road, input_link_next_level_road, input_link_next_kind, input_features], layer_last)
            model = keras.Model([input_link_next_kind_road, input_link_next_level_road, input_link_next_kind,
                                 input_features, input_angles], layer_last2)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=['accuracy', F1Score()])
            trainer.model = model
            print(trainer.model.summary())
            trainer.model.fit(
                [ds[0], ds[1], ds[2], ds[3], ds[5]],
                ds[4],
                epochs=args.ephoc,
                batch_size=16,
                callbacks=[NsmlSaveCallback()], shuffle=True)

        def model_3():
            input_link_next_kind_road = keras.Input(shape=(10,), dtype='float32', name='input_link_next_kind_road')
            input_link_next_kind_road_embedd = keras.layers.Embedding(10, 4, input_length=1)(input_link_next_kind_road)
            flat4 = keras.layers.Flatten()(input_link_next_kind_road_embedd)

            input_link_next_level_road = keras.Input(shape=(11,), dtype='float32', name='input_link_next_level_road')
            input_link_next_level_road_embedd = keras.layers.Embedding(11, 4, input_length=1)(input_link_next_level_road)
            flat5 = keras.layers.Flatten()(input_link_next_level_road_embedd)

            input_link_next_kind = keras.Input(shape=(21,), dtype='float32', name='input_link_next_kind')
            input_link_next_kind_embedd = keras.layers.Embedding(21, 4, input_length=1)(input_link_next_kind)
            flat6 = keras.layers.Flatten()(input_link_next_kind_embedd)

            layer_nodes_next = keras.layers.concatenate([flat4, flat5, flat6])
            layer_nodes_next = keras.layers.Dense(64, activation='relu')(layer_nodes_next)
            # layer_nodes_next = keras.layers.Dense(32, activation='relu')(layer_nodes_next)

            # input_node_type = keras.Input(shape=(7,), dtype='float32', name='input_node_type')
            # input_node_type_embedd = keras.layers.Embedding(7, 4, input_length=1)(input_node_type)
            # flat7 = keras.layers.Flatten()(input_node_type_embedd)
            # layer_nodes_type = keras.layers.Dense(64, activation='relu')(flat7)

            input_features = keras.Input(shape=(1,), dtype='float32', name='input_features')

            # layer_embedded = keras.layers.concatenate([layer_nodes_now, layer_nodes_next])
            # layer_embedded = keras.layers.Dense(64, activation='relu')(layer_embedded)
            layer_embedded = keras.layers.Dense(64, activation='relu')(layer_nodes_next)
            layer_embedded = keras.layers.Dense(32, activation='relu')(layer_embedded)

            layer_last = keras.layers.concatenate([layer_embedded, input_features])
            layer_last = keras.layers.Dense(32, activation='relu')(layer_last)
            layer_last = keras.layers.Dropout(0.1)(layer_last)
            layer_last = keras.layers.Dense(16, activation='relu')(layer_last)
            layer_last = keras.layers.Dense(8, activation='relu')(layer_last)
            layer_last = keras.layers.Dense(1, activation='sigmoid')(layer_last)

            # model = keras.Model([input_link_now_kind_road, input_link_now_level_road, input_link_now_kind,
            #                      input_link_next_kind_road, input_link_next_level_road, input_link_next_kind, input_features], layer_last)
            model = keras.Model([input_link_next_kind_road, input_link_next_level_road, input_link_next_kind,
                                 input_features], layer_last)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=['accuracy', F1Score()])
            trainer.model = model
            print(trainer.model.summary())
            trainer.model.fit(
                [ds[3], ds[4], ds[5], ds[6]],
                ds[7],
                epochs=args.ephoc,
                batch_size=16,
                callbacks=[NsmlSaveCallback()], shuffle=True)

        def model_2():
            input_link_next_kind_road_in = keras.Input(shape=(10,), dtype='float32', name='input_link_next_kind_road')
            input_link_next_kind_road = keras.layers.Dense(16, activation='relu')(input_link_next_kind_road_in)

            input_link_next_level_road_in = keras.Input(shape=(11,), dtype='float32', name='input_link_next_level_road')
            input_link_next_level_road = keras.layers.Dense(16, activation='relu')(input_link_next_level_road_in)

            input_link_next_kind_in = keras.Input(shape=(21,), dtype='float32', name='input_link_next_kind')
            input_link_next_kind = keras.layers.Dense(32, activation='relu')(input_link_next_kind_in)
            input_link_next_kind = keras.layers.Dense(16, activation='relu')(input_link_next_kind)

            layer_nodes_next = keras.layers.concatenate([input_link_next_kind_road, input_link_next_level_road, input_link_next_kind])
            layer_nodes_next = keras.layers.Dense(64, activation='relu')(layer_nodes_next)

            input_features = keras.Input(shape=(1,), dtype='float32', name='input_features')

            layer_last = keras.layers.concatenate([layer_nodes_next, input_features])
            layer_last = keras.layers.Dense(32, activation='relu')(layer_last)
            layer_last = keras.layers.Dropout(0.1)(layer_last)
            layer_last = keras.layers.Dense(16, activation='relu')(layer_last)
            layer_last = keras.layers.Dense(8, activation='relu')(layer_last)
            layer_last = keras.layers.Dense(1, activation='sigmoid')(layer_last)

            # model = keras.Model([input_link_now_kind_road, input_link_now_level_road, input_link_now_kind,
            #                      input_link_next_kind_road, input_link_next_level_road, input_link_next_kind, input_features], layer_last)
            model = keras.Model([input_link_next_kind_road_in, input_link_next_level_road_in, input_link_next_kind_in,
                                 input_features], layer_last)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=['accuracy', F1Score()])
            trainer.model = model
            print(trainer.model.summary())
            trainer.model.fit(
                [ds[0], ds[1], ds[2], ds[3]],
                ds[4],
                epochs=args.ephoc,
                batch_size=16,
                callbacks=[NsmlSaveCallback()], shuffle=True)

        model_3()

if __name__ == '__main__':
    main()
