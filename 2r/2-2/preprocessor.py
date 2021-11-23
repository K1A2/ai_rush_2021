import os

import tensorflow.keras.utils as ut
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

class Dataset(object):
    def __init__(self):
        self._passable_counts = None
        self._links_info = None
        self._nodes = None
        self.angle_count = 20
        pd.set_option('display.max_columns', 100)

    def preprocess_networks(self, networks_dir):
        """
        사용할 feature 예시로, 주행기록 기반으로 (middir, middirNext) 별 통행 수를 집계합니다.
        집계된 테이블은 train/test data 의 (middir, middirNext) 컬럼으로 조인 됩니다.
        """
        routelinks = pd.read_parquet(os.path.join(networks_dir, 'routelinks.parquet'))
        print('f', routelinks, sep='\n', end='\n')
        passable_counts = routelinks[['userid', 'middir', 'middirNext']] \
            .groupby(['middir', 'middirNext']).count().reset_index() \
            .rename({'userid': 'count'}, axis=1)
        # count = passable_counts[['count']].fillna({'count': 0}).astype({'count': float})
        # print(count.info)
        # print((count - count.mean()) / count.std())
        count = passable_counts[['count']]
        passable_counts[['count']] = (count - count.mean()) / count.std()
        self._passable_counts = passable_counts

        passcodes = pd.read_parquet(os.path.join(networks_dir, 'passcodes.parquet'))
        angles = passcodes['angle'].values
        angle_code = np.zeros((len(angles), ))
        for i in range(len(angles)):
            angle = angles[i]
            divide = 360 // self.angle_count
            angle_code[i] = angle // divide
        passcodes[['angle']] = angle_code
        self._passcodes = passcodes

        links = pd.read_parquet(os.path.join(networks_dir, 'links.parquet'))
        links_info = links[['middir', 'snodeMid', 'enodeMid', 'roadKind',
                            'roadKindDetail', 'linkKind', 'roadLevel']]
        self._links_info = links_info

        nodes = pd.read_parquet(os.path.join(networks_dir, 'nodes.parquet'))
        self._nodes = nodes

        print('e', passable_counts, sep='\n', end='\n')
        return [passable_counts, links_info, nodes, passcodes]

    def set_preprop_networks(self, datas):
        # nsml.save 시 저장한걸 nsml.load 로 다시 불러오기 위한 인터페이스입니다.
        self._passable_counts, self._links_info, self._nodes, self._passcodes = datas

    def preprocess(self, data_path: str, label_path: str = None):
        """
        :param data_path: 통행코드 오류 판정 대상
        :param label_path: 라벨 (optional, only for train)
        """

        passables = pd.read_csv(data_path)
        print('raw',passables, sep='\n', end='\n')
        if label_path:
            labels = pd.read_csv(label_path, names=['fix'])
            passables = pd.concat([passables, labels], axis=1)

        print('passable_l', self._passable_counts, sep='\n', end='\n')
        print('passable_1', passables, sep='\n', end='\n')
        print('nodes', self._nodes, sep='\n', end='\n')

        link_all = passables[['middir', 'middirNext', 'nodeMid']]
        link_all = link_all.merge(
            self._passcodes,
            on=['middir', 'middirNext', 'nodeMid'],
            how='left'
        )[['pass']]

        link_now = passables[['middir']]
        link_now = link_now.merge(
            self._links_info[['middir', 'roadKindDetail', 'linkKind', 'roadKind', 'roadLevel']],
            on=['middir'],
            how='left')

        link_now_kind_detail = link_now['roadKindDetail'].values
        link_now_kind_road = link_now['roadKind'].values
        link_now_level_road = link_now['roadLevel'].values
        link_now_kind = link_now['linkKind'].values

        # link_now_kind_detail = ut.to_categorical(link_now_kind_detail, num_classes=10)
        # link_now_kind_road = ut.to_categorical(link_now_kind_road, num_classes=10)
        # link_now_level_road = ut.to_categorical(link_now_level_road, num_classes=11)
        # link_now_kind = ut.to_categorical(link_now_kind, num_classes=21)

        # link_no = np.concatenate([link_now_kind, link_now_kind_detail], axis=1)

        link_next = passables[['middirNext']]
        r = self._links_info[['middir', 'roadKindDetail', 'linkKind', 'roadKind', 'roadLevel']].rename({'middir': 'middirNext'}, axis=1)
        link_next = link_next.merge(
            r,
            on=['middirNext'],
            how='left')

        link_next_kind_detail = link_next['roadKindDetail'].values
        link_next_kind_road = link_next['roadKind'].values
        link_next_level_road = link_next['roadLevel'].values
        link_next_kind = link_next['linkKind'].values

        # link_next_kind_detail = ut.to_categorical(link_next_kind_detail, num_classes=10)
        # link_next_kind_road = ut.to_categorical(link_next_kind_road, num_classes=10)
        # link_next_level_road = ut.to_categorical(link_next_level_road, num_classes=11)
        # link_next_kind = ut.to_categorical(link_next_kind, num_classes=21)

        # link_ne = np.concatenate([link_next_kind, link_next_kind_detail], axis=1)

        node = passables[['nodeMid']]
        node = node.merge(
            self._nodes[['mid', 'kind']].rename({'mid': 'nodeMid'}, axis=1),
            on=['nodeMid'],
            how='left')
        node = node[['kind']].astype({'kind': int})

        node[(node['kind'] == 8)] = 7
        node_type = node['kind'].values - 1
        # node_type = ut.to_categorical(node_type, num_classes=7)

        angles = passables[['middir', 'middirNext', 'nodeMid']]
        angles = angles.merge(
            self._passcodes[['middir', 'middirNext', 'nodeMid', 'angle']],
            on=['middir', 'middirNext', 'nodeMid'],
            how='left'
        )
        angles = angles['angle']

        df = passables.merge(
            self._passable_counts,
            on=['middir', 'middirNext'],
            how='left')\
            .fillna({'count': 0})\
            .astype({'count': int})
        print(f"total dataset records count {len(df)}")

        print('after', df, link_now, link_next, sep='\n', end='\n')

        n_records = len(df)

        features = df[['count']].astype({'count': float})
        if label_path:
            def count(l, name, Y):
                d = dict()
                d_with_Y = dict()
                for i in range(len(l)):
                    type = l[i]
                    y = Y[i]
                    if type in d:
                        d[type] += 1
                    else:
                        d[type] = 1
                    if y == 1:
                        if type in d_with_Y:
                            d_with_Y[type] += 1
                        else:
                            d_with_Y[type] = 1
                print(name)
                print(d)
                print(d_with_Y)
                print()

            target = df[['fix']]
            target_v = target.values

            print(1)
            count(link_now_kind_detail, 'link_now_kind_detail', target_v)
            count(link_now_kind_road, 'link_now_kind_road', target_v)
            count(link_now_level_road, 'link_now_level_road', target_v)
            count(link_now_kind, 'link_now_kind', target_v)
            count(node_type, 'node_type', target_v)

            # print(target_v)
            # link_all_v = link_all.values
            # for i in range(len(target_v)):
            #     f = target_v[i]
            #     p = link_all_v[i]
            #     if p == 0 and f == 1:
            #         target_v[i] = 0
            #
            # print(2)
            # count(link_now_kind_detail, 'link_now_kind_detail', target_v)
            # count(link_now_kind_road, 'link_now_kind_road', target_v)
            # count(link_now_level_road, 'link_now_level_road', target_v)
            # count(link_now_kind, 'link_now_kind', target_v)
            # count(node_type, 'node_type', target_v)

            # print('features', features, sep='\n', end='\n')
            # print('target', target, sep='\n', end='\n')
            # print(features.values.shape)
            link_now_kind_road = link_now_kind_road.reshape((len(link_now_kind_road), 1))
            link_now_kind_detail = link_now_kind_detail.reshape((len(link_now_kind_detail), 1))
            link_now_level_road = link_now_level_road.reshape((len(link_now_level_road), 1))
            link_now_kind = link_now_kind.reshape((len(link_now_kind), 1))

            link_next_kind_road = link_next_kind_road.reshape((len(link_next_kind_road), 1))
            link_next_kind_detail = link_next_kind_detail.reshape((len(link_next_kind_detail), 1))
            link_next_level_road = link_next_level_road.reshape((len(link_next_level_road), 1))
            link_next_kind = link_next_kind.reshape((len(link_next_kind), 1))

            node_type = node_type.reshape((len(node_type), 1))

            # node_type = ut.to_categorical(node_type, num_classes=7)
            features_value = features['count'].values
            angles = angles.values
            # x = np.concatenate([link_now_kind_road, link_now_kind_detail, link_now_level_road, link_now_kind,
            #                     link_next_kind_road, link_next_kind_detail, link_next_level_road, link_next_kind,
            #                     features.values], axis=1)

            # -----------------------------11
            # x = np.concatenate([link_now_kind_road, link_now_level_road, link_now_kind,
            #                     link_next_kind_road, link_next_level_road, link_next_kind,
            #                     node_type, features.reshape((len(features), 1))], axis=1)
            x = np.concatenate([link_now_kind_road, link_now_level_road, link_now_kind,
                                link_next_kind_road, link_next_level_road, link_next_kind,
                                features_value.reshape((len(features_value), 1))], axis=1)
            # , angles.reshape((len(angles), 1))
            y = target_v

            # print(x.shape, y.shape)

            # sm = SMOTE(sampling_strategy='not majority')
            # sm = SVMSMOTE(sampling_strategy='not majority')
            sm = RandomOverSampler()
            # x, y = sm.fit_resample(x, y)
            # sm = SMOTENC(categorical_features=[0,1,2,3,4,5], sampling_strategy='not majority')
            x, y = sm.fit_resample(x, y)
            x = x.T

            # dataset = tf.data.Dataset.from_tensor_slices((features.values, target.values))
            # train 용으로는 셔플하고 반복하고 맘대로
            # return dataset.shuffle(n_records)
            # print(link_no, link_ne, features.values,target.values, sep='\n')
            # print(link_no.shape, link_ne.shape, features.values.shape,target.values.shape, sep='\n')

            # link_now_kind_road = x[0]
            # link_now_kind_detail = x[1]
            # link_now_level_road = x[2]
            # link_now_kind = x[3]
            # link_next_kind_road = x[4]
            # link_next_kind_detail = x[5]
            # link_next_level_road = x[6]
            # link_next_kind = x[7]

            print(x[0].shape)
            print(x[0])

            # -----------------
            link_now_kind_road = x[0]
            link_now_level_road = x[1]
            link_now_kind = x[2]
            link_next_kind_road = x[3]
            link_next_level_road = x[4]
            link_next_kind = x[5]
            # node_type = x[6]


            # link_next_kind_road = x[0]
            # link_next_level_road = x[1]
            # link_next_kind = x[2]
            # node_type = x[3]

            print(3)
            # count(link_now_kind_detail, 'link_now_kind_detail', y)
            # count(link_now_kind_road, 'link_now_kind_road', y)
            # count(link_now_level_road, 'link_now_level_road', y)
            # count(link_now_kind, 'link_now_kind', y)
            # count(node_type, 'node_type', y)

            y = np.reshape(y, (len(y), 1))
            print(x.shape, y.shape)

            # link_now_kind_detail = ut.to_categorical(link_now_kind_detail, num_classes=10)
            link_now_kind_road = ut.to_categorical(link_now_kind_road, num_classes=10)
            link_now_level_road = ut.to_categorical(link_now_level_road, num_classes=11)
            link_now_kind = ut.to_categorical(link_now_kind, num_classes=21)

            # link_next_kind_detail = ut.to_categorical(link_next_kind_detail, num_classes=10)
            link_next_kind_road = ut.to_categorical(link_next_kind_road, num_classes=10)
            link_next_level_road = ut.to_categorical(link_next_level_road, num_classes=11)
            link_next_kind = ut.to_categorical(link_next_kind, num_classes=21)

            node_type = ut.to_categorical(node_type, num_classes=7)
            # ----------------
            features = x[6].reshape((len(x[6]), 1))

            # features_value = x[3].reshape((len(x[3]), 1))
            # angles = ut.to_categorical(x[4], num_classes=self.angle_count)




            # df_counts = pd.DataFrame()
            # df_counts['count'] = features_value
            # df_counts['normalized'] = (df_counts['count'] - df_counts['count'].mean()) / df_counts['count'].std()
            # print('cccc',self._passable_counts)
            # df_counts = df_counts.drop_duplicates()
            # print(df_counts)
            # self._passable_counts = self._passable_counts.astype({'count': float})
            # self._passable_counts = self._passable_counts.merge(
            #     df_counts,
            #     on=['count'],
            #     how='left')
            #
            # features_value = features_value.reshape((len(features_value), 1))
            # angles = x[5].reshape((len(x[5]), 1))



            print(link_now_kind_road, link_now_level_road, link_now_kind,
                    link_next_kind_road, link_next_level_road, link_next_kind,
                    features_value, node_type, angles, y, sep='\n')
            return [link_now_kind_road, link_now_level_road, link_now_kind,
                    link_next_kind_road, link_next_level_road, link_next_kind,
                    features, y]
            # return [link_next_kind_road, link_next_level_road, link_next_kind,
            #         features_value, y, angles]
        else:
            # infer 용으로는 순서 및 수 유지되어야함
            # dataset = tf.data.Dataset.from_tensor_slices(features.values)
            link_now_kind_detail = ut.to_categorical(link_now_kind_detail, num_classes=10)
            link_now_kind_road = ut.to_categorical(link_now_kind_road, num_classes=10)
            link_now_level_road = ut.to_categorical(link_now_level_road, num_classes=11)
            link_now_kind = ut.to_categorical(link_now_kind, num_classes=21)

            link_next_kind_detail = ut.to_categorical(link_next_kind_detail, num_classes=10)
            link_next_kind_road = ut.to_categorical(link_next_kind_road, num_classes=10)
            link_next_level_road = ut.to_categorical(link_next_level_road, num_classes=11)
            link_next_kind = ut.to_categorical(link_next_kind, num_classes=21)

            node_type = ut.to_categorical(node_type, num_classes=7)
            angles = ut.to_categorical(angles, num_classes=self.angle_count)

            # return [link_now_kind_road, link_now_level_road, link_now_kind,
            #         link_next_kind_road, link_next_level_road, link_next_kind,
            #         features.values]
            return [link_next_kind_road, link_next_level_road, link_next_kind,
                    features.values]

    def preprocess_networks_2(self):
        return [self._passable_counts, self._links_info, self._nodes, self._passcodes]