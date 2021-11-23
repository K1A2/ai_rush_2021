import pandas as pd
import tensorflow as tf
import os

import numpy as np
from nsml import DATASET_PATH
import math
from decimal import Decimal
import numpy as np
from scipy.stats import mode

LABEL_COLUMNS = ["route_id", "plate_no", "operation_id", "station_seq", "next_duration"]

class Normalizer():
    def __init__(self):
        self.train_mean = None
        self.train_std = None
        self.train_max = None
        self.train_min = None
        self.duration_standard = None
    
    def build(self, train_data, standard):
        self.train_mean = train_data.mean()
        self.train_std = train_data.std()
        self.train_max = train_data.max()
        self.train_min = train_data.min()
        self.duration_standard = standard
    
    def normalize(self, df):
        return (df - self.train_mean) / (self.train_std)
        # return (df - self.train_min) / (self.train_max - self.train_min)

    def normalize_lable(self, df):
        return (df - self.train_mean['prev_duration']) / (self.train_std['prev_duration'])

    def get_lable(self, ls):
        std = self.train_std['prev_duration']
        mean = self.train_mean['prev_duration']
        l = []
        for i in ls:
            l.append(i * std + mean)
        return l

# CREATE YOUR OWN PREPROCESSOR FOR YOUR MODEL
class Preprocessor():
    def __init__(self):
        self.normalizer = Normalizer()
        self.train_data_path = os.path.join(DATASET_PATH, "train", "train_data", "data")
        self.train_label_path = os.path.join(DATASET_PATH, "train", "train_label")
        self.count = 2
        self.virtual_speed = 8.3

    def _load_train_dataset(self, train_data_path = None):
        train_data = pd.read_parquet(self.train_data_path) \
            .sort_values(by = ["route_id", "plate_no", "operation_id", "station_seq"], ignore_index = True)
        train_label = pd.read_csv(self.train_label_path, header = None, names = LABEL_COLUMNS) \
            .sort_values(by = ["route_id", "plate_no", "operation_id", "station_seq"], ignore_index = True)
        
        return train_data, train_label

    def time_translater(self, times):
        import datetime
        h = []
        m = []
        for i in range(len(times)):
            time = datetime.datetime.fromtimestamp(times.iat[i]).strftime("%H %M %S").split()
            time_num_h = int(time[0]) / 24
            time_num_m = int(time[1]) / 60
            # time_num_h = int(time[0])
            # time_num_m = int(time[1])
            h.append(time_num_h)
            m.append(time_num_m)
        return h, m

    def day_translater(self, days):
        for i in range(len(days)):
            day = days.iat[i]
            day /= 7
            days.iat[i] = day
        return days.copy()

    def lat_lng_to_distance(self, prev_station_lat, prev_station_lng, station_lat, station_lng, next_station_lat, next_station_lng):
        def dd_to_dms(pos):
            degree, minute, second = Decimal('0.0'), Decimal('0.0'), Decimal('0.0')
            pos = Decimal(pos)
            degree = pos // 1
            pos = (pos - degree) * 60
            minute = pos // 1
            pos = (pos - minute) * 60
            second = round(pos // Decimal('0.001') / 1000, 2)
            return degree, minute, second

        def get_distance(pos1, pos2):
            gcs = []
            for i in range(2):
                g = []
                for j in range(3):
                    g.append(pos1[i][j] - pos2[i][j])
                gcs.append(g)
            lat = gcs[0]
            lng = gcs[1]
            return math.sqrt((lat[0] * Decimal('88.9036') + lat[1] * Decimal('1.4817') + lat[2] * Decimal('0.0246')) ** 2 +
                             (lng[0] * Decimal('111.3194') + lng[1] * Decimal('1.8553') + lng[2] * Decimal('0.0309')) ** 2)

        gcs_prev = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        gcs_now = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        gcs_next = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

        gcs_prev[0], gcs_prev[1] = dd_to_dms(prev_station_lat), dd_to_dms(prev_station_lng)
        gcs_now[0], gcs_now[1] = dd_to_dms(station_lat), dd_to_dms(station_lng)
        gcs_next[0], gcs_next[1] = dd_to_dms(next_station_lat), dd_to_dms(next_station_lng)
        return get_distance(gcs_prev, gcs_now), get_distance(gcs_now, gcs_next)

    def distance_translater(self, pos):
        pos_prev = []
        pos_next = []
        for i in range(len(pos)):
            distance = self.lat_lng_to_distance(pos["prev_station_lat"].iat[i], pos["prev_station_lng"].iat[i], pos["station_lat"].iat[i],
                                                pos["station_lng"].iat[i], pos["next_station_lat"].iat[i], pos["next_station_lng"].iat[i])
            pos_prev.append(distance[0] * 1000)
            pos_next.append(distance[1] * 1000)
        return [pos_prev, pos_next]

    def time_translater_one_hot(self, times):
        h = []
        for i in range(len(times)):
            h_0 = [0] * 24
            time_num_h = int(times.iat[i])
            h_0[time_num_h] = 1
            h.append(h_0)
        return np.asarray(h, dtype='float32')

    def time_translater_one_hot_ts(self, times):
        import datetime
        h = []
        m = []
        for i in range(len(times)):
            time = datetime.datetime.fromtimestamp(times.iat[i]).strftime("%H %M %S").split()
            # h_0 = [0] * 24
            # m_0 = [0] * 60
            time_num_h = int(time[0])
            time_num_m = int(time[1])
            # h_0[time_num_h] = 1
            # m_0[time_num_m] = 1
            h.append(time_num_h)
            m.append(time_num_m)
        return h, m

    def day_translater_one_hot(self, days):
        d = []
        for i in range(len(days)):
            # d_0 = [0] * 7
            day = int(days.iat[i])
            # d_0[day - 1] = 1
            d.append(day - 1)
        return d

    def prev_duration_translater_one_hot(self, durations):
        h = []
        m = []
        s = []
        for i in range(len(durations)): #아마 s단위
            sec_0 = [0] * 60
            min_0 = [0] * 60
            hour_0 = [0] * 24
            duration = int(durations.iat[i])
            duration_sec, nam = duration % 60, duration // 60
            duration_min, duration_hour = nam % 60, nam // 60
            if duration_hour >= 24:
                duration_hour -= 24
            hour_0[duration_hour] = 1
            min_0[duration_min] = 1
            sec_0[duration_sec] = 1
            h.append(hour_0)
            m.append(min_0)
            s.append(sec_0)
        return np.asarray(h, dtype='float32'), np.asarray(m, dtype='float32'), np.asarray(s, dtype='float32')

    def insert_srandard(self, seq, prev_seq, standard):
        l = []
        for i in range(len(seq)):
            s = int(seq.iat[i])
            ps = int(prev_seq.iat[i])
            key = str(ps) + str(s)
            l.append(standard[key][4])
        return l

    def train_reshape_x(self, x, y):
        datas = x[:, 0]
        bus_drive = []
        ys = []
        now = datas[0]
        last_pos = 0
        for i in range(len(datas)):
            operation = datas[i]
            if now == operation:
                if i - last_pos >= self.count:
                    start = i - self.count + 1
                else:
                    start = last_pos
                bus = x[start:i + 1, 1:]
                dim = bus.shape
                if dim[0] != self.count:
                    bus = np.concatenate([bus, np.zeros((self.count - dim[0], dim[1]))])
                bus_drive.append(bus)
                ys.append(y[i])
            else:
                now = operation
                last_pos = i
                bus = x[i:i + 1, 1:]
                dim = bus.shape
                bus = np.concatenate([bus, np.zeros((self.count - dim[0], dim[1]))])
                bus_drive.append(bus)
                ys.append(y[i])
        x = np.asarray(bus_drive)
        y = np.asarray(ys)
        print(x.shape, y.shape)
        return x, y

    # def train_reshape_x(self, x, y):
    #     datas = x[:, 1:]
    #     result_x = []
    #     result_y = []
    #     for i in range(self.count, len(datas) + 1):
    #         result_x.append(datas[i - self.count:i,:])
    #         result_y.append(y[i - 1])
    #     x = np.asarray(result_x)
    #     y = np.asarray(result_y)
    #     return x, y

    # def test_reshape_x(self, x):
    #     result_x = []
    #     if len(x) < 10:
    #         zero = np.zeros((10 - len(x), 9), dtype='float')
    #         x = np.concatenate([zero, x[:, 1:]])
    #         result_x.append(x)
    #         x = np.asarray(result_x)
    #     else:
    #         datas = x[:, 1:]
    #         for i in range(self.count, len(datas) + 1):
    #             result_x.append(datas[i - self.count:i,:])
    #         x = np.asarray(result_x)
    #     return x

    def test_reshape_x(self, x):
        datas = x[:, 0]
        bus_drive = []
        now = datas[0]
        last_pos = 0
        for i in range(len(datas)):
            operation = datas[i]
            if now == operation:
                if i - last_pos >= self.count:
                    start = i - self.count + 1
                else:
                    start = last_pos
                bus = x[start:i + 1, 1:]
                dim = bus.shape
                if dim[0] != self.count:
                    bus = np.concatenate([bus, np.zeros((self.count - dim[0], dim[1]))])
                bus_drive.append(bus)
            else:
                now = operation
                last_pos = i
                bus = x[i:i + 1, 1:]
                dim = bus.shape
                bus = np.concatenate([bus, np.zeros((self.count - dim[0], dim[1]))])
                bus_drive.append(bus)
        x = np.asarray(bus_drive)
        print(x.shape)
        return x

    def preprocess_train_dataset(self):
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 100)
        train_data, train_label = self._load_train_dataset()

        # route_see = pd.read_csv(os.path.join(DATASET_PATH, "train", "train_data", "info", 'routes.csv'),
        #                       names=['route_id', 'route_name', 'start_point_name', 'end_point_name', 'main_bus_stop', 'turning_point_sequence'])
        # seq_see = pd.read_csv(os.path.join(DATASET_PATH, "train", "train_data", "info", 'shapes.csv'),
        #                       names=['route_id', 'station_seq', 'station_id', 'next_station_id', 'distance'])
        # ll = []
        # for i in seq_see['route_id'].values.tolist():
        #     if i not in ll:
        #         ll.append(i)
        # jj = []
        # for i in ll:
        #     jj += route_see.index[route_see['route_id'] == i].tolist()
        # kk = []
        # for i in jj:
        #     if i >= 0:
        #         kk.append(route_see['route_name'][i])
        # print(kk)
        # print(seq_see)
        # print('max:',max(seq_see['station_seq'].values.tolist()))
        # print(seq_see['station_seq'])
        # 253

        # graph = train_data[['station_seq', 'prev_station_seq', 'prev_duration']]
        # ys = dict()
        # for i in range(graph.shape[0]):
        #     seq = int(graph['station_seq'].iat[i])
        #     prev_seq = int(graph['prev_station_seq'].iat[i])
        #     du = graph['prev_duration'].iat[i]
        #     key = str(prev_seq) + str(seq)
        #     if key in ys:
        #         ys[key].append(du)
        #     else:
        #         ys[key] = [du]
        # mean = []
        # ma = []
        # mi = []
        # mid = []
        # mo = []
        # yss = dict()
        # count = []
        # for k, i in ys.items():
        #     print(k, len(i))
        #     count.append(len(i))
        #     if len(i) == 0:
        #         mean.append(0)
        #         ma.append(0)
        #         mi.append(0)
        #         mid.append(0)
        #     else:
        #         i = np.asarray(i)
        #         i.sort()
        #         yss[k] = [float(i.mean()), float(i.max()), float(i.min()),float(np.median(i)), float(mode(i)[0].tolist()[0])]
        # print(yss, min(count), sep='\n')




        train_data = train_data.drop(columns=["route_id", "plate_no", "station_id", "next_station_id",
                                            "prev_station_id"])
        train_label = train_label.drop(columns=["route_id", "plate_no", "operation_id", "station_seq"])

        train_data = train_data.astype('float')
        stations = train_data[['station_seq', 'station_lng', 'station_lat']].copy()
        prev_stations = train_data[['prev_station_seq', 'prev_station_lng', 'prev_station_lat', 'prev_station_distance', 'prev_duration']].copy()
        next_stations = train_data[['next_station_seq', 'next_station_lng', 'next_station_lat', 'next_station_distance']].copy()
        times = train_data[['ts', 'dow', 'hour']].copy()

        time_h, time_m = self.time_translater(times['ts'])
        train_data['h'] = time_h
        train_data['m'] = time_m
        train_data['dow'] = self.day_translater(times['dow'])

        # train_data['standard'] = self.insert_srandard(train_data['station_seq'].copy(), train_data['prev_station_seq'].copy(), yss)

        train_data['station_seq'] = stations['station_seq'] / 114
        train_data['prev_station_seq'] = prev_stations['prev_station_seq'] / 114
        train_data['next_station_seq'] = next_stations['next_station_seq'] / 114

        straight_distance = self.distance_translater(train_data[['prev_station_lat', 'prev_station_lng', 'station_lat',
                                                                 'station_lng', 'next_station_lat', 'next_station_lng']].copy())
        train_data['prev_straight_distance'] = straight_distance[0]
        train_data['next_straight_distance'] = straight_distance[1]

        now_station_coordinate = stations[['station_lng', 'station_lat']]
        prev_station_coordinate = prev_stations[['prev_station_lng', 'prev_station_lat']]
        duration = prev_stations['prev_duration']

        datas = train_data[['operation_id', 'h', 'm', 'dow', 'prev_straight_distance', "next_straight_distance",
                            "next_station_distance", "prev_station_distance", "prev_duration"]]

        print(datas)

        normalize_datas = datas[['h', 'm', 'dow', 'prev_straight_distance', "next_straight_distance",
                            "next_station_distance", "prev_station_distance", "prev_duration"]].copy()
        # self.normalizer.build(normalize_datas, yss)
        self.normalizer.build(normalize_datas, '')
        datas[['h', 'm', 'dow', 'prev_straight_distance', "next_straight_distance",
                            "next_station_distance", "prev_station_distance", "prev_duration"]] = self.normalizer.normalize(normalize_datas)

        print(datas)

        x = datas.values
        # x = np.reshape(x, (len(x), 1, 10))
        y = train_label.values

        x, y = self.train_reshape_x(x, y)

        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        print(x.shape, y.shape)
        return dataset
    
    def preprocess_test_data(self, test_data):
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 100)

        # test_last_data = test_data.tail(1).copy()
        # same_name = test_data['plate_no'] == test_last_data['plate_no'].iat[0]
        # same_id = test_data['route_id'] == test_last_data['route_id'].iat[0]
        # same_op = test_data['operation_id'] == test_last_data['operation_id'].iat[0]
        # less_st = test_data['station_seq'] <= test_last_data['station_seq'].iat[0]
        # test_data = test_data[same_name & same_id & same_op & less_st].sort_values(by=['station_seq'], ignore_index=True)
        test_data = test_data.drop(columns=["route_id", "plate_no", "station_id", "next_station_id","prev_station_id"])
        test_data = test_data.tail(self.count)

        test_data = test_data.astype('float')
        stations = test_data[['station_seq', 'station_lng', 'station_lat']].copy()
        prev_stations = test_data[['prev_station_seq', 'prev_station_lng', 'prev_station_lat', 'prev_station_distance', 'prev_duration']].copy()
        next_stations = test_data[['next_station_seq', 'next_station_lng', 'next_station_lat', 'next_station_distance']].copy()
        times = test_data[['ts', 'dow', 'hour']].copy()

        time_h, time_m = self.time_translater(times['ts'])
        test_data['h'] = time_h
        test_data['m'] = time_m
        test_data['dow'] = self.day_translater(times['dow'])

        # yss = self.normalizer.duration_standard
        # test_data['standard'] = self.insert_srandard(test_data['station_seq'].copy(), test_data['prev_station_seq'].copy(), yss)

        test_data['station_seq'] = stations['station_seq'] / 114
        test_data['prev_station_seq'] = prev_stations['prev_station_seq'] / 114
        test_data['next_station_seq'] = next_stations['next_station_seq'] / 114

        straight_distance = self.distance_translater(test_data[['prev_station_lat', 'prev_station_lng', 'station_lat',
                                                                 'station_lng', 'next_station_lat', 'next_station_lng']].copy())
        test_data['prev_straight_distance'] = straight_distance[0]
        test_data['next_straight_distance'] = straight_distance[1]

        now_station_coordinate = stations[['station_lng', 'station_lat']]
        prev_station_coordinate = prev_stations[['prev_station_lng', 'prev_station_lat']]
        duration = prev_stations['prev_duration']

        datas = test_data[['operation_id', 'h', 'm', 'dow', 'prev_straight_distance', "next_straight_distance",
                            "next_station_distance", "prev_station_distance", "prev_duration"]]

        normalize_datas = datas[['h', 'm', 'dow', 'prev_straight_distance', "next_straight_distance",
                            "next_station_distance", "prev_station_distance", "prev_duration"]].copy()
        datas[['h', 'm', 'dow', 'prev_straight_distance', "next_straight_distance",
                            "next_station_distance", "prev_station_distance", "prev_duration"]] = self.normalizer.normalize(normalize_datas)
        # datas = test_data[['operation_id', 'dow', 'h', 'm', 'prev_station_seq', 'prev_station_distance',
        #                     'station_seq', 'next_station_distance', 'prev_duration', 'standard']]
        #
        # normalize_datas = datas[['dow', 'h', 'm', 'prev_station_seq', 'prev_station_distance',
        #                     'station_seq', 'next_station_distance', 'prev_duration', 'standard']].copy()
        # datas[['dow', 'h', 'm', 'prev_station_seq', 'prev_station_distance',
        #                     'station_seq', 'next_station_distance', 'prev_duration', 'standard']] = self.normalizer.normalize(normalize_datas)

        x = datas.values
        # x = np.reshape(x, (len(x), 1, 10))

        x = self.test_reshape_x(x)

        dataset = tf.data.Dataset.from_tensor_slices(x)
        return dataset

    def preprocess_train_dataset_features(self):
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 100)
        train_data, train_label = self._load_train_dataset()

        train_data = train_data.drop(columns=["route_id", "plate_no", "station_id", "next_station_id",
                                              "prev_station_id"])
        train_label = train_label.drop(columns=["route_id", "plate_no", "operation_id", "station_seq"])
        train_data = train_data.astype('float')