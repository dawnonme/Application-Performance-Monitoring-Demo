import csv
import random
import collections
import datetime

KPIS = {
    # System Metrics
    'sys_CPU_throughput': -5.0,
    'sys_RAM_throughput': -1.5,
    'sys_IO_throughput': 150,
    'sys_UDP_error': -2.5,
    'sys_UDP_IO_rate': 150,
    'sys_load': -5.0,
    'max_cycle_duration_time': -1.0,

    # User Metrics
    'usr_CPU_throughput': -1.5,
    'usr_RAM_throughput': -1.5,
    'usr_IO_throughput': 30,
    'usr_UDP_IO_rate': 30,
    'response_time_per_RFQ': -0.5,

    # Market Metrics
    'num_transactions': 10,

    # Symphony Metrics
    'symphony_message_rate': -100,
}

NUM_INSTANCES_TRAIN = 100000
NUM_INSTANCES_TEST = 10000
NUM_RAW = -1
OUTPUT_PATH_TRAIN = './data/train.csv'
OUTPUT_PATH_TEST = './data/test.csv'


class MockSystem:
    @staticmethod
    def initialize():
        kpis = collections.defaultdict(int)
        for k in KPIS:
            kpis[k] = KPIS[k]
        return kpis

    def component1(self, kpis):
        for k in kpis:
            rand = random.random()
            while rand == 0:
                rand = random.random()
            kpis[k] = 0.4 * rand * abs(KPIS[k])
        return kpis

    def component2(self, kpis):
        for k in kpis:
            rand = random.random()
            while rand == 0:
                rand = random.random()
            kpis[k] = 0.3 * rand * abs(KPIS[k])
        return kpis

    def component3(self, kpis):
        for k in kpis:
            rand = random.random()
            while rand == 0:
                rand = random.random()
            kpis[k] = 0.3 * rand * abs(KPIS[k])
        return kpis

    def one_pass(self):
        # component1 ==> component2 ==> component3
        kpis = MockSystem.initialize()
        kpis = self.component1(kpis)
        kpis = self.component2(kpis)
        kpis = self.component3(kpis)
        label = MockSystem.evaluate(kpis)
        output_row = list(kpis.values()) + ['', label]
        return output_row

    @staticmethod
    def evaluate(kpis):
        res = 0
        for k, v in KPIS.items():
            assert (kpis[k] != 0)
            if v < 0:
                res -= kpis[k]
            else:
                res += kpis[k]
        if res > 50:
            return 1
        elif res > 40:
            return 2
        elif res > 30:
            return 3
        return 0


system = MockSystem()


def make_data(data_type):
    if data_type not in ['train', 'test']:
        print('Invalid data type!')
        return
    num, path = (NUM_INSTANCES_TRAIN, OUTPUT_PATH_TRAIN) if data_type == 'train' \
                                                         else (NUM_INSTANCES_TEST, OUTPUT_PATH_TEST)
    print('Creating %s data...' % data_type)
    with open(path, 'w', newline='') as fout:
        csv_writer = csv.writer(fout, delimiter=',')
        first_row = ['ID'] + list(KPIS.keys()) + ['', 'Label']
        csv_writer.writerow(first_row)

        label_dict = collections.defaultdict(int)

        for i in range(num):
            row = system.one_pass()
            label_dict[row[-1]] += 1
            row = [i] + row
            csv_writer.writerow(row)
        print(label_dict)


def make_raw_data(num):
    if num <= 0:
        return
    minutes = lambda s, e: (s + datetime.timedelta(minutes=x)
                            for x in range((e - s).seconds // 60 + 1))

    today = datetime.datetime(2019, 7, 29, 15, 0, 0)
    time_stamps = [
        m for m in minutes(today, today + datetime.timedelta(minutes=num))
    ]
    print(len(time_stamps))
    for k, v in KPIS.items():
        path = './data/' + k + '.csv'
        with open(path, 'w', newline='') as fout:
            csv_writer = csv.writer(fout, delimiter=',')
            first_row = ['time', k, 'max_' + k]
            csv_writer.writerow(first_row)

            for i in range(num):
                row = [time_stamps[i], random.random() * abs(v), abs(v) * 1.5]
                csv_writer.writerow(row)


if __name__ == '__main__':
    make_data('train')
    make_data('test')
    make_raw_data(NUM_RAW)
