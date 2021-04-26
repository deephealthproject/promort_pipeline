from cassandra.auth import PlainTextAuthProvider
from cassandra_dataset import CassandraDataset

class EnvLoader():
    def __init__(self, inet_pass, num):
        self.inet_pass = inet_pass
        self.num = num
        self.cd = None
    def start(self):
        if (self.cd is None):
            print('####### Starting Cassandra dataset')
            ap = PlainTextAuthProvider(username='inet',
                                       password=self.inet_pass)
            cd = CassandraDataset(ap, ['cassandra_db'])
            cd.load_rows('inet_256_rows.pckl')
            cd.init_datatable(table='imagenet.data_256')
            cd.split_setup(batch_size=32, split_ratios=[1]*self.num,
                           max_patches=1300000)
            self.cd = cd
