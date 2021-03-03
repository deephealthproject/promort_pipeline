import getpass
from pyeddl import eddl
from pyecvl import ecvl
from BPH import BatchPatchHandler as BPHand
from uuid import UUID
# Read Cassandra password
try: 
    from private_data import cass_pass
except ImportError:
    cass_pass = getpass('Insert Cassandra password: ')

h = BPHand(num_classes=2, aug=None, table="promort.data",
           label_col="label", data_col="data", id_col="patch_id",
           username="user", cass_pass=cass_pass,
           cassandra_ips=["cassandra_db"])

h.schedule_batch([UUID('92cd1dee-78da-4310-a35f-193a6d49809e'),
                  UUID('e4dd636b-cdbd-4ae2-a4d0-1e428da1a002')])
x,y = h.block_get_batch()
print(x.getShape())
print(y.getShape())

