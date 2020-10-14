import io
import itertools
import numpy as np
import random
import pickle
import PIL
import PIL.Image
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import time
import threading
from tqdm import trange, tqdm

# pip3 install cassandra-driver
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra.cluster import ExecutionProfile

# Handler for large query results
class PagedResultHandler():
    def __init__(self, future):
        self.all_rows = []
        self.error = None
        self.finished_event = threading.Event()
        self.future = future
        self.future.add_callbacks(
            callback=self.handle_page,
            errback=self.handle_error)
    def handle_page(self, rows):
        self.all_rows += rows
        if self.future.has_more_pages:
            self.future.start_fetching_next_page()
        else:
            self.finished_event.set()
    def handle_error(self, exc):
        self.error = exc
        self.finished_event.set()


## ecvl reader for Cassandra
class CassandraDataset():
    def __init__(self, auth_prov, cassandra_ips,
                 table, batch_size=8, num_classes=2,
                 max_patches=None, split_ratios=[7,1,2], balance=None,
                 augs=[], seed=None):
        """Create ECVL Dataset from Cassandra DB

        :param auth_prov: Authenticator for Cassandra
        :param cassandra_ips: List of Cassandra ip's
        :param table: Table from which data are read
        :param batch_size: Batch size
        :param num_classes: Number of classes
        :param max_patches: Number of patches to be read
        :param split_ratios: Ratio among training, validation and test
        :param balance: Ratio among the different classes (defaults to [1, 1, ...])
        :param augs: Data augmentations to be used
        :param seed: Seed for random generators
        :returns: 
        :rtype: 

        """
        # seed random generators
        random.seed(seed)
        np.random.seed(seed)
        ## cassandra parameters
        prof_dict = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory = cassandra.query.dict_factory)
        prof_tuple = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory = cassandra.query.tuple_factory)
        profs = {'dict': prof_dict, 'tuple': prof_tuple}
        self.cluster = Cluster(cassandra_ips,
                               execution_profiles=profs,
                               protocol_version=4,
                               auth_provider=auth_prov)
        self.cluster.connect_timeout = 10 #seconds
        self.sess = self.cluster.connect()
        self.table = table
        query = f"SELECT label, data FROM {self.table} WHERE \
        sample_name=? AND label=? AND sample_rep=? AND x=? AND y=?"
        self.prep = self.sess.prepare(query)
        ## internal parameters
        self.row_keys = None
        self.augs = augs
        self.num_classes = num_classes
        self.labs = [2**i for i in range(self.num_classes)]
        self.max_patches = max_patches
        self.batch_size = batch_size
        self.current_split = 0
        self.current_index = []
        self.raw_batch = []
        self.num_batches = []
        self.locks = None
        self.sample_names = None
        self.n = None
        self._stats = None
        self._rows = None
        self._init_stats()
        self.split = None
        self.num_splits = len(split_ratios)
        self.balance = None
        self.split_ratios = np.array(split_ratios)
        self.split_setup(seed=seed)
    def __del__(self):
        self.cluster.shutdown()
    def _init_stats(self):
        self.sample_names = self.sess.execute(f'SELECT DISTINCT sample_name, label \
        FROM {self.table} ;', execution_profile='tuple')
        self.sample_names = {name[0] for name in self.sample_names}
        self.sample_names = list(self.sample_names)
        random.shuffle(self.sample_names)
        query = f"SELECT sample_name, label, sample_rep, x, y \
        FROM {self.table} WHERE sample_name=? and label=? ;"
        prep = self.sess.prepare(query)
        self._rows = {}
        for sn in self.sample_names:
            self._rows[sn]={}
        pbar = tqdm(desc='Reading list of patches',
                    total=self.num_classes*len(self.sample_names))
        conc_lev = 16 # concurrent async queries to cassandra
        for l in self.labs:
            loc_names = self.sample_names.copy()
            futures = []
            # while there are samples to be processed
            while (len(loc_names)>0 or len(futures)>0):
                # fill the pool with samples
                while (len(loc_names)>0 and len(futures)<conc_lev):
                    sn = loc_names.pop()
                    res = self.sess.execute_async(prep, (sn, l),
                                                  execution_profile='dict')
                    futures.append((sn, PagedResultHandler(res)))
                # check if a query slot can be freed
                for future in futures:
                    sn, handler = future
                    if(handler.finished_event.is_set()):
                        if handler.error:
                            raise handler.error
                        res = handler.all_rows
                        random.shuffle(res)
                        self._rows[sn][l] = res
                        futures.remove(future)
                        pbar.update(1)
                # sleep 10 ms
                time.sleep(.01)
        pbar.close()
        counters = [[len(s[i]) for i in self.labs] for s in self._rows.values()]
        self._stats = np.array(counters)
    def _update_params(self, max_patches=None, split_ratios=None, augs=None,
                       balance=None, batch_size=None):
        # update batch_size
        if (batch_size is not None):
            self.set_batchsize(batch_size)
        # update augmentations
        if (augs is not None):
            self.augs=augs
        # update number of patches
        if (max_patches is not None):
            self.max_patches=max_patches
        if (self.max_patches is None): # if None use all patches
            self.max_patches=int(1e18) # i.e., large enough number
        # update and normalize split ratios
        if (split_ratios is not None):
            self.split_ratios = np.array(split_ratios)
        self.split_ratios = self.split_ratios/self.split_ratios.sum()
        self.num_splits=len(self.split_ratios)
        # create a lock per split
        self.locks=[threading.Lock() for i in range(self.num_splits)]
        # update and normalize balance
        if (balance is not None):
            self.balance = np.array(balance)
        if (self.balance is None): # default to uniform
            self.balance = np.ones(self.num_classes)
        assert(self.balance.shape[0]==self.num_classes)
        self.balance = self.balance/self.balance.sum()
    def split_setup(self, max_patches=None, split_ratios=None, augs=None,
                    balance=None, batch_size=None, seed=None):
        """(Re)Insert the patches in the splits, according to split and class ratios

        :param max_patches: Number of patches to be read. If None use the current value.
        :param split_ratios: Ratio among training, validation and test. If None use the current value.
        :param augs: Data augmentations to be used. If None use the current ones.
        :param balance: Ratio among the different classes. If None use the current value.
        :param batch_size: Batch size. If None use the current value.
        :param seed: Seed for random generators
        :returns: 
        :rtype: 

        """
        # seed random generators
        random.seed(seed)
        np.random.seed(seed)
        # update dataset parameters
        self._update_params(max_patches=max_patches, split_ratios=split_ratios,
                            augs=augs, balance=balance, batch_size=batch_size)
        ## partitioning sample names in training, validation and test
        # get a copy of the list of rows
        print('Copying data structures...')
        l_rows = pickle.loads(pickle.dumps(self._rows))
        tots = self._stats.sum(axis=0)
        stop_at = self.split_ratios.reshape((-1,1)) * tots
        stop_at[0] = tots # put all scraps in training
        # insert patches into bags until they're full
        bags = [[]]*self.num_splits # bag-0, bag-1, etc.
        cows = np.zeros([self.num_splits, self.num_classes])
        curr = 0 # current bag
        for (i, p_num) in enumerate(self._stats):
            # check if current bag can hold the sample set, if not increment bag
            # note: bag-0 (training) can always contain a sample set
            while ((cows[curr]+p_num)>stop_at[curr]).any():
                curr += 1; curr %= self.num_splits
            bags[curr] += [i]
            cows[curr] += p_num
            curr += 1; curr %= self.num_splits
        ## insert into the splits, taking into account the target class balance
        borders = self.max_patches * self.split_ratios.cumsum()
        borders = borders.round().astype(int)
        borders = np.pad(borders, [1,0])
        max_split = [borders[i+1]-borders[i] for i in range(self.num_splits)]
        def enough_rows(sp, sample_num, lab):
            bag = bags[sp]
            sample_name = self.sample_names[bag[sample_num]]
            num = len(l_rows[sample_name][lab])
            return (num>0)
        def find_row(sp, sample_num, lab):
            max_sample = len(bags[sp])
            cur_sample = sample_num
            inc = 0
            while (inc<max_sample and not enough_rows(sp, cur_sample, lab)):
                   cur_sample +=1; cur_sample %= max_sample
                   inc += 1
            if (inc>=max_sample): # row not found
                   cur_sample = -1 
            return cur_sample
        sp_rows = []
        pbar = tqdm(desc='Choosing patches', total=self.max_patches)
        for sp in range(self.num_splits): # for each split
            sp_rows.append([])
            max_sample = len(bags[sp])
            tmp = max_split[sp] * self.balance.cumsum()
            tmp = tmp.round().astype(int)
            tmp = np.pad(tmp, [1,0])
            max_class = [tmp[i+1]-tmp[i] for i in range(tmp.shape[0]-1)]
            for cl in range(self.num_classes):
                cur_sample = 0
                tot = 0
                while (tot<max_class[cl]):
                    if (not enough_rows(sp, cur_sample, self.labs[cl])):
                        cur_sample = find_row(sp, cur_sample, self.labs[cl])
                    if (cur_sample<0): # not found, skip to next class
                        break
                    bag = bags[sp]
                    sample_name = self.sample_names[bag[cur_sample]]
                    row = l_rows[sample_name][self.labs[cl]].pop(0)
                    sp_rows[sp].append(row)
                    tot+=1
                    cur_sample +=1; cur_sample %= max_sample
                    pbar.update(1)
        pbar.close()
        # build common sample list
        self.split = []
        self.row_keys = []
        start = 0
        for sp in range(self.num_splits):
            self.split.append(None)
            sz = len(sp_rows[sp])
            random.shuffle(sp_rows[sp])
            self.row_keys += sp_rows[sp]
            self.split[sp] = np.arange(start, start+sz)
            start += sz
        self.row_keys = np.array(self.row_keys)
        self.n = self.row_keys.shape[0] # set size
        self._reset_indexes()
    def _reset_indexes(self):
        self.current_index = []
        self.raw_batch = []
        self.num_batches = []
        for cs in range(self.num_splits):
            with self.locks[cs]:
                self.current_index.append(0)
                self.raw_batch.append(None)
                self.num_batches.append(len(self.split[cs]+self.batch_size-1)
                                        // self.batch_size)
                # preload batches
                self._preload_raw_batch(cs)
    def set_batchsize(self, bs):
        self.batch_size = bs
        self._reset_indexes()
    def rewind_splits(self, chosen_split=None, shuffle=False):
        """Rewind/reshuffle rows in chosen split and reset its current index

        :param chosen_split: Split to be rewinded. If None rewind all the splits.
        :param shuffle: Apply random permutation (def: False)
        :returns: 
        :rtype: 

        """
        if (chosen_split is None):
            splits = range(self.num_splits)
        else:
            splits = [chosen_split]
        for cs in splits:
            with self.locks[cs]:
                if (shuffle):
                    self.split[cs] = np.random.permutation(self.split[cs])
                # reset index and preload batch
                self.current_index[cs] = 0
                self._preload_raw_batch(cs)
    def _get_img(self, cs):
        def ret(item):
            # read label
            lab = item['label'] # read 32-bit int
            # convert to bits
            lab = np.unpackbits(np.array([lab], dtype=">i4").view(np.uint8))
            lab = lab[-self.num_classes:] # take last num_classes bits
            # read image
            raw_img = item['data']
            in_stream = io.BytesIO(raw_img)
            img = PIL.Image.open(in_stream) # xyc, RGB
            arr = np.array(img) # yxc, RGB
            arr = arr[..., ::-1] # yxc, BGR
            # apply augmentations on eimg and then convert back to array
            if (len(self.augs)>cs and self.augs[cs] is not None):
                eimg = ecvl.Image.fromarray(arr, "yxc", ecvl.ColorType.BGR)
                self.augs[cs].Apply(eimg)
                arr = np.array(eimg) #yxc, BGR
            return (arr, lab)
        return(ret)
    def _save_futures(self, rows, cs):
        # get whole batch asynchronously
        keys_ = [row.values() for row in rows]
        futures = []
        for keys in keys_:
            futures.append(self.sess.execute_async(self.prep, keys,
                                                   execution_profile='dict'))
        self.raw_batch[cs] = futures
    def _get_raw_batch(self, cs):
        items = []
        futures = self.raw_batch[cs]
        for future in futures:
            items.append(future.result().one())
        return items
    def _compute_batch(self, cs):
        items = self._get_raw_batch(cs)
        all = map(self._get_img(cs), items)
        feats, labels = zip(*all) # transpose
        feats = np.array(feats)
        labels = np.array(labels)
        # transpose: byxc, BGR -> bcyx BGR
        bb = (Tensor(feats.transpose(0,3,1,2)), Tensor(labels))
        return bb
    def _preload_raw_batch(self, cs):
        if (self.current_index[cs]>=self.split[cs].shape[0]):
            return # end of split, stop prealoding
        idx_ar = self.split[cs][self.current_index[cs] :
                                self.current_index[cs] + self.batch_size]
        self.current_index[cs] += idx_ar.size #increment index
        bb = self.row_keys[idx_ar]
        self._save_futures(bb, cs)
    def load_batch(self, split=None):
        """Read a batch from Cassandra DB.

        :param split: Split to read from (default to current_split)
        :returns: (x,y) with X tensor of features and y tensor of labels
        :rtype: 

        """
        if (split is None):
            cs = self.current_split
        else:
            cs = split
        with self.locks[cs]:
            # compute batch from preloaded raw data
            batch = self._compute_batch(cs)
            # start preloading the next batch
            self._preload_raw_batch(cs)
        return(batch)
