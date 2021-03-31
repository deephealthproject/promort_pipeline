#ifndef BATCHPATCHHANDLER_H
#define BATCHPATCHHANDLER_H

#include <cassandra.h>
#include <string>
#include <vector>
#include <future>
#include <utility>
#include <mutex>
#include <opencv2/core.hpp>
#include <eddl/tensor/tensor.h>
#include <ecvl/core.h>
#include <ecvl/augmentations.h>
#include "credentials.hpp"
using namespace std;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "ThreadPool.hpp"


class BatchPatchHandler{
private:
  // parameters
  int num_classes;
  static const int _max_multilabs = 32;
  bool multi_label;
  ecvl::Augmentation* aug = NULL;
  string table;
  string label_col;
  string data_col;
  string id_col;
  string username;
  string password;
  vector<string> cassandra_ips;
  string s_cassandra_ips;
  int port = 9042;
  // Cassandra connection and execution
  CassCluster* cluster = cass_cluster_new();
  CassSession* session = cass_session_new();
  const CassPrepared* prepared;
  // concurrency
  ThreadPool* pool;
  mutex mtx;
  // batch parameters
  int bs;
  bool init_batch = false;
  int chan;
  int height = -1;
  int width;
  int tot_dims;
  // current batch
  future<pair<unique_ptr<Tensor>, unique_ptr<Tensor>>> batch;
  pair<unique_ptr<Tensor>, unique_ptr<Tensor>> t_batch; // test
  unique_ptr<Tensor> t_feats;
  unique_ptr<Tensor> t_labs;
  // methods
  void connect();
  vector<char> file2buf(string filename);
  ecvl::Image buf2img(const vector<char>& buf);
  cv::Mat buf2mat(const vector<char>& buf);
  void get_img(const CassResult* result, int off);
  void get_images(const vector<string>& keys);
  vector<CassFuture*> keys2futures(const vector<string>& keys);
  void future2img(CassFuture* query_future, int off);
public:
  BatchPatchHandler(int num_classes, ecvl::Augmentation* aug, string table,
		    string label_col, string data_col, string id_col,
		    string username, string cass_pass,
		    vector<string> cassandra_ips, int thread_par=32, int port=9042);
  ~BatchPatchHandler();
  void schedule_batch(const vector<py::object>& keys);
  pair<unique_ptr<Tensor>, unique_ptr<Tensor>> load_batch(const vector<string>& keys);
  pair<shared_ptr<Tensor>, shared_ptr<Tensor>> block_get_batch();
};

#endif
