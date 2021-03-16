#include "batchpatchhandler.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>
#include <ecvl/support_eddl.h>


BatchPatchHandler::~BatchPatchHandler(){
  cass_session_free(session);
  cass_cluster_free(cluster);
}

void BatchPatchHandler::connect(){
  // FIXME: now using only first cassandra server
  cass_cluster_set_contact_points(cluster, cassandra_ips[0].c_str());
  cass_cluster_set_credentials(cluster, username.c_str(), password.c_str());
  cass_cluster_set_port(cluster, port);
  CassFuture* connect_future = cass_session_connect(session, cluster);
  CassError rc = cass_future_error_code(connect_future);
  cass_future_free(connect_future);
  if (rc == CASS_OK) {
    printf("Successfully connected!\n");
  } else {
    throw runtime_error("Error: unable to connect to Cassandra DB. ");
  }
}

BatchPatchHandler::BatchPatchHandler(int num_classes, ecvl::Augmentation* aug,
				     string table, string label_col,
				     string data_col, string id_col,
				     string username, string cass_pass,
				     vector<string> cassandra_ips,
				     int thread_par, int port) :
  num_classes(num_classes), aug(aug), table(table), label_col(label_col),
  data_col(data_col), id_col(id_col), username(username),
  password(cass_pass), cassandra_ips(cassandra_ips), port(port)
{
  // set multi-label or not
  multi_label = (num_classes>_max_multilabs) ? false : true;
  // connect to cluster and init session
  connect();
  // assemble query
  std::stringstream ss;
  ss << "SELECT " << label_col << ", " << data_col <<
    " FROM " << table << " WHERE " << id_col << "=?" << endl;
  string query = ss.str();
  // prepare statement
  CassFuture* prepare_future = cass_session_prepare(session, query.c_str());
  prepared = cass_future_get_prepared(prepare_future);
  if (prepared == NULL) {
    /* Handle error */
    cass_future_free(prepare_future);
    throw runtime_error("Error in query: " + query);
  }
  cass_future_free(prepare_future);
  // init thread pool
  pool = new ThreadPool(thread_par);
}

vector<char> BatchPatchHandler::file2buf(string filename){
  ifstream file(filename, std::ios::binary | std::ios::ate);
  streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  vector<char> buffer(size);
  if (file.read(buffer.data(), size)){
    file.close();
    return buffer;
  } else
    throw runtime_error("Error reading file");
}

cv::Mat BatchPatchHandler::buf2mat(const vector<char>& buf){
  cv::InputArray ia(buf);
  cv::Mat img = cv::imdecode(ia, cv::IMREAD_UNCHANGED);
  // img is BGR
  return(img);
} 

ecvl::Image BatchPatchHandler::buf2img(const vector<char>& buf){
  cv::Mat m = buf2mat(buf);
  ecvl::Image r = ecvl::MatToImage(m);
  if (aug){
    aug->Apply(r);
  }
  return(r);
}

void BatchPatchHandler::get_img(const CassResult* result, int off){
  // decode result
  const CassRow* row = cass_result_first_row(result);
  if (row == NULL) {
    // Handle error
    throw runtime_error("Error: query returned empty set");
  }
  const CassValue* c_lab =
    cass_row_get_column_by_name(row, label_col.c_str());
  const CassValue* c_data =
    cass_row_get_column_by_name(row, data_col.c_str());
  cass_int32_t lab;
  cass_value_get_int32(c_lab, &lab);
  const cass_byte_t* data;
  size_t sz;
  cass_value_get_bytes(c_data, &data, &sz);
  // convert buffer to image as vector<float>
  vector<char> buf(data, data+sz);
  ecvl::Image im = buf2img(buf);
  if (!im.contiguous_){
        throw runtime_error("Image data not contiguous.");
  }
  // free Cassandra result memory (values included)
  cass_result_free(result);
  ////////////////////////////////////////////////////////////////////////
  // run by just one thread
  ////////////////////////////////////////////////////////////////////////
  mtx.lock();
  // if unset, set images parameters
  if (height<0){
    height = im.Height();
    width = im.Width();
    chan = im.Channels();
    tot_dims = chan * height * width;
  }
  // allocate batch if needed
  if (init_batch){
    t_feats = unique_ptr<Tensor>(new Tensor({bs, chan, height, width}));
    t_labs = unique_ptr<Tensor>(new Tensor({bs, num_classes}));
  }
  init_batch = false;
  mtx.unlock();
  ////////////////////////////////////////////////////////////////////////

  // copy image and label to tensors
  float* p_feats =t_feats->ptr + off*tot_dims;
  uint8_t* p_im = im.data_;
  for(int i=0; i<tot_dims; ++i){
    *(p_feats++) = static_cast<float>(*(p_im++));
  }
  // alternative way: using ecvl ImageToTensor
  // Tensor* tf = t_feats.get();
  // ecvl::ImageToTensor(im, tf, off);

  // convert label 
  float* p_labs = t_labs->ptr + off*num_classes;
  if (multi_label){ // multi-label encoding
    for(int i=0; i<num_classes; ++i){
      *(p_labs++) = static_cast<float>(lab&1);
      lab>>=1;
    }
  } else { // int to one-hot
    for(int i=0; i<num_classes; ++i){
      float b = (i==lab)? 1.0 : 0.0;
      *(p_labs++) = b;
    }
  }
}

void BatchPatchHandler::future2img(CassFuture* query_future, int off){
  // get result (blocking)
  const CassResult* result = cass_future_get_result(query_future);
  if (result == NULL) {
    // Handle error
    const char* error_message;
    size_t error_message_length;
    cass_future_error_message(query_future,
			      &error_message, &error_message_length);
    cass_future_free(query_future);
    throw runtime_error("Error: unable to execute query, " +
			string(error_message));
  }
  cass_future_free(query_future);
  get_img(result, off);
}

vector<CassFuture*> BatchPatchHandler::keys2futures(const vector<string>& keys){
  vector<CassFuture*> futs;
  futs.reserve(bs);
  for(auto it=keys.begin(); it!=keys.end(); ++it){
    string id = *it;
    // prepare query
    CassStatement* statement = cass_prepared_bind(prepared);
    CassUuid cuid;
    cass_uuid_from_string(id.c_str(), &cuid);
    cass_statement_bind_uuid_by_name(statement, id_col.c_str(), cuid);
    CassFuture* query_future = cass_session_execute(session, statement);
    cass_statement_free(statement);
    futs.push_back(query_future);
  }
  return(futs);
}


void BatchPatchHandler::get_images(const vector<string>& keys){
  vector<CassFuture*> futs = keys2futures(keys);
  vector<future<void>> asys;
  asys.reserve(bs);
  for(auto i=0; i!=bs; ++i){
    // recover data and label
    // future2img(futs[i], i);
    // asys.push_back(async(launch::async, &BatchPatchHandler::future2img, this, futs[i], i));
    asys.emplace_back(pool->enqueue(&BatchPatchHandler::future2img, this, futs[i], i));
  }
  // barrier
  for(auto it=asys.begin(); it!=asys.end(); ++it){
    it->wait();
  }
}

pair<unique_ptr<Tensor>, unique_ptr<Tensor>> BatchPatchHandler::load_batch(const vector<string>& keys){
  bs = keys.size();
  init_batch = true;
  // get images and assemble batch
  get_images(keys);
  auto r = make_pair(move(t_feats), move(t_labs));
  return(r);
}

void BatchPatchHandler::schedule_batch(const vector<py::object>& keys){
  // convert uuids to strings
  vector<string> ks;
  ks.reserve(keys.size());
  for(auto it=keys.begin(); it!=keys.end(); ++it){
    string s = py::str(*it);
    ks.push_back(s);
  }
  batch = async(launch::async, &BatchPatchHandler::load_batch, this, ks);
  // t_batch = load_batch(ks);
}

pair<shared_ptr<Tensor>, shared_ptr<Tensor>> BatchPatchHandler::block_get_batch(){
  auto b = batch.get();
  auto r = make_pair(shared_ptr<Tensor>(move(b.first)),
		     shared_ptr<Tensor>(move(b.second)));
  return(r);
}

