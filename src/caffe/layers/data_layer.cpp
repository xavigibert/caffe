#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    num_source_map_(0),
    cur_source_map_(-1),
    source_map_(NULL),
    reader_(param) {
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
  // clean up resources                                                                                                                                                      
  if (this->layer_param_.data_param().backend() == DataParameter_DB_LMDB_FILE ) {
    close(fd_data_);
    for (int i = 0; i < num_source_map_; ++i)
      delete source_map_[i];
    delete[] source_map_;
  }
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
  // XGS: Support for DB indexed files
  if (this->layer_param_.data_param().backend() == DataParameter_DB_LMDB_FILE) {
    num_source_map_ = this->layer_param_.data_param().source_map_size();
    CHECK(num_source_map_ > 0);
    source_map_ = new char*[num_source_map_];
    for (int i = 0; i < num_source_map_; ++i)
      source_map_[i] = strdup(this->layer_param_.data_param().source_map(i).c_str());
    cur_source_map_ = 0;
    fd_data_ = open(source_map_[cur_source_map_], O_RDONLY);
    CHECK(fd_data_ >= 0) << "Failed to open data file "
                         << source_map_[cur_source_map_] << std::endl;
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    if( datum.has_src_offset() )
    {
      CHECK_EQ(this->layer_param_.data_param().backend(), DataParameter_DB_LMDB_FILE);
      CHECK(datum.has_file_idx());
      int image_size = datum.channels() * datum.height() * datum.width();
      uint8_t* pixels = new uint8_t[image_size];
      if( datum.file_idx() != this->cur_source_map_ )
      {
        close(fd_data_);
        cur_source_map_ = datum.file_idx();
        fd_data_ = open(source_map_[cur_source_map_], O_RDONLY);
        CHECK(fd_data_ >= 0) << "Failed to open data file " 
                             << source_map_[cur_source_map_] << std::endl;
      }
      loff_t pos = lseek64(fd_data_, datum.src_offset(), SEEK_SET);
      CHECK_EQ(pos, datum.src_offset());
      ssize_t num_read = read(fd_data_, pixels, image_size);
      CHECK_EQ(num_read, image_size);
      datum.set_data(pixels, image_size);
      delete[] pixels;
    }
    else
      CHECK_NE(this->layer_param_.data_param().backend(), DataParameter_DB_LMDB_FILE);
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
