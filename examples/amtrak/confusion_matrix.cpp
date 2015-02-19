#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <iostream>     // std::cout, std::ostream, std::hex
#include <sstream>      // std::stringbuf

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 4;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " computes the confusion matrix at layer ip2.\n"
    "Usage: confusion_matrix  pretrained_net_param"
    "  feature_extraction_proto_file  num_mini_batches  [CPU/GPU]"
    "  [DEVICE_ID=0]\n";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  Caffe::set_phase(Caffe::TEST);

  arg_pos = 0;  // the name of the executable
  string pretrained_binary_proto(argv[++arg_pos]);

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */
  string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  int num_mini_batches = atoi(argv[++arg_pos]);

  // Initialize confusion matrix
  int num_classes = 10;
  vector<int> conf_matrix(num_classes * num_classes, 0);

  LOG(ERROR)<< "Extacting Features";

  vector<Blob<float>*> input_vec;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward(input_vec);
    const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
      ->blob_by_name("ip2");
    const shared_ptr<Blob<Dtype> > label_blob = feature_extraction_net
      ->blob_by_name("label");

    int batch_size = feature_blob->num();
    int dim_features = feature_blob->count() / batch_size;
    Dtype* feature_blob_data;
    Dtype* label_blob_data;
    for (int n = 0; n < batch_size; ++n) {
      label_blob_data = label_blob->mutable_cpu_data() +
        label_blob->offset(n);
      int true_label = int(label_blob_data[0]);
      feature_blob_data = feature_blob->mutable_cpu_data() +
        feature_blob->offset(n);

      // Find maximum score
      int det_label = 0;
      Dtype max_score = feature_blob_data[0];
      for (int d = 1; d < dim_features; ++d) {
        if (feature_blob_data[d] > max_score)
        {
          max_score = feature_blob_data[d];
          det_label = d;
        }
      }
      //LOG(ERROR) << "True label = " << true_label 
      //           << ", Det Label = " << det_label;

      conf_matrix[det_label + true_label*num_classes]++;
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

  LOG(ERROR)<< "Successfully extracted the features!";
  
  // Print confusion matrix
  std::stringbuf buffer;
  std::ostream os (&buffer);
  os << "Confusion matrix:\n";
  for (int r = 0; r < num_classes; r++) {
    for (int c = 0; c < num_classes; c++) {
      os << conf_matrix[c + r * num_classes] << ",";
    }
    os << "\n";
  }
  LOG(ERROR)<< buffer.str();

  return 0;
}

