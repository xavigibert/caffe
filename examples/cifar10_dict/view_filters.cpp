#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <fstream>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using namespace std;

int main(int argc, char** argv)
{
  if( argc < 3 )
    return 1;

  string param_file = argv[1];
  string layer_name = argv[2];
  NetParameter param;
  CHECK(ReadProtoFromBinaryFile(param_file, &param))
    << "Failed to parse parameters NetParameter file: " << param_file;

  for( int i = 0; i < param.layer_size(); i++ )
  {
    const LayerParameter& layer = param.layer(i);
    if( layer.name() == layer_name )
    {
      for( int j = 0; j < layer.blobs_size(); j++ )
      {
        char fname[256];
        sprintf(fname, "%s_blob%d.bin", argv[2], j);
        std::ofstream blob_file(fname, std::ios::out | std::ios::binary);
        const BlobProto& blob = layer.blobs(j);
        cout << fname << std::endl;
        cout << "  [";
        for( int k = 0; k < blob.shape().dim_size(); k++)
        {
          cout << blob.shape().dim(k);
          if( k < blob.shape().dim_size()-1 )
            cout << ",";
        }
        cout << "]" << std::endl;
        for( int k = 0; k < blob.data_size(); k++ )
        {
          float data = blob.data(k);
          blob_file.write((char*)&data, sizeof(float));
        }
      }
    }
  }

  return 0;
}
