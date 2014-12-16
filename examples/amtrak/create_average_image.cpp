// Computes the average image pixel intensity for the whole training
// set and stores the value in binaryproto file

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/proto/caffe.pb.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

void WriteProtoToBinaryFile(const google::protobuf::Message& proto, const char* filename) {
  std::fstream output(filename, std::ios::out | std::ios::trunc | std::ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void compute_mean(char* image_filename,
                  char* average_filename) {
  // Open file
  std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << image_filename;
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t rows;
  uint32_t cols;

  image_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
  image_file.read(reinterpret_cast<char*>(&num_items), 4);
  num_items = swap_endian(num_items);
  image_file.read(reinterpret_cast<char*>(&rows), 4);
  rows = swap_endian(rows);
  image_file.read(reinterpret_cast<char*>(&cols), 4);
  cols = swap_endian(cols);

  unsigned char* pixels = new unsigned char[rows * cols];
  double sum_all = 0;

  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int item_id = 0; item_id < num_items; ++item_id) {
    image_file.read((char*)pixels, rows * cols);
    double sum = 0;
    for( int i = 0; i < rows * cols; i++ )
      sum += (double)pixels[i];
    sum_all += sum / (rows*cols);
  }

  sum_all /= num_items;
  LOG(INFO) << "Average value = " << sum_all << std::endl;

  // Save average image as protobuf
  caffe::BlobProto sum_blob;
  sum_blob.set_num(1);
  sum_blob.set_channels(1);
  sum_blob.set_height(rows);
  sum_blob.set_width(cols);
  for( int i = 0; i < rows * cols; i++ )
    sum_blob.add_data(float(sum_all));
  WriteProtoToBinaryFile(sum_blob, average_filename);
}

int main(int argc, char** argv) {
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("This script computes the average pixel value in the FRA/Amtrak dataset.\n"
        "Usage:\n"
        "    create_average_image input_image_file output_protoblob\n");

  if( argc < 3 ) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
        "examples/amtrak/create_average_image");
    return 0;
  } else {
    google::InitGoogleLogging(argv[0]);
    compute_mean(argv[1], argv[2]);
  }
  return 0;
}
