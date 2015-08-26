// This script converts the MNIST dataset to a lmdb (default) or
// leveldb (--backend=leveldb) format used by caffe to load data.
// Usage:
//    convert_mnist_data [FLAGS] input_image_file input_label_file
//                        output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <set>

#include "caffe/proto/caffe.pb.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;

  gflags::SetUsageMessage("This script converts the FRA/Amtrak dataset from\n"
        "one file per task/class/split to a single file per split\n"
        "Usage:\n"
        "    combine_amtrak_test_data.bin split_idx num_classes\n");

  if( argc < 3 ) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
        "examples/amtrak/convert_amtrak_data");
    return 0;
  }
  google::InitGoogleLogging(argv[0]);

  int split_idx = atoi(argv[1]);
  int num_classes = atoi(argv[2]);

  const char* data_dir = "data/amtrak";
  const char* src_image_template = "%s/fastVsBg_%d_%02d-images-idx3-ubyte";
  const char* src_label_template = "%s/fastVsBg_%d_%02d-labels-idx1-ubyte";
  const char* dst_image_template = "examples/amtrak/test%d_fastVsBg-images-idx3-ubyte";
  const char* dst_label_template = "examples/amtrak/test%d_fastVsBg-labels-idx3-ubyte";

  // Read header from first image file
  char fname[256];
  sprintf(fname, src_image_template, data_dir, split_idx, 0);

  // Create output files
  std::ifstream first_image_file(fname, std::ios::in | std::ios::binary);
  CHECK(first_image_file) << "Unable to open file " << fname;

  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items, num_labels;
  uint32_t rows, swor;
  uint32_t cols, sloc;

  first_image_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
  first_image_file.read(reinterpret_cast<char*>(&num_items), 4);
  num_items = swap_endian(num_items);
  first_image_file.read(reinterpret_cast<char*>(&swor), 4);
  rows = swap_endian(swor);
  first_image_file.read(reinterpret_cast<char*>(&sloc), 4);
  cols = swap_endian(sloc);
  LOG(INFO) << "Read image size " << rows << "x" << cols << " from file " << fname;
  first_image_file.close();
  CHECK_EQ((rows*cols)%8, 0);

  // Open output files
  sprintf(fname, dst_image_template, split_idx);
  std::ofstream dst_image_file(fname, std::ios::out | std::ios::binary);
  CHECK(dst_image_file) << "Unable to open file " << fname;
  sprintf(fname, dst_label_template, split_idx);
  std::ofstream dst_label_file(fname, std::ios::out | std::ios::binary);
  CHECK(dst_label_file) << "Unable to open file " << fname;

  // Write file headers
  magic = swap_endian(2051);
  dst_image_file.write(reinterpret_cast<const char*>(&magic), 4);
  // Save placeholder for number of items at offset 4
  num_items = 0;
  dst_image_file.write(reinterpret_cast<const char*>(&num_items), 4);
  dst_image_file.write(reinterpret_cast<const char*>(&swor), 4);
  dst_image_file.write(reinterpret_cast<const char*>(&sloc), 4);

  magic = swap_endian(2049);
  dst_label_file.write(reinterpret_cast<const char*>(&magic), 4);
  // Save placeholder for number of items at offset 4
  dst_label_file.write(reinterpret_cast<const char*>(&num_items), 4);


  char* buffer = new char[cols*rows];
  std::map<uint64_t,char> bg_set;

  uint32_t cnt_duplicate = 0;
  uint32_t cnt_saved = 0;

  for( int clIdx = 0; clIdx < num_classes; clIdx++ ) {
    sprintf(fname, src_image_template, data_dir, split_idx, clIdx);
    std::ifstream src_image_file(fname, std::ios::in | std::ios::binary);
    CHECK(src_image_file) << "Unable to open file " << fname;
    src_image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
    src_image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    src_image_file.read(reinterpret_cast<char*>(&swor), 4);
    rows = swap_endian(swor);
    src_image_file.read(reinterpret_cast<char*>(&sloc), 4);
    cols = swap_endian(sloc);
    LOG(INFO) << "Parsing " << num_items << " samples " << rows << "x" << cols << " from file " << fname;

    sprintf(fname, src_label_template, data_dir, split_idx, clIdx);
    std::ifstream src_label_file(fname, std::ios::in | std::ios::binary);
    CHECK(src_label_file) << "Unable to open file " << fname;
    src_label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2049) << "Incorrect image file magic.";
    src_label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    CHECK_EQ(num_items, num_labels);

    int cnt_fg = 0;

    for( int i = 0; i < num_items; ++i ) {
      char label = 0;
      src_label_file.read(&label, 1);
      src_image_file.read(buffer, rows*cols);
      uint64_t checksum = 0;
      uint64_t* tmp = reinterpret_cast<uint64_t*>(buffer);
      for( int j = 0; j < rows*cols/8; ++j ) {
        checksum += tmp[j];
      }
      // Translate label
      label = (label == 0)  ? -1 : clIdx;
      // Skip if duplicate
      if( bg_set.count(checksum) )
      {
        cnt_duplicate++;
        CHECK(label==-1 || bg_set[checksum]==label)
            << "A foreground sample cannot have multiple labels, "
            << fname << " offset=" << i << ", label=" << int(label)
            << ", prev_label=" << int(bg_set[checksum]);
        continue;
      }
      // Transfer sample
      bg_set[checksum] = label;
      dst_label_file.write(&label, 1);
      dst_image_file.write(buffer, rows*cols);
      cnt_saved++;
      if( label!=-1 )
        cnt_fg++;
    }
    LOG(INFO) << "Class " << clIdx << " has " << cnt_fg << " positive samples";
  }

  // Change num_items
  num_items = swap_endian(cnt_saved);
  dst_label_file.seekp(4, std::ios_base::beg);
  dst_label_file.write(reinterpret_cast<const char*>(&num_items), 4);
  dst_image_file.seekp(4, std::ios_base::beg);
  dst_image_file.write(reinterpret_cast<const char*>(&num_items), 4);

  LOG(INFO) << "Saved " << cnt_saved << " images. Found " << cnt_duplicate << " duplicates.";

  return 0;
}
