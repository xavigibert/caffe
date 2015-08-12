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

#include "caffe/proto/caffe.pb.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

DEFINE_string(backend, "lmdb", "The backend for storing the result");

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void convert_dataset(int num_files, char** image_filename,
        char** label_filename, int* max_samples,
        const char* db_path, const string& db_backend) {
  // Open files
  std::ifstream image_file(image_filename[0], std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << image_filename[0];
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

  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;
  // leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;

  // Open db
  if (db_backend == "leveldb") {  // leveldb
    LOG(INFO) << "Opening leveldb " << db_path;
    leveldb::Status status = leveldb::DB::Open(
        options, db_path, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << db_path
        << ". Is it already existing?";
    batch = new leveldb::WriteBatch();
  } else if (db_backend == "lmdb") {  // lmdb
    LOG(INFO) << "Opening lmdb " << db_path;
    CHECK_EQ(mkdir(db_path, 0744), 0)
        << "mkdir " << db_path << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed. Does the lmdb already exist? ";
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }

  // Calculate number of subsplits
  int max_max_samples = 0;
  for(int i = 0; i < num_files; i++) {
    if(max_samples[i] > max_max_samples)
      max_max_samples = max_samples[i];
  }
  int num_subsplits = (max_max_samples+999) / 1000;
  if( num_subsplits < 1 )
    num_subsplits = 1;

  // Storing to db
  char label;
  //char* pixels = new char[rows * cols];
  int count = 0;
  const int kMaxKeyLength = 10;
  char key_cstr[kMaxKeyLength];
  string value;

  Datum datum;
  datum.set_channels(1);
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int sub_idx = 0; sub_idx < num_subsplits; sub_idx++) {
  for (int file_idx = 0; file_idx < num_files; file_idx++) {
    // Open labels file
    std::ifstream label_file(label_filename[file_idx], std::ios::in | std::ios::binary);
    CHECK(label_file) << "Unable to open file " << label_filename[file_idx];
    uint32_t num_labels;
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    //CHECK_EQ(num_items, num_labels);
    if( max_samples[file_idx] < 0 )
      max_samples[file_idx] = num_labels;
    CHECK(max_samples[file_idx] <= num_labels);

    LOG(INFO) << "File " << file_idx << ": " << max_samples[file_idx] << " of "
              << num_labels << " items.";

    int start_id = sub_idx * max_samples[file_idx] / num_subsplits;
    label_file.seekg(start_id,std::ios_base::cur);
    int end_id = (sub_idx+1) * max_samples[file_idx] / num_subsplits;
    for (int item_id = start_id; item_id < end_id; ++item_id) {
      uint64_t src_offset = 16 + item_id * rows * cols;
      label_file.read(&label, 1);
      datum.set_file_idx(file_idx);
      datum.set_src_offset(src_offset);
      datum.set_label(label);
      snprintf(key_cstr, kMaxKeyLength, "%08d", count);
      datum.SerializeToString(&value);
      string keystr(key_cstr);

      // Put in db
      if (db_backend == "leveldb") {  // leveldb
        batch->Put(keystr, value);
      } else if (db_backend == "lmdb") {  // lmdb
        mdb_data.mv_size = value.size();
        mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
        mdb_key.mv_size = keystr.size();
        mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
        CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
          << "mdb_put failed";
      } else {
        LOG(FATAL) << "Unknown db backend " << db_backend;
      }

      if (++count % 1000 == 0) {
        // Commit txn
        if (db_backend == "leveldb") {  // leveldb
          db->Write(leveldb::WriteOptions(), batch);
          delete batch;
          batch = new leveldb::WriteBatch();
        } else if (db_backend == "lmdb") {  // lmdb
          CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
            << "mdb_txn_commit failed";
          CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        } else {
          LOG(FATAL) << "Unknown db backend " << db_backend;
        }
      }
    }
  }
  }
  // write the last batch
  if (count % 1000 != 0) {
    if (db_backend == "leveldb") {  // leveldb
      db->Write(leveldb::WriteOptions(), batch);
      delete batch;
      delete db;
    } else if (db_backend == "lmdb") {  // lmdb
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
      mdb_close(mdb_env, mdb_dbi);
      mdb_env_close(mdb_env);
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }
    LOG(ERROR) << "Processed " << count << " files.";
  }
}

int main(int argc, char** argv) {
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("This script converts the FRA/Amtrak dataset to\n"
        "the lmdb/leveldb format used by Caffe to load data.\n"
        "Usage:\n"
        "    convert_amtrak_data [FLAGS] num_files input_image_files input_label_files"
        "num_samples_per_file output_db_file\n"
        "The Amtrak dataset could be downloaded at\n"
        "    http://TBD\n"
        "You should gunzip them after downloading,"
        "or directly use data/amtrak/get_amtrak.sh\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const string& db_backend = FLAGS_backend;

  if( argc < 2 ) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
        "examples/amtrak/convert_amtrak_data");
    return 0;
  }

  int num_files = atoi(argv[1]);

  if (argc != 3+3*num_files) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
        "examples/amtrak/convert_amtrak_data");
  } else {
    google::InitGoogleLogging(argv[0]);
    char** input_image_files = &argv[2];
    char** input_label_files = &argv[2+num_files];
    char** str_samples_per_file = &argv[2+2*num_files];
    int* samples_per_file = new int[num_files];
    for( int i = 0; i < num_files; i++ ) {
      if( str_samples_per_file[i][0] == 'A' )
        samples_per_file[i] = -1;
      else
        samples_per_file[i] = atoi(str_samples_per_file[i]);
    }
    char* db_path = argv[2+3*num_files];
    
    convert_dataset(num_files, input_image_files, input_label_files, \
                    samples_per_file, db_path, db_backend);
    
    delete[] samples_per_file;
  }
  return 0;
}
