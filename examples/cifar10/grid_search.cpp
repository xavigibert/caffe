#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <time.h>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <opencv2/core/core.hpp>

// TEMPORARY DEBUG CODE

#include <sys/stat.h>
#include <fcntl.h>

static void save_to_matlab(const char* fname, const double* ptr, int m, int k) {
  mode_t mode = S_IRUSR | S_IWUSR;
  int fd = open(fname, O_WRONLY | O_CREAT | O_TRUNC, mode);
  CHECK_GE(fd, 0);
  int s = sizeof(double);
  write(fd, (char*)&s, sizeof(int));
  write(fd, (char*)&m, sizeof(int));
  write(fd, (char*)&k, sizeof(int));
  write(fd, (const char*)ptr, m*k*sizeof(double));
  close(fd);
}

static void save_to_matlab(const char* fname, const float* ptr, int m, int k) {
  mode_t mode = S_IRUSR | S_IWUSR;
  int fd = open(fname, O_WRONLY | O_CREAT | O_TRUNC, mode);
  CHECK_GE(fd, 0);
  int s = sizeof(float);
  write(fd, (char*)&s, sizeof(int));
  write(fd, (char*)&m, sizeof(int));
  write(fd, (char*)&k, sizeof(int));
  write(fd, (const char*)ptr, m*k*sizeof(float));
  close(fd);
}

static void save_to_matlab(const char* fname, cv::Mat M) {
  if (M.type() == CV_32FC1)
    save_to_matlab(fname, M.ptr<float>(), M.rows, M.cols);
  else if (M.type() == CV_64FC1)
    save_to_matlab(fname, M.ptr<double>(), M.rows, M.cols);
  else
    LOG(FATAL) << "Invalid type";
}

// END TEMPORARY CODE

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;


DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_int32(initdict, -1,
    "The strategy used to reinitialize dictionary layers. "
    "(0 use training samples for all dicts, 1 for all except first, ...).");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  caffe::Caffe::SetDevice(FLAGS_gpu);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

void GenerateInitialDictionaries(caffe::Solver<float>* solver,
    std::vector<boost::shared_ptr<caffe::Blob<float> > >& initial_dicts,
    std::vector<int>& layer_ids) {
  // Enumerate dictionary layers in training network
  layer_ids.clear();
  caffe::Net<float>* net = solver->net().get();
  CHECK_EQ(net->layers().size(), net->bottom_vecs().size());
  for (int i = 0; i < net->layers().size(); ++i) {
    caffe::DictionaryLayer<float>* dict_layer =
        dynamic_cast<caffe::DictionaryLayer<float>*>(net->layers()[i].get());
    if (dict_layer != NULL)
      layer_ids.push_back(i);
  }
  // Allocate container for dictionaries
  initial_dicts.resize(layer_ids.size());
  for (int i = 0; i < layer_ids.size(); ++i) {
    caffe::DictionaryLayer<float>* layer =
        dynamic_cast<caffe::DictionaryLayer<float>*>(net->layers()[layer_ids[i]].get());
    LOG(INFO) << "Layer name: " << layer->layer_param().name();
    int k = layer->num_output();
    int m = layer->kernel_dim();
    std::vector<int> shape = layer->blobs()[0]->shape();
    initial_dicts[i].reset(new caffe::Blob<float>(shape));
    int k1 = shape[0];
    int m1 = shape[1] * shape[2] * shape[3];
    CHECK_EQ(k, k1);
    CHECK_EQ(m, m1);
    LOG(INFO) << "mxk = " << m << "x" << k;
  }
  // Run network to generate initial dictionary atoms
  int iter = 0;
  bool done = false;
  while (!done) {
    done = true;
    std::vector<caffe::Blob<float>*> bottom_vec;  // Dummy bottom vector
    float iter_loss;
    net->Forward(bottom_vec, &iter_loss);
    for (int i = FLAGS_initdict; i < layer_ids.size(); ++i) {
      std::vector<int> shape = initial_dicts[i]->shape();
      int k = shape[0];
      int m = shape[1] * shape[2] * shape[3];
      if (iter < k) {
        // Transfer the first sample of the minibatch to dictionary
        CHECK_EQ(net->bottom_vecs().size(), net->layers().size());
        const caffe::Blob<float>* bottom_blob = net->bottom_vecs()[layer_ids[i]][0];
        CHECK_GE(bottom_blob->count(), m);
        const float* x = bottom_blob->cpu_data();
        // Compute norm of x
        float norm_x = 0.f;
        for (int j = 0; j < m; ++j)
          norm_x += x[j]*x[j];
        norm_x = sqrt(norm_x);
        float* D = initial_dicts[i]->mutable_cpu_data();
        for (int j = 0; j < m; ++j) {
          D[iter+j*k] = x[j] / norm_x;
        }
        done = false;
      }
    }
    ++iter;
  }
}

// Initialize dictionary with training data (select 1 first sample of each batch)
void InitializeDictionaries(caffe::Solver<float>* solver,
    std::vector<boost::shared_ptr<caffe::Blob<float> > >& initial_dicts,
    std::vector<int>& layer_ids) {
  caffe::Net<float>* net = solver->net().get();
  for (int i = FLAGS_initdict; i < layer_ids.size(); ++i) {
    caffe::DictionaryLayer<float>* layer =
        dynamic_cast<caffe::DictionaryLayer<float>*>(net->layers()[layer_ids[i]].get());
    CHECK(layer);
    CHECK_EQ(layer->blobs()[0]->count(), initial_dicts[i]->count());
    CHECK_EQ(layer->blobs()[0]->count(), 4096*64);
    save_to_matlab("mat_D0.bin", layer->blobs()[0]->mutable_cpu_data(), 4096, 64);
    memcpy(layer->blobs()[0]->mutable_cpu_data(), initial_dicts[i]->cpu_data(),
         initial_dicts[i]->count() * sizeof(float));
    save_to_matlab("mat_D1.bin", layer->blobs()[0]->mutable_cpu_data(), 4096, 64);
  }
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // Precompute initialization for dictionary layers
  std::vector<boost::shared_ptr<caffe::Blob<float> > > initial_dicts;
  std::vector<int> dict_layer_ids;
  if (FLAGS_initdict >= 0) {
    shared_ptr<caffe::Solver<float> >
      solver(caffe::GetSolver<float>(solver_param));
    GenerateInitialDictionaries(solver.get(), initial_dicts, dict_layer_ids);
  }

  LOG(INFO) << "Starting Optimization";
  shared_ptr<caffe::Solver<float> >
    solver(caffe::GetSolver<float>(solver_param));

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Solve(FLAGS_snapshot);
  } else if (FLAGS_weights.size()) {
    CopyLayers(&*solver, FLAGS_weights);
    if (FLAGS_initdict >= 0) {
      LOG(INFO) << "Reinitializing dictionary using training samples";
      InitializeDictionaries(solver.get(), initial_dicts, dict_layer_ids);
    }
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight =
        caffe_net.blob_loss_weights()[caffe_net.output_blob_indices()[i]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

class TunableLogRange {
public:
  TunableLogRange(double min_val, double max_val, int steps) :
      steps_(steps) {
    CHECK_GT(steps, 0);
    log_min_ = log(min_val);
    if (steps > 1)
      log_inc_ = (log(max_val)-log(min_val))/(steps-1);
    else
      log_inc_ = 0.;
  }
  double val(int& total_idx) {
    int idx = total_idx % steps_;
    total_idx /= steps_;
    return exp(log_min_ + log_inc_*idx);
  }
  int steps() const { return steps_; }
private:
  double log_min_;
  double log_inc_;
  int steps_;
};

// Search for optimal model parameters.
int search() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  // Read solver param
  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  CHECK(solver_param.has_net());
  caffe::NetParameter net_param;
  caffe::ReadNetParamsFromTextFileOrDie(solver_param.net(), &net_param);
  solver_param.clear_net();
  solver_param.mutable_net_param()->CopyFrom(net_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // Find layers dict1 and ip2
  caffe::LayerParameter* dict1_param = NULL;
  caffe::LayerParameter* ip2_param = NULL;
  for (int j = 0; j < net_param.layer_size(); ++j) {
    if (net_param.layer(j).name() == "dict1")
      dict1_param = net_param.mutable_layer(j);
    else if (net_param.layer(j).name() == "ip2")
      ip2_param = net_param.mutable_layer(j);
  }

  // Compute number of combinations for each tunable parameter
  // param(0).lr_mult [0.01 - 1]  (5)
  // dictionary_param().rank [8 - 128] (5)
  // dictionary_param().lambda [0.001 - 0.01] (5)
  // dictionary_param().etha_rec [0.0001 - 0.01] (5)

  // Initial tuning (range 1)
//  TunableLogRange range_lr_mult(0.05, 0.2, 3);
//  TunableLogRange range_rank(16, 128, 4);
//  TunableLogRange range_num_output(32, 256, 7);
//  TunableLogRange range_lambda(0.001, 0.01, 5);
//  TunableLogRange range_etha_rec(0.00002, 0.002, 3);

  // Refined tuning (range 2)
//  TunableLogRange range_lr_mult(0.01, 1.0, 7);
//  TunableLogRange range_rank(64, 64, 1);
//  TunableLogRange range_num_output(64, 92, 2);
//  TunableLogRange range_lambda(0.002, 0.002, 1);
//  TunableLogRange range_etha_rec(0.000002, 0.00002, 3);

  TunableLogRange range_lr_mult_dict1(0.125, 2.0, 5);
  TunableLogRange range_rank(64, 64, 1);
  TunableLogRange range_num_output(64, 64, 1);
  TunableLogRange range_lambda(0.001, 0.008, 4);
  TunableLogRange range_etha_rec(0.002, 0.000002, 4);
  TunableLogRange range_wd_mult_ip2(1.0, 100.0, 7);
  TunableLogRange range_wd_solver(0.002, 0.002, 1);
  TunableLogRange range_momentum(0.85, 0.85, 1);
  TunableLogRange range_base_lr(0.0000125, 0.0001, 4);

  int num_configs = range_lr_mult_dict1.steps() * range_rank.steps()
      * range_lambda.steps() * range_etha_rec.steps() * range_num_output.steps()
      * range_wd_mult_ip2.steps() * range_wd_solver.steps()
      * range_momentum.steps() * range_base_lr.steps();

  // Reduce number of iterations
  solver_param.set_max_iter(2500);
  solver_param.set_test_interval(2500);
  solver_param.set_test_initialization(false);
  solver_param.set_lr_policy("fixed");

  // Precompute initialization for dictionary layers
  std::vector<boost::shared_ptr<caffe::Blob<float> > > initial_dicts;
  std::vector<int> dict_layer_ids;
  if (FLAGS_initdict >= 0) {
    shared_ptr<caffe::Solver<float> >
      solver(caffe::GetSolver<float>(solver_param));
    GenerateInitialDictionaries(solver.get(), initial_dicts, dict_layer_ids);
  }

  time_t timer;
  time(&timer);
  cv::RNG rng(timer);

  // Build list of configurations
  double best_loss = 1000.0f;
  for (int i = 0; i < 1000; ++i) {
    LOG(INFO) << "Starting Optimization";

    // Set parameters to new random combinations
    int conf_idx = rng.uniform(0, num_configs);
    float lr_mult_dict1 = range_lr_mult_dict1.val(conf_idx);
    int rank = (int)round(range_rank.val(conf_idx));
    float lambda = range_lambda.val(conf_idx);
    float etha_rec = range_etha_rec.val(conf_idx);
    int num_output = (int)round(range_num_output.val(conf_idx));
    if (num_output < rank) continue;
    float wd_mult_ip2 = range_wd_mult_ip2.val(conf_idx);
    float wd_solver = range_wd_solver.val(conf_idx);
    float momentum = range_momentum.val(conf_idx);
    float base_lr = range_base_lr.val(conf_idx);
    dict1_param->mutable_param(0)->set_lr_mult(lr_mult_dict1);
    dict1_param->mutable_dictionary_param()->set_rank(rank);
    dict1_param->mutable_dictionary_param()->set_lambda(lambda);
    dict1_param->mutable_dictionary_param()->set_etha_rec(etha_rec);
    dict1_param->mutable_dictionary_param()->set_num_output(num_output);
    ip2_param->mutable_param(0)->set_decay_mult(wd_mult_ip2);
    solver_param.set_weight_decay(wd_solver);
    solver_param.set_momentum(momentum);
    solver_param.set_base_lr(base_lr);

    CHECK_EQ(conf_idx, 0);
    solver_param.mutable_net_param()->CopyFrom(net_param);

    shared_ptr<caffe::Solver<float> >
      solver(caffe::GetSolver<float>(solver_param));

    if (FLAGS_snapshot.size()) {
      LOG(INFO) << "Resuming from " << FLAGS_snapshot;
      solver->Solve(FLAGS_snapshot);
    } else if (FLAGS_weights.size()) {
      CopyLayers(&*solver, FLAGS_weights);
      if (FLAGS_initdict >= 0) {
        LOG(INFO) << "Reinitializing dictionary using training samples";
        InitializeDictionaries(solver.get(), initial_dicts, dict_layer_ids);
      }
      solver->Solve();
    } else {
      solver->Solve();
    }
    LOG(INFO) << "Optimization Done.";

    // Save result
    if (solver->test_loss() < best_loss) {
      best_loss = solver->test_loss();
    }
    int fd = open("search_results.csv", O_CREAT|O_APPEND|O_WRONLY, 0600);
    char tmp_str[128];
    sprintf(tmp_str, "%.8f,%.8f,%.8f,%.8f,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
        best_loss, solver->test_loss(), solver->test_accuracy(), lr_mult_dict1,
        num_output, rank, lambda, etha_rec, wd_mult_ip2, wd_solver, momentum,
        base_lr);
    write(fd, tmp_str, strlen(tmp_str));
    close(fd);
    LOG(INFO) << "Results: " << tmp_str;
  }
  return 0;
}
RegisterBrewFunction(search);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  search          search for optimal model parameters\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
