#include <algorithm>
#include <vector>

#include "caffe/neuron_layers.hpp"

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

// END TEMPORARY CODE

namespace caffe {

template <typename Dtype>
void MklLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  MklParameter mkl_param = this->layer_param().mkl_param();
  int channels = bottom[0]->channels();
  kernel_type_ = mkl_param.kernel_type();
  CHECK_EQ(kernel_type_, MklParameter_KernelType_POLYNOMIAL)
      << "Only POLYNOMIAL kernels are supported by Mkl layer";
  neg_slope_ = mkl_param.neg_slope();
  degree_ = mkl_param.degree();
  CHECK(degree_ == 3 || degree_ == 5) 
      << "Only values 3 and 5 are supported for degree";
  channel_shared_ = mkl_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, degree_)));
    } else {
      int dim[2] = {channels, degree_};
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(dim, dim+2)));
    }
    // Fill coefficients
    const double def_coeffs[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    //const double def_coeffs[5] = {0.001, 0.0011, 0.0012, 0.0013, 0.0014};
    Dtype* coeffs = this->blobs_[0]->mutable_cpu_data();
    const int num_vects = channel_shared_ ? 1 : channels;
    for (int c = 0; c < num_vects; ++c) {
      for (int d = 0; d < degree_; ++d)
        coeffs[c+d*num_vects] = d < 5 ? Dtype(def_coeffs[d]) : Dtype(0);
    }
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), degree_)
        << "Number of parameters is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels * degree_)
        << "Number of parameters is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1) * degree_));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1) * degree_));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void MklLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void MklLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // DEBUG
//  save_to_matlab("mat_cpu_fwd_x0.bin", bottom[0]->cpu_data(), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
//  save_to_matlab("mat_cpu_fwd_x95.bin", bottom[0]->cpu_data() + bottom[0]->offset(95), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
//  save_to_matlab("mat_cpu_fwd_coeff.bin", this->blobs_[0]->cpu_data(), degree_, (channel_shared_ ? 1 : bottom[0]->channels()));
  // END DEBUG
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* coeff_data = this->blobs_[0]->cpu_data();

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  const int num_vects = channel_shared_ ? 1 : channels;
  const Dtype* w0 = coeff_data;
  const Dtype* w1 = coeff_data + num_vects;
  const Dtype* w2 = coeff_data + 2*num_vects;
  const Dtype* w3 = coeff_data + 3*num_vects;
  const Dtype* w4 = coeff_data + 4*num_vects;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    Dtype x = bottom_data[i];
    if (x > Dtype(0) || degree_ == 3) {
      const Dtype t2 = w2[c] * x;
      top_data[i] = w0[c] + (Dtype(1)+w1[c])*x + t2*t2;
    }
    else {
      const Dtype t2 = w3[c] * x;
      top_data[i] = w0[c] + (Dtype(neg_slope_)+w4[c])*x + t2*t2;
    }
  }
  // DEBUG
//  save_to_matlab("mat_cpu_fwd_y0.bin", top[0]->cpu_data(), top[0]->width(), top[0]->width() * top[0]->channels());
//  save_to_matlab("mat_cpu_fwd_y95.bin", top[0]->cpu_data() + top[0]->offset(95), top[0]->width(), top[0]->width() * top[0]->channels());
  // END DEBUG
}

template <typename Dtype>
void MklLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // DEBUG
//  save_to_matlab("mat_cpu_bwd_x0.bin", bottom_memory_.cpu_data(), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
//  save_to_matlab("mat_cpu_bwd_x95.bin", bottom_memory_.cpu_data() + bottom[0]->offset(95), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
//  save_to_matlab("mat_cpu_bwd_coeff.bin", this->blobs_[0]->cpu_data(), degree_, (channel_shared_ ? 1 : bottom[0]->channels()));
//  save_to_matlab("mat_cpu_bwd_y0.bin", top[0]->cpu_data(), top[0]->width(), top[0]->width() * top[0]->channels());
//  save_to_matlab("mat_cpu_bwd_y95.bin", top[0]->cpu_data() + top[0]->offset(95), top[0]->width(), top[0]->width() * top[0]->channels());
//  save_to_matlab("mat_cpu_bwd_dy0.bin", top[0]->cpu_diff(), top[0]->width(), top[0]->width() * top[0]->channels());
//  save_to_matlab("mat_cpu_bwd_dy95.bin", top[0]->cpu_diff() + top[0]->offset(95), top[0]->width(), top[0]->width() * top[0]->channels());
  // END DEBUG
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* coeff_data = this->blobs_[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  const int num_vects = channel_shared_ ? 1 : channels;
  const Dtype* w0 = coeff_data;
  const Dtype* w1 = coeff_data + num_vects;
  const Dtype* w2 = coeff_data + 2*num_vects;
  const Dtype* w3 = coeff_data + 3*num_vects;
  const Dtype* w4 = coeff_data + 4*num_vects;

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* coeff_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      Dtype* d0 = coeff_diff;
      Dtype* d1 = coeff_diff + num_vects;
      Dtype* d2 = coeff_diff + 2*num_vects;
      Dtype* d3 = coeff_diff + 3*num_vects;
      Dtype* d4 = coeff_diff + 4*num_vects;
      Dtype x = bottom_data[i];
      Dtype dy = top_diff[i];
      d0[c] += dy;
      if (x > Dtype(0) || degree_ == 3) {
        d1[c] += dy * x;
        d2[c] += dy * Dtype(2) * x * x * w2[c];
      }
      else
      {
        d3[c] += dy * Dtype(2) * x * x * w3[c];
        d4[c] += dy * x;
      }
      // DEBUG
//      if ((i+1) % top[0]->offset(1) == 0) {
//        int n = (i+1) / top[0]->offset(1)- 1;
//        if (n<10) {
//          char s[64];
//          sprintf(s, "mat_cpu_bwd_dcoeff%d.bin", n);
//          save_to_matlab(s, this->blobs_[0]->cpu_diff(), degree_, (channel_shared_ ? 1 : bottom[0]->channels()));
//        }
//      }
      // END DEBUG
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      Dtype x = bottom_data[i];
      if (x > Dtype(0) || degree_ == 3) {
        bottom_diff[i] = top_diff[i] * (Dtype(1) + w1[c]
            + Dtype(2) * w2[c] * w2[c] * x);
      }
      else {
        bottom_diff[i] = top_diff[i] * (neg_slope_ + w4[c]
            + Dtype(2) * w3[c] * w3[c] * x);
      }
    }
  }
  // DEBUG
//  save_to_matlab("mat_cpu_bwd_dcoeff.bin", this->blobs_[0]->cpu_diff(), degree_, (channel_shared_ ? 1 : bottom[0]->channels()));
//  save_to_matlab("mat_cpu_bwd_dx0.bin", bottom[0]->cpu_diff(), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
//  save_to_matlab("mat_cpu_bwd_dx95.bin", bottom[0]->cpu_diff() + bottom[0]->offset(95), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
//  LOG(FATAL) << "FIX ME!";
  // END DEBUG
}


#ifdef CPU_ONLY
STUB_GPU(MklLayer);
#endif

INSTANTIATE_CLASS(MklLayer);
REGISTER_LAYER_CLASS(Mkl);

}  // namespace caffe
