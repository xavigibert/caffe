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

// CUDA kernel for forward
template <typename Dtype>
__global__ void MklForward(const int n, const int channels, const int dim,
    const Dtype* bottom_data, Dtype* top_data, const Dtype* coeff_data,
    const int div_factor, const int degree, Dtype neg_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    const int num_vects = div_factor == channels ? 1 : channels;
    const Dtype* w0 = coeff_data;
    const Dtype* w1 = coeff_data + num_vects;
    const Dtype* w2 = coeff_data + 2*num_vects;
    const Dtype* w3 = coeff_data + 3*num_vects;
    const Dtype* w4 = coeff_data + 4*num_vects;
    Dtype x = bottom_data[index];
    bool pos = x > Dtype(0) || degree == 3;
    const Dtype t2 = pos ? w2[c] * x : w3[c] * x;
    top_data[index] = w0[c] + t2*t2 +
        (pos ? (Dtype(1)+w1[c])*x : (neg_slope+w4[c])*x);
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void MklBackward(const int n, const int channels, const int dim,
    const Dtype* top_diff, const Dtype* bottom_data, Dtype* bottom_diff,
    const Dtype* coeff_data, const int div_factor, const int degree,
    Dtype neg_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    const int c = (index / dim) % channels / div_factor;
    const int num_vects = div_factor == channels ? 1 : channels;
    const Dtype* w1 = coeff_data + num_vects;
    const Dtype* w2 = coeff_data + 2*num_vects;
    const Dtype* w3 = coeff_data + 3*num_vects;
    const Dtype* w4 = coeff_data + 4*num_vects;
    const Dtype x = bottom_data[index];
    bool pos = x > Dtype(0) || degree == 3;
    bottom_diff[index] = pos ?
        top_diff[index] * (Dtype(1) + w1[c] + Dtype(2) * w2[c] * w2[c] * x) :
        top_diff[index] * (neg_slope + w4[c] + Dtype(2) * w3[c] * w3[c] * x);
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void MklParamBackward(const int n, const int channels, const int dim,
    const Dtype* top_diff, const Dtype* bottom_data, Dtype* out_diff,
    const Dtype* coeff_data, const int div_factor, const int degree) {
  CUDA_KERNEL_LOOP(index, n) {
    const int c = (index / dim) % channels / div_factor;
    const int num_vects = div_factor == channels ? 1 : channels;
    const Dtype* w2 = coeff_data + 2*num_vects;
    const Dtype* w3 = coeff_data + 3*num_vects;
    Dtype* d0 = out_diff;
    Dtype* d1 = out_diff + n;
    Dtype* d2 = out_diff + 2*n;
    Dtype* d3 = out_diff + 3*n;
    Dtype* d4 = out_diff + 4*n;
    const Dtype x = bottom_data[index];
    const Dtype dy = top_diff[index];
    d0[index] = dy;
    bool pos = x > Dtype(0) || degree == 3;
    d1[index] = pos ? dy * x : Dtype(0);
    d2[index] = pos ? dy * Dtype(2) * x * x * w2[c] : Dtype(0);
    if (degree > 3) {
      d3[index] = pos ? Dtype(0) : dy * Dtype(2) * x * x * w3[c];
      d4[index] = pos ? Dtype(0) : dy * x;
    }
  }
}

template <typename Dtype>
void MklLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // DEBUG
//  save_to_matlab("mat_gpu_fwd_x0.bin", bottom[0]->cpu_data(), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
//  save_to_matlab("mat_gpu_fwd_coeff.bin", this->blobs_[0]->cpu_data(), degree_, (channel_shared_ ? 1 : bottom[0]->channels()));
  // END DEBUG
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* coeff_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  MklForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, coeff_data, div_factor,
      degree_, Dtype(neg_slope_));
  CUDA_POST_KERNEL_CHECK;

  // DEBUG
//  save_to_matlab("mat_gpu_fwd_y0.bin", top[0]->cpu_data(), top[0]->width(), top[0]->width() * top[0]->channels());
//  save_to_matlab("mat_gpu_fwd_y95.bin", top[0]->cpu_data() + top[0]->offset(95), top[0]->width(), top[0]->width() * top[0]->channels());
  // END DEBUG
}

template <typename Dtype>
void MklLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // DEBUG
//  save_to_matlab("mat_gpu_bwd_x0.bin", bottom_memory_.cpu_data(), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
//  save_to_matlab("mat_gpu_bwd_x95.bin", bottom_memory_.cpu_data() + bottom[0]->offset(95), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
//  save_to_matlab("mat_gpu_bwd_coeff.bin", this->blobs_[0]->cpu_data(), degree_, (channel_shared_ ? 1 : bottom[0]->channels()));
//  save_to_matlab("mat_gpu_bwd_y0.bin", top[0]->cpu_data(), top[0]->width(), top[0]->width() * top[0]->channels());
//  save_to_matlab("mat_gpu_bwd_y95.bin", top[0]->cpu_data() + top[0]->offset(95), top[0]->width(), top[0]->width() * top[0]->channels());
//  save_to_matlab("mat_gpu_bwd_dy0.bin", top[0]->cpu_diff(), top[0]->width(), top[0]->width() * top[0]->channels());
//  save_to_matlab("mat_gpu_bwd_dy95.bin", top[0]->cpu_diff() + top[0]->offset(95), top[0]->width(), top[0]->width() * top[0]->channels());
  // END DEBUG
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* coeff_data = this->blobs_[0]->cpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  int div_factor = channel_shared_ ? channels : 1;

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* coeff_diff = this->blobs_[0]->mutable_gpu_diff();
    int cdim = channels * dim;
    for (int n = 0; n < bottom[0]->num(); ++n) {
      // compute element-wise diff
      // NOLINT_NEXT_LINE(whitespace/operators)
      MklParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
          CAFFE_CUDA_NUM_THREADS>>>(
          cdim, channels, dim, top_diff + top[0]->offset(n),
          bottom_data + bottom[0]->offset(n),
          backward_buff_.mutable_gpu_data(), coeff_data, div_factor, degree_);
      CUDA_POST_KERNEL_CHECK;
      int num_param = channel_shared_ ? degree_ : degree_ * channels;
      // DEBUG
//      if (n<10) {
//        char s[64];
//        sprintf(s, "mat_gpu_bwd_buff%d.bin", n);
//        save_to_matlab(s, backward_buff_.cpu_data(), num_param, backward_buff_.count()/num_param);
//        sprintf(s, "mat_gpu_bwd_mult%d.bin", n);
//        save_to_matlab(s, multiplier_.cpu_data(), backward_buff_.count()/num_param, 1);
//      }
//      coeff_diff = this->blobs_[0]->mutable_gpu_diff();
      // END DEBUG
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num_param, backward_buff_.count()/num_param, 1.,
          backward_buff_.gpu_data(),
          multiplier_.gpu_data(), 1., coeff_diff);
      // DEBUG
      //if (n<10) {
      //  char s[64];
      //  sprintf(s, "mat_gpu_bwd_dcoeff%d.bin", n);
      //  save_to_matlab(s, this->blobs_[0]->cpu_diff(), degree_, (channel_shared_ ? 1 : bottom[0]->channels()));
      //}
      // END DEBUG
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* coeff_data = this->blobs_[0]->gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MklBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff, coeff_data,
        div_factor, degree_, Dtype(neg_slope_));
    CUDA_POST_KERNEL_CHECK;
  }
  // DEBUG
  //save_to_matlab("mat_gpu_bwd_dcoeff.bin", this->blobs_[0]->cpu_diff(), degree_, (channel_shared_ ? 1 : bottom[0]->channels()));
  //save_to_matlab("mat_gpu_bwd_dx0.bin", bottom[0]->cpu_diff(), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
  //save_to_matlab("mat_gpu_bwd_dx95.bin", bottom[0]->cpu_diff() + bottom[0]->offset(95), bottom[0]->width(), bottom[0]->width() * bottom[0]->channels());
  //LOG(FATAL) << "FIX ME!";
  // END DEBUG
}


INSTANTIATE_LAYER_GPU_FUNCS(MklLayer);


}  // namespace caffe
