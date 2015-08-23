#include <algorithm>
#include <vector>

#include "caffe/neuron_layers.hpp"

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
  degree_ = mkl_param.degree();
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
    //const double def_coeffs[5] = {0.1184, 0.5, 0.4056, 0.0, 0.0};
    //const double def_coeffs[5] = {0.1184, 0.5, 0.4056, 0.0, -0.0495};
    const double def_coeffs[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    Dtype* coeffs = this->blobs_[0]->mutable_cpu_data();
    int num_vects = channel_shared_ ? 1 : channels;
    for (int c = 0; c < num_vects; ++c) {
      for (int d = 0; d < degree_; ++d)
        coeffs[d+c*degree_] = d < 5 ? Dtype(def_coeffs[d]) : Dtype(0);
    }
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), degree_)
        << "Negative slope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels * degree_)
        << "Negative slope size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
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
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    const Dtype* weights = coeff_data + c*degree_;
    //Dtype pow_x = Dtype(1);   // x^d
    //Dtype result = weights[0] + bottom_data[i];
    Dtype result = weights[0] + (Dtype(1) + weights[1]) * bottom_data[i];
//    for (int d = 1; d < degree_; ++d) {
//      pow_x *= bottom_data[i];
//      result += weights[d] * pow_x;
//    }
    for (int d = 2; d < degree_; ++d)
      result += pow(weights[d] * bottom_data[i], Dtype(d));
    top_data[i] = result;
  }
}

template <typename Dtype>
void MklLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
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

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* coeff_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      Dtype* diff = coeff_diff + c*degree_;
      //Dtype pow_x = Dtype(1);  // x^d
//      for (int d = 0; d < degree_; ++d) {
//        diff[d] += top_diff[i] * pow_x;
//        pow_x *= bottom_data[i];
//      }
      diff[0] += top_diff[i];
      diff[1] += top_diff[i] * weights[0];
      for (int d = 2; d < degree_; ++d)
        diff[d] += Dtype(d) * bottom_data[i] * pow(weights[d] * bottom_data[i], Dtype(d-1));
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      const Dtype* weights = coeff_data + c*degree_;
      //Dtype pow_x = Dtype(1);  // x^(d-1)
      Dtype result = Dtype(1) + weights[1];
//      for (int d = 1; d < degree_; ++d) {
//        result += d * weights[d] * pow_x;
//        pow_x *= bottom_data[i];
//      }
      for (int d = 2; d < degree_; ++d)
        result += Dtype(d) * weights[d] * pow(weights[d] * bottom_data[i], Dtype(d-1));
      bottom_diff[i] = top_diff[i] * result;
    }
  }
}


//#ifdef CPU_ONLY
//STUB_GPU(MklLayer);
//#endif

INSTANTIATE_CLASS(MklLayer);
REGISTER_LAYER_CLASS(Mkl);

}  // namespace caffe
