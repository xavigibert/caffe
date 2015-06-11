#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define EPSILON (1e-6)

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
void DictionaryLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  DictionaryParameter dict_param = this->layer_param_.dictionary_param();
  CHECK(!dict_param.has_kernel_size() !=
      !(dict_param.has_kernel_h() && dict_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(dict_param.has_kernel_size() ||
      (dict_param.has_kernel_h() && dict_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!dict_param.has_pad() && dict_param.has_pad_h()
      && dict_param.has_pad_w())
      || (!dict_param.has_pad_h() && !dict_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!dict_param.has_stride() && dict_param.has_stride_h()
      && dict_param.has_stride_w())
      || (!dict_param.has_stride_h() && !dict_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (dict_param.has_kernel_size()) {
    this->kernel_h_ = this->kernel_w_ = dict_param.kernel_size();
  } else {
    this->kernel_h_ = dict_param.kernel_h();
    this->kernel_w_ = dict_param.kernel_w();
  }
  CHECK_GT(this->kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(this->kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!dict_param.has_pad_h()) {
    this->pad_h_ = this->pad_w_ = dict_param.pad();
  } else {
    this->pad_h_ = dict_param.pad_h();
    this->pad_w_ = dict_param.pad_w();
  }
  if (!dict_param.has_stride_h()) {
    this->stride_h_ = this->stride_w_ = dict_param.stride();
  } else {
    this->stride_h_ = dict_param.stride_h();
    this->stride_w_ = dict_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = this->kernel_w_ == 1 && this->kernel_h_ == 1
      && this->stride_h_ == 1 && this->stride_w_ == 1 && this->pad_h_ == 0 && this->pad_w_ == 0;
  // Configure output channels and groups.
  this->channels_ = bottom[0]->channels();
  this->num_output_ = this->layer_param_.dictionary_param().num_output();
  CHECK_GT(this->num_output_, 0);
  // Check all other configuration parameters
  this->lambda_ = this->layer_param_.dictionary_param().lambda();
  this->num_iter_cg_ = this->layer_param_.dictionary_param().num_iter_cg();
  CHECK_GT(this->num_iter_cg_, 0);
  this->num_iter_irls_ = this->layer_param_.dictionary_param().num_iter_irls();
  CHECK_GT(this->num_iter_irls_, 0);
  this->do_learn_dictionary_ = this->layer_param_.dictionary_param().do_learn_dictionary();
  this->init_rate_ = this->layer_param_.dictionary_param().init_rate();
  this->init_delay_ = this->layer_param_.dictionary_param().init_delay();
  this->soft_th_ = this->layer_param_.dictionary_param().soft_th();
  this->num_blocks_ = this->layer_param_.dictionary_param().num_blocks();
  CHECK_GT(this->num_blocks_, 0);
  this->initial_block_size_ = this->layer_param_.dictionary_param().initial_block_size();
  CHECK_GT(this->initial_block_size_, 0);
  this->block_increase_rate_ = this->layer_param_.dictionary_param().block_increase_rate();
  this->max_block_size_ = this->layer_param_.dictionary_param().max_block_size();
  this->max_iter_bcd_ = this->layer_param_.dictionary_param().max_iter_bcd();
  this->epsilon_bcd_ = this->layer_param_.dictionary_param().epsilon_bcd();
  this->dict_update_interval_ = this->layer_param_.dictionary_param().dict_update_interval();
  this->dict_update_delay_ = this->layer_param_.dictionary_param().dict_update_delay();
  this->stat_decay_gamma_ = this->layer_param_.dictionary_param().stat_decay_gamma();
  this->replace_min_counts_ = this->layer_param_.dictionary_param().replace_min_counts();
  this->replace_threshold_ = this->layer_param_.dictionary_param().replace_threshold();
  this->reserve_bias_output_ = this->layer_param_.dictionary_param().reserve_bias_output();
  this->skip_count_ = this->layer_param_.dictionary_param().skip_batch_count();
  // Handle the parameters: dictionary (weights) and biases.
  // - blobs_[0] holds the dictionary (mxk)
  // - blobs_[1] holds the matrix A (kxk) (sum (code * code^T))
  // - blobs_[2] holds the matrix B (mxk) (sum ( x * code^T))
  // - blobs_[3] holds partial sums of matrix A (kxk) (sum (code * code^T)) p times
  // - blobs_[4] holds partial sums of matrix B (mxk) (sum ( x * code^T)) p times
  // - blobs_[5] holds a copy of the sample counters (1x6) as well as the counts and
  //             pseudocounts per dictionary atom
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
    read_counters_from_blob();
  } else {
    this->blobs_.resize(6);
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        this->num_output_, this->channels_, this->kernel_h_, this->kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.dictionary_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // Initialize matrix A and B to zeros
    this->blobs_[1].reset(new Blob<Dtype>(
        1, 1, this->num_output_, this->num_output_));
    this->blobs_[2].reset(new Blob<Dtype>(
        1, 1, this->channels_ * this->kernel_h_ * this->kernel_w_, this->num_output_));
    // Initialize partial sums of A and B to zeros
    this->blobs_[3].reset(new Blob<Dtype>(
        1, this->num_blocks_, this->num_output_, this->num_output_));
    this->blobs_[4].reset(new Blob<Dtype>(
        1, this->num_blocks_, this->channels_ * this->kernel_h_ * this->kernel_w_, this->num_output_));
    // Initialize counters to zero
    int m = channels_ * kernel_h_ * kernel_w_;
    int k = num_output_;
    LOG(INFO) << "m = " << m << ", k = " << k;
    this->blobs_[5].reset(new Blob<Dtype>(
        1, 1, 1, 6+2*k));
    // Inititialize counters
    if (this->phase_ == TRAIN) {
      cnt_init_delay_ = this->init_delay_;
      cnt_init_vectors_ = this->num_output_;
      if (reserve_bias_output_) {
        // Last element will be the bias term
        --cnt_init_vectors_;
        vector<Dtype> bias_elem(m, (Dtype)1./sqrt(m));
        replace_dictionary_atom_cpu(m, k, cnt_init_vectors_,
            this->blobs_[0]->mutable_cpu_data(), &bias_elem[0],
            this->blobs_[1]->mutable_cpu_data(),
            this->blobs_[2]->mutable_cpu_data(),
            this->blobs_[5]->mutable_cpu_data() + k+6,
            this->blobs_[5]->mutable_cpu_data() + 6, false);
      }
      sample_idx_ = 0;
      block_idx_ = 0;
      block_size_ = this->initial_block_size_;
      block_pos_ = 0;
    }
    else
    {
      cnt_init_delay_ = 0;
      cnt_init_vectors_ = 0;
      sample_idx_ = 0;
      block_idx_ = 0;
      block_size_ = 0;
      block_pos_ = 0;
    }
    // Copy counters to blobs_[5] positions 0 to 5
    save_counters_to_blob();
    // Note: blobs_[5] positions 6 to k+5 are samples pseudo-counts, which are 0
    // Update position k+6 to 2*k+5 with -1 indicating that such positions have not
    // been initialized, yet
    Dtype* atom_cnts = this->blobs_[5]->mutable_cpu_data() + k+6;
    caffe_set(k, Dtype(-1.), atom_cnts);
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), false);
  this->param_propagate_down_[0] = this->do_learn_dictionary_;
}

template <typename Dtype>
void DictionaryLayer<Dtype>::read_counters_from_blob() {
  Dtype* counters = this->blobs_[5]->mutable_cpu_data();
  cnt_init_delay_ = int(counters[0]);
  cnt_init_vectors_ = int(counters[1]);
  sample_idx_ = int(counters[2]);
  block_idx_ = int(counters[3]);
  block_size_ = int(counters[4]);
  block_pos_ = int(counters[5]);
}

template <typename Dtype>
void DictionaryLayer<Dtype>::save_counters_to_blob() {
  Dtype* counters = this->blobs_[5]->mutable_cpu_data();
  counters[0] = Dtype(cnt_init_delay_);
  counters[1] = Dtype(cnt_init_vectors_);
  counters[2] = Dtype(sample_idx_);
  counters[3] = Dtype(block_idx_);
  counters[4] = Dtype(block_size_);
  counters[5] = Dtype(block_pos_);
}

template <typename Dtype>
void DictionaryLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  compute_output_shape();
  CHECK_EQ(top.size(), 2);
  // top[2*i] Corresponds to the sparse codes
  // top[2*i+1] Is a scalar containing the objective function (loss)
  for (int top_id = 0; top_id < top.size()/2; ++top_id) {
    top[2*top_id]->Reshape(num_, num_output_, height_out_, width_out_);
    top[2*top_id+1]->Reshape(1, 1, 1, 1);
  }
  conv_out_spatial_dim_ = height_out_ * width_out_;
  kernel_dim_ = channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = num_output_ * kernel_dim_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = num_output_ * conv_out_spatial_dim_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);
  // We need to store sparse codes in one pixel per row order and then transpose
  // to generate the output.
  sparse_codes_buffer_.Reshape(1, height_out_, width_out_, num_output_);
  // We need to set up buffers for intermediate data used during sparse coding
  int m = kernel_dim_;
  int k = num_output_;
  vec_d_buffer_.Reshape(1, 1, 1, k);       // D^T * x
  vec_r_buffer_.Reshape(1, 1, 1, k);       // Residual vector
  vec_p_buffer_.Reshape(1, 1, 1, k);       // Descent direction
  vec_w_buffer_.Reshape(1, 1, 1, k);       // Vector w
  C_buffer_.Reshape(1, 1, k, std::max(k,m));
  Z_buffer_.Reshape(1, 1, k, m);           // inv(Y+D^T*D)*D^T
  W_buffer_.Reshape(1, 1, k, m);
  tmp_buffer_.Reshape(1, 1, 1, 4*std::max(k,m));    // Temporary storage
  mod_alpha_diff_buffer_.Reshape(1, 1, 1, k);
  DtDinv_buffer_.Reshape(1, 1, 1, k);
  Ddagger_buffer_.Reshape(1, 1, k, m);
  // Initialize orthonormalization order of dictionary columns
  dict_order_.resize(k);
  for (int i = 0; i < k; ++i)
    dict_order_[i] = i;
}

template <typename Dtype>
void DictionaryLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void DictionaryLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  skip_count_ = std::max(0,skip_count_ - 1);
  if (skip_count_ > 0)
    return;
  Dtype* dictionary = this->blobs_[0]->mutable_cpu_data();
  Dtype* A = this->blobs_[1]->mutable_cpu_data();
  Dtype* B = this->blobs_[2]->mutable_cpu_data();
  int sizeA = this->blobs_[1]->count();
  int sizeB = this->blobs_[2]->count();
  Dtype* partA = this->blobs_[3]->mutable_cpu_data() + sizeA * block_idx_;
  Dtype* partB = this->blobs_[4]->mutable_cpu_data() + sizeB * block_idx_;
  //int num_blocks = this->num_blocks_;
  bool do_learn_dictionary = this->do_learn_dictionary_;
  // Normalize dictionary (make sure that the norm for each column is <= 1)
  //if (this->phase_ == TRAIN && do_learn_dictionary)
  //  normalize_dictionary_cpu(kernel_dim_, num_output_, dictionary);
  // Orthonormalize dictionary (make sure that D^T*D=I)
  if (this->phase_ == TRAIN)
    orthogonalize_dictionary_cpu(kernel_dim_, num_output_, dictionary, &dict_order_[0]);
  /*
  int m = kernel_dim_;
  int k = num_output_;
  // Precompute C = C^T * D
  Dtype* C = C_buffer_.mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, k, k, m,
       (Dtype)1., dictionary, dictionary,
       (Dtype)0., C);
  // Save diag of D^T * D
  Dtype* diagDtD = diagDtD_buffer_.mutable_cpu_data();  // diag(D^T*D)
  for (int i = 0; i < k; ++i)
    diagDtD[i] = C[i*k+i];
  */
  // Perform sparse coding (and optionally dictionary learning) on each input vector
  for (int i = 0; i < top.size()/2; ++i) {        
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[2*i]->mutable_cpu_data();
    double loss = 0.;
    //LOG(INFO) << "Input norm = " << sqrt(caffe_cpu_dot<Dtype>(bottom[i]->count(),
    //    bottom_data, bottom_data)/bottom[i]->count());
    for (int n = 0; n < this->num_; ++n) {
      // Perform forward sparse coding
      loss += this->forward_cpu_sparse_coding(bottom_data + bottom[i]->offset(n), dictionary,
          A, B, partA, partB, top_data + top[2*i]->offset(n));
      bool initialized = cnt_init_delay_ == 0 && cnt_init_vectors_ == 0;
      if (this->phase_ == TRAIN && do_learn_dictionary_ && initialized) {
        // Update counters
        ++sample_idx_;
        ++block_pos_;
        /*
        if (block_pos_>= block_size_) {
          // Handle block swapping
          block_pos_ = 0;
          block_idx_ = (block_idx_ + 1) % num_blocks;
          partA = this->blobs_[3]->mutable_cpu_data() + sizeA * block_idx_;
          partB = this->blobs_[4]->mutable_cpu_data() + sizeB * block_idx_;
          // Clear oldest block
          caffe_set(sizeA, Dtype(0.), partA);
          caffe_set(sizeB, Dtype(0.), partB);
          // Recompute sum from partial sums
          caffe_set(sizeA, Dtype(0.), A);
          caffe_set(sizeB, Dtype(0.), B);
          for (int bi = 0; bi < num_blocks; ++bi) {
            const Dtype* pA = this->blobs_[3]->mutable_cpu_data() + sizeA * bi;
            const Dtype* pB = this->blobs_[4]->mutable_cpu_data() + sizeB * bi;
            caffe_add(sizeA, A, pA, A);
            caffe_add(sizeB, B, pB, B);
          }
          // Update block size according to configuration settings
          block_size_ = int(block_size_ * block_increase_rate_);
          if (block_size_ > max_block_size_)
            block_size_ = max_block_size_;
          LOG(INFO) << "Setting block size to " << block_size_ << " at sample index " << sample_idx_;
        }
        */
      }
    }
    // Put objective value in second output
    top_data = top[2*i+1]->mutable_cpu_data();
    *top_data = Dtype(loss/num_);
  }
  // Save counters to a blob, so they get reloaded if training is restarted
  save_counters_to_blob();
}

template <typename Dtype>
void DictionaryLayer<Dtype>::update_dictionary_cpu(int m, int k,
      const Dtype* alpha, const Dtype* x, Dtype* D,
      Dtype* A, Dtype* B, Dtype* partA, Dtype* partB, bool do_update_dict) {
  // Update matrix A = A + alpha * alpha^T
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, k, k, 1,
       Dtype(1.), alpha, alpha, Dtype(1.), A);
  // Update matrix B = B + x * alpha^T
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, m, k, 1,
       Dtype(1.), x, alpha, Dtype(1.), B);
  // Update partial sums partA and partB
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, k, k, 1,
       Dtype(1.), alpha, alpha, Dtype(1.), partA);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, m, k, 1,
       Dtype(1.), x, alpha, Dtype(1.), partB);
  if( do_update_dict ) {
    Dtype* tmp = tmp_buffer_.mutable_cpu_data();  // Dtype[m];  jth column of D (unnormalized)
    Dtype* Aj = tmp + m;                            // Dtype[k];  jth column of A
    double prev_cost = objective_function_cpu(m, k, D, x, alpha);
    double conv_criteria = prev_cost * epsilon_bcd_;
    LOG(INFO) << "Iter " << 0 << ": Objective = " << objective_function_cpu(m, k, D, x, alpha);
    // Update dictionary
    for (int iter = 0; iter < max_iter_bcd_; ++iter) {
      int k1 = reserve_bias_output_ ? k-1 : k;
      for (int j = 0; j < k1; ++j) {
        // Update column j
        Dtype Ajj = A[j*k+j];
        for (int i = 0; i < m; ++i)
          tmp[i] = B[i*k+j] + Ajj * D[i*k+j];
        for (int i = 0; i < k; ++i)
          Aj[i] = A[i*k+j];
        // Compute tmp = (b_j-D*a_j)+A_jj*d_j
        caffe_cpu_gemv<Dtype>(CblasNoTrans, m, k,
             -Dtype(1.), D, Aj, Dtype(1.), tmp);
        // Update column j of D
        // d_j = uj / max(A_jj,norm(uj));
        Dtype norm_tmp = caffe_cpu_dot<Dtype>(m, tmp, tmp);
        norm_tmp = sqrt(norm_tmp);
        if (norm_tmp < Ajj)
          norm_tmp = Ajj;
        for (int i = 0; i < m; ++i)
          D[i*k+j] = tmp[i] / norm_tmp;
      }
      // Check for convergence criteria
      double cost = objective_function_cpu(m, k, D, x, alpha);
      LOG(INFO) << "Iter " << iter+1 << ": Objective = " << cost;
      if (prev_cost - cost < conv_criteria)
        break;
      prev_cost = cost;
    }
  }
}

//template <typename Dtype>
//void DictionaryLayer<Dtype>::normalize_dictionary_cpu(int m, int k, Dtype* D) {
//  Dtype* col_sum = tmp_buffer_.mutable_cpu_data();
//  Dtype* col_sum2 = col_sum + k;
//  Dtype* col_std = col_sum + 2*k;
//  Dtype* col_bias = col_sum + 3*k;
//  caffe_set<Dtype>(k, Dtype(0), col_sum);
//  caffe_set<Dtype>(k, Dtype(0), col_sum2);
//  caffe_set<Dtype>(k, Dtype(0), col_std);
//  caffe_set<Dtype>(k, Dtype(0), col_bias);
//  for (int row_idx = 0; row_idx < m; ++row_idx) {
//    Dtype* ptr = &D[k*row_idx];
//    for (int col_idx = 0; col_idx < k; ++col_idx) {
//      col_sum[col_idx] += ptr[col_idx];
//      col_sum2[col_idx] += ptr[col_idx]*ptr[col_idx];
//    }
//  }
//  int k1 = reserve_bias_output_ ? k-1 : k;
//  for (int j = 0; j < k1; ++j) {
//    col_bias[j] = reserve_bias_output_ ? col_sum[j]/m : Dtype(0);
//    col_std[j] = sqrt(col_sum2[j]/m - col_bias[j]*col_bias[j]);
//  }
//  for (int col_idx = 0; col_idx < k1; ++col_idx) {
//    if (col_std[col_idx] < Dtype(1.))
//      col_std[col_idx] = Dtype(1.);
//    else
//      col_std[col_idx] = sqrt(col_std[col_idx]);
//  }
//  for (int row_idx = 0; row_idx < m; ++row_idx) {
//    Dtype* ptr = &D[k*row_idx];
//    for (int col_idx = 0; col_idx < k1; ++col_idx)
//      ptr[col_idx] = (ptr[col_idx] - col_bias[col_idx]) / col_std[col_idx];
//  }
//}

//template <typename Dtype>
//void DictionaryLayer<Dtype>::orthonormalize_dictionary_cpu(int m, int k, Dtype* D, int* order) {
//  // Orthonormalize in a temporary matrix Z (D transposed and permutted)
//  Dtype* Z = Z_buffer_.mutable_cpu_data();
//  for (int i = 0; i < k; ++i) {
//    int si = order[i];  // Source column si (destination row i)
//    for (int j = 0; j < m; ++j) {
//      Z[i*m+j] = D[j*k+si];
//    }
//  }
//  // Apply Gram-Schmidt orthonormalization
//  for (int i = 0; i < k; ++i) {
//    // Normalize vector i
//    Dtype* vi = Z + i*m;
//    Dtype norm = sqrt(caffe_cpu_dot<Dtype>(m, vi, vi));
//    caffe_scal(m, Dtype(1.)/norm, vi);
//    // Remove projection with other vectors
//    for (int i1 = i+1; i1 < k; ++i1) {
//      Dtype* vi1 = Z + i1*m;
//      Dtype proj = caffe_cpu_dot<Dtype>(m, vi, vi1);
//      caffe_axpy<Dtype>(m, -proj, vi, vi1);
//    }
//  }
//  // Move result back to D
//  for (int i = 0; i < k; ++i) {
//    int di = order[i]; // Source row i (destination column di)
//    for (int j = 0; j < m; ++j) {
//      D[j*k+di] = Z[i*m+j];
//    }
//  }
//}

template <typename Dtype>
void DictionaryLayer<Dtype>::orthogonalize_dictionary_cpu(int m, int k, Dtype* D, int* order) {
  // Orthonormalize in a temporary matrix Z (D transposed and permutted)
  Dtype* Z = Z_buffer_.mutable_cpu_data();
  for (int i = 0; i < k; ++i) {
    int si = order[i];  // Source column si (destination row i)
    for (int j = 0; j < m; ++j) {
      Z[i*m+j] = D[j*k+si];
    }
  }
  // We will save the norms in a vector, so we can unnormalize after we finish
  // orthonormalization
  Dtype* norm = tmp_buffer_.mutable_cpu_data();
  // Apply Gram-Schmidt orthonormalization
  for (int i = 0; i < k; ++i) {
    // Normalize vector i
    Dtype* vi = Z + i*m;
    norm[i] = sqrt(caffe_cpu_dot<Dtype>(m, vi, vi));
    caffe_scal(m, Dtype(1.)/norm[i], vi);
    // Remove projection with other vectors
    for (int i1 = i+1; i1 < k; ++i1) {
      Dtype* vi1 = Z + i1*m;
      Dtype proj = caffe_cpu_dot<Dtype>(m, vi, vi1);
      caffe_axpy<Dtype>(m, -proj, vi, vi1);
    }
  }
  // Unnormalize columns whose norm was originally less than 1
  for (int i = 0; i < k; ++i) {
    if (norm[i] < (Dtype)1.)
      caffe_scal(m, norm[i], Z + i*m);
  }
  // Move result back to D
  for (int i = 0; i < k; ++i) {
    int di = order[i]; // Source row i (destination column di)
    for (int j = 0; j < m; ++j) {
      D[j*k+di] = Z[i*m+j];
    }
  }
}

// Perform B = A^T
template<typename Dtype>
void DictionaryLayer<Dtype>::transpose_cpu(int m, int k, const Dtype* A, Dtype* B) {
  for (int i = 0; i < m; ++i) {
    const Dtype* src = A + i*k;
    Dtype* dst = B + i;
    for (int j = 0; j < k; ++j, dst+=m) {
        *dst = src[j];
    }
  }
}

template <typename Dtype>
double DictionaryLayer<Dtype>::forward_cpu_sparse_coding(const Dtype* input,
    Dtype* dictionary, Dtype* A, Dtype* B, Dtype* partA, Dtype* partB,
    Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  // Perform sparse coding for each input vector
  int m = kernel_dim_;
  int k = num_output_;
  Dtype* vec_d = vec_d_buffer_.mutable_cpu_data();      // D^T * x
  Dtype* vec_r = vec_r_buffer_.mutable_cpu_data();      // Residual vector
  Dtype* vec_p = vec_p_buffer_.mutable_cpu_data();      // Descent direction
  Dtype* vec_w = vec_w_buffer_.mutable_cpu_data();      // Vector w
  Dtype* sparse_codes = sparse_codes_buffer_.mutable_cpu_data();
  Dtype* pseudocounts = this->blobs_[5]->mutable_cpu_data() + 6;
  Dtype* counts = this->blobs_[5]->mutable_cpu_data() + k+6;
  // Precompute C = D^T * D.  NOT NEEDED, WE FORCE D^T * D = I
  Dtype* C = C_buffer_.mutable_cpu_data();              // (2*lambda*diag(1/abs(alpha[]))+D^T*D)
  caffe_set<Dtype>(k*k, (Dtype)0., C);

  //caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, k, k, m,
  //     (Dtype)1., dictionary, dictionary,
  //     (Dtype)0., C);
  // Save diag of D^T * D
  //Dtype* diagDtD = diagDtD_buffer_.mutable_cpu_data();  // diag(D^T*D)
  //for (int i = 0; i < k; ++i)
  //  diagDtD[i] = C[i*k+i];
  // Initialize loss
  double loss = 0.;
  for (int i = 0; i < conv_out_spatial_dim_; ++i)
  {
    const Dtype* x = col_buff + i*m;  // Input sample
    Dtype* vec_alpha = sparse_codes + i*k;   // Sparse code vector
    // Initialize sparse code
    for (int j = 0; j < k; ++j)
      vec_alpha[j] = 1.0;
    // Perform num_iter_irls iterations of iteratively reweighted
    // least squares using the previous result as starting value
    for (int iter_irls = 0; iter_irls < num_iter_irls_; ++iter_irls)
    {
      // Build matrix C = diag(2*lambda/fabs(alpha[]) + D^T * D
      for (int j = 0; j < k; ++j)
        //C[j*k+j] = 2*lambda_/(fabs(vec_alpha[j])+EPSILON) + diagDtD[j];
        C[j*k+j] = 2*lambda_/(fabs(vec_alpha[j])+EPSILON) + (Dtype)1.;
      // Build vector d = D^T * x
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, k, 1, m,
           (Dtype)1., dictionary, x,
           (Dtype)0., vec_d);
      // Perform conjugate gradient descent to approximately solve
      // C * alpha = d     for alpha
      conjugate_gradient_cpu(k, C, vec_d, vec_alpha, num_iter_cg_, vec_p, vec_r, vec_w);
    }
    // Apply hard threshold to sparse codes
    for (int j = 0; j < k; ++j) {
      if (fabs(vec_alpha[j]) < lambda_)
        vec_alpha[j] = (Dtype)0.;
    }
    loss += objective_function_cpu(m, k, dictionary, x, vec_alpha);
  }

  // Perform online dictionary update
  bool initialized = cnt_init_delay_ == 0 && cnt_init_vectors_ == 0;
  if (this->phase_ == TRAIN && do_learn_dictionary_ && initialized) {
    for (int i = 0; i < conv_out_spatial_dim_; ++i) {
      // Premultiply A and B by gamma
      int global_sample_idx = i+sample_idx_*conv_out_spatial_dim_;
      //Dtype gamma = (Dtype)std::min(stat_decay0_ + global_sample_idx*stat_decay_rate_,
      //    stat_decay1_);
      Dtype gamma = 1.0 - 1.0 / (1.0 + global_sample_idx*stat_decay_gamma_ );
      if (gamma < Dtype(1.)) {
        caffe_scal<Dtype>(this->blobs_[1]->count(), gamma,
            this->blobs_[1]->mutable_cpu_data());
        caffe_scal<Dtype>(this->blobs_[2]->count(), gamma,
            this->blobs_[2]->mutable_cpu_data());
        caffe_scal<Dtype>(this->blobs_[3]->count(), gamma,
            this->blobs_[3]->mutable_cpu_data());
        caffe_scal<Dtype>(this->blobs_[4]->count(), gamma,
            this->blobs_[4]->mutable_cpu_data());
        caffe_scal<Dtype>(k, sqrt(gamma), pseudocounts);
      }
      // Perform dictionary update
      Dtype* vec_alpha = sparse_codes + i*num_output_;  // Sparse code
      const Dtype* x = col_buff + i*kernel_dim_;  // Input sample
      //bool do_update_dict = global_sample_idx % dict_update_interval_== 0
      //    && global_sample_idx >= dict_update_delay_;
      bool do_update_dict = false;
      update_dictionary_cpu(kernel_dim_, num_output_, vec_alpha, x, dictionary,
                        A, B, partA, partB, do_update_dict);
      // Update counts, pseudocounts and replace dictionary atoms, if necessary
      Dtype norm_alpha = Dtype(0.);
      for (int j = 0; j < k; ++j) {
        if (fabs(vec_alpha[j]) > norm_alpha)
          norm_alpha = fabs(vec_alpha[j]);
      }
      //Dtype norm_alpha = sqrt(caffe_cpu_dot<Dtype>(k, vec_alpha, vec_alpha));
      for (int j = 0; j < k; ++j) {
        if (counts[j] >= Dtype(0.))
          counts[j] += Dtype(1.);
        pseudocounts[j] += norm_alpha;
      }
    }
  }
  // Sparse codes are in pixel-first order, we need to transpose them so they
  // are in channel-first order
  transpose_cpu(conv_out_spatial_dim_, num_output_, sparse_codes, output);
/*
  // Perform dictionary initialization
  if (this->phase_ == TRAIN) {
    if (cnt_init_delay_ > 0) {
      // Delay initialization
      --cnt_init_delay_;
    } else if (cnt_init_vectors_ > 0) {
      // Initialize dictionary with input samples
      vector<float> val(conv_out_spatial_dim_, 0.f);
      caffe_rng_uniform<float>(conv_out_spatial_dim_, 0.f, 1.f, &val[0]);
      for (int i = 0; i < conv_out_spatial_dim_; ++i)
      {
        if (val[i] <= init_rate_ && cnt_init_vectors_ > 0) {
          --cnt_init_vectors_;
          //LOG(INFO) << "Initializing atom " << cnt_init_vectors_ << " with vector " << i;
          // Replace dictionary element and update counts
          const Dtype* x = col_buff + i*m;    // Input sample
          replace_dictionary_atom_cpu(m, k, cnt_init_vectors_, dictionary, x,
              A, B, counts, pseudocounts, reserve_bias_output_);
        }
      }
    } else {
      for (int i = 0; i < conv_out_spatial_dim_; ++i)
      {
        // Check if some elements need to be replaced
        int replace_idx = -1;
        Dtype min_utilization = Dtype(replace_threshold_) / Dtype(k);
        int k1 = reserve_bias_output_ ? k-1 : k;
        for (int j = 0 ; j < k1; ++j) {
          if (counts[j] >= replace_min_counts_) {
            Dtype Ajj = A[j*k+j];
            Dtype utilization = sqrt(Ajj) / pseudocounts[j];
            if (utilization < min_utilization) {
              min_utilization = utilization;
              replace_idx = j;
            }
          }
        }
        if (replace_idx >= 0) {
          // Replace dictionary element and update counts
          const Dtype* x = col_buff + i*m;    // Input sample
          replace_dictionary_atom_cpu(m, k, replace_idx, dictionary, x, A, B,
              counts, pseudocounts, reserve_bias_output_);
          LOG(INFO) << "Replaced dictionary column " << replace_idx;
        }
      }
      // TEMPORARY CODE
      //save_to_matlab("mat_D.bin", dictionary, m, k);
      // END TEMPORARY CODE
    }
  }
*/
  return loss/conv_out_spatial_dim_;
}

// Replace dictionary atom with input sample and reset counters, corresponding
// row and column in matrix A, and corresponding column in matrix B
template <typename Dtype>
void DictionaryLayer<Dtype>::replace_dictionary_atom_cpu(int m, int k, int idx,
      Dtype* D, const Dtype* x, Dtype* A, Dtype* B, Dtype* counts,
      Dtype* pseudocounts, bool debias) {
  // We have observed that when x is very small, the dictionary learning is
  // unstable. Therefore, we normalize x to unit norm
  Dtype sum_x = (Dtype)0., sum_x2 = (Dtype)0.;
  for (int j = 0; j < m; ++j) {
    sum_x += x[j];
    sum_x2 += x[j] * x[j];
  }
  Dtype bias_x = debias ? sum_x/m : Dtype(0);
  Dtype std_x = sqrt(sum_x2/m  - bias_x*bias_x);
  for (int j = 0; j < m; ++j) {
    D[j*k + idx] = (x[j] - bias_x) / std_x;
    B[j*k + idx] = Dtype(0.);
  }  
  for (int j =0; j < k; ++j) {
    A[j*k + idx] = Dtype(0.);
    A[idx*k + j] = Dtype(0.);
  }
  counts[idx] = Dtype(0.);
  pseudocounts[idx] = Dtype(0.);
}

template <typename Dtype>
double DictionaryLayer<Dtype>::objective_function_cpu(int m, int k,
        const Dtype* D, const Dtype* x, const Dtype* alpha) {
  // Compute objective function
  // Cost(alpha) = 0.5*||x_t-D*alpha||_2^2 + lambda*||alpha||_1
  Dtype* tmp = tmp_buffer_.mutable_cpu_data();
  memcpy(tmp, x, m*sizeof(Dtype));
  caffe_cpu_gemv<Dtype>(CblasNoTrans, m, k,
                        (Dtype)(-1.), D, alpha,
                        (Dtype)1., tmp);
  double cost = 0.;
  for (int i = 0; i < k; ++i)
    cost += fabs(alpha[i]);
  cost = 0.5 * caffe_cpu_dot<Dtype>(m, tmp, tmp) + lambda_ * cost;
  return cost;
}

template <typename Dtype>
void DictionaryLayer<Dtype>::conjugate_gradient_cpu(int k, const Dtype* C,
      const Dtype* d, Dtype* x, int num_iter, Dtype* temp_p, Dtype* temp_r, Dtype* temp_w) {
  // Initialize the residual
  memcpy(temp_r, d, k*sizeof(Dtype));
  caffe_cpu_gemv<Dtype>(CblasNoTrans, k, k,
       (Dtype)(-1.), C, x, (Dtype)1., temp_r);
  // Compute norm of the residual
  Dtype prev_norm_r = caffe_cpu_dot<Dtype>(k, temp_r, temp_r);
  if (fabs(prev_norm_r) < EPSILON) {
    return;   // Accept initial solution
  }
  // Initialize the descent direction
  memcpy(temp_p, temp_r, k*sizeof(Dtype));
  // Perform num_iter_cg iterations of conjugate gradient descent
  for (int iter_cg = 0; iter_cg < num_iter; ++iter_cg)
  {
    // w = C * p
    caffe_cpu_gemv<Dtype>(CblasNoTrans, k, k,
         (Dtype)1., C, temp_p, (Dtype)0., temp_w);
    // alpha = norm(r) / dot(p, w)
    Dtype dot_p_w = caffe_cpu_dot<Dtype>(k, temp_p, temp_w);
    //if (fabs(dot_p_w) < EPSILON)
    //  return false;
    Dtype alpha = prev_norm_r / dot_p_w;
    CHECK(!isnan(alpha));
    // x = x + alpha*p
    caffe_axpy<Dtype>(k, alpha, temp_p, x);
    // r = r - alpha*w
    caffe_axpy<Dtype>(k, -alpha, temp_w, temp_r);
    // Compute norm of new residual
    Dtype norm_r = caffe_cpu_dot<Dtype>(k, temp_r, temp_r);
    // Compute beta
    Dtype beta = norm_r / prev_norm_r;
    CHECK(!isnan(beta));
    // p = r + beta*p
    caffe_cpu_axpby<Dtype>(k, (Dtype)1., temp_r, beta, temp_p);
    prev_norm_r = norm_r;
    if (fabs(prev_norm_r) < EPSILON) {
      return;
    }
  }
  return;
}

template <typename Dtype>
void DictionaryLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* D = this->blobs_[0]->cpu_data();
  Dtype* D_diff = this->blobs_[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), Dtype(0), D_diff);
  }
  //if (skip_count_ > 0)
  //  return;
  CHECK_EQ(conv_out_spatial_dim_, 1) << "Convolutional dictionaries not implemented, yet!";
  //bool initialized = cnt_init_delay_ == 0 && cnt_init_vectors_ == 0;
  // We do not want to backpropagate gradients until our dictionary is fully
  // initialized with input samples
  //if (!initialized)
  //  return;
  // Temporary storage and precomputed constants
  int m = kernel_dim_;
  int k = num_output_;
  Dtype* tmp1 = tmp_buffer_.mutable_cpu_data();
  Dtype* tmp2 = tmp_buffer_.mutable_cpu_data() + k;
  Dtype* tmp_dl_dx = tmp_buffer_.mutable_cpu_data() + 2*k;
  Dtype* DtDinv = DtDinv_buffer_.mutable_cpu_data();
  Dtype* Ddagger = Ddagger_buffer_.mutable_cpu_data();
  precompute_pseudoinverse_cpu(m, k, D, DtDinv, Ddagger);

  for (int idx = 0; idx < top.size()/2; ++idx) {
    const Dtype* top_diff = top[2*idx]->cpu_diff();
    const Dtype* top_data = top[2*idx]->cpu_data();
    const Dtype* bottom_data = bottom[idx]->cpu_data();
    Dtype* bottom_diff = bottom[idx]->mutable_cpu_diff();
    if (this->param_propagate_down_[0] || propagate_down[idx]) {
      for (int n = 0; n < this->num_; ++n) {
        const Dtype* alpha = top_data + top[idx*2]->offset(n);
        const Dtype* alpha_diff = top_diff + top[idx*2]->offset(n);
        int cnt_nz_alpha = 0;
        for (int i = 0; i < k; ++i) {
          if (alpha[i] != (Dtype)0.)
            ++cnt_nz_alpha;
        }
        //LOG(INFO) << "cnt_nz_alpha = " << cnt_nz_alpha;
        // Precompute modified output gradient
        Dtype* mod_alpha_diff = mod_alpha_diff_buffer_.mutable_cpu_data();
        for (int i = 0; i < k; ++i)
          mod_alpha_diff[i] = alpha[i] == (Dtype)0. ? (Dtype)0. : alpha_diff[i];
        // dl/dx is necessary for both gradients
        Dtype* dl_dx = propagate_down[idx] ?
              bottom_diff + bottom[idx]->offset(n) : tmp_dl_dx;
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[idx] || this->param_propagate_down_[0]) {
          this->backward_cpu_gemm(mod_alpha_diff, Ddagger, dl_dx);
        }
        // gradient w.r.t. dictionary. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->dict_cpu_backprop(bottom_data + bottom[idx]->offset(n),
              dl_dx, mod_alpha_diff, Ddagger, DtDinv, tmp1, tmp2, D_diff);
        }
      }
    }
  }
}

template <typename Dtype>
void DictionaryLayer<Dtype>::precompute_pseudoinverse_cpu(int m, int k,
    const Dtype* D, Dtype* DtDinv, Dtype* Ddagger) {
  // Precompute diag(inv(D^T*D))
  for (int i = 0; i < k; ++i)
    DtDinv[i] = (Dtype)1. / caffe_cpu_strided_dot(m, D+i, k, D+i, k);
  // Compute transpose of D^dagger by multiplying each row of D with diag(inv(D^T*D))
  for (int j = 0; j < m; ++j)
    caffe_mul(k, D+j*k, DtDinv, Ddagger+j*k);
}


template <typename Dtype>
void DictionaryLayer<Dtype>::dict_cpu_backprop(const Dtype* x, const Dtype* dl_dx,
    const Dtype* mod_alpha_diff, const Dtype* Dtdagger, const Dtype* diagDtDinv,
    Dtype* tmp1, Dtype* tmp2, Dtype* D_diff) {
  int m = kernel_dim_;
  int k = num_output_;
  // Compute intermediate products
  // tmp1 = x^T * (D^dagger)^T
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, k, m, (Dtype)1., x,
      Dtdagger, (Dtype)0., tmp1);
  // tmp2 = dl_dx * inv(D^T*D) = dl_dalpha *. diag(inv(D^T*D))
  caffe_mul(k, mod_alpha_diff, diagDtDinv, tmp2);
  // Compute gradient of dictionary and add it to D_diff
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, k, 1, -(Dtype)2., dl_dx,
      tmp1, (Dtype)1., D_diff);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, k, 1, (Dtype)1., x,
      tmp2, (Dtype)1., D_diff);
}

template <typename Dtype>
void DictionaryLayer<Dtype>::backward_cpu_gemm(const Dtype* mod_alpha_diff,
    const Dtype* D, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, kernel_dim_,
      1, num_output_,
      (Dtype)1., D, mod_alpha_diff,
      (Dtype)0., col_buff);
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

#ifdef CPU_ONLY
STUB_GPU(DictionaryLayer);
#endif

INSTANTIATE_CLASS(DictionaryLayer);
REGISTER_LAYER_CLASS(Dictionary);

}  // namespace caffe