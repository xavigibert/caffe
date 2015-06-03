#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define EPSILON (1e-6)

namespace caffe {

template <typename Dtype>
void DictionaryLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
  /*
  Dtype* dictionary = this->blobs_[0]->mutable_gpu_data();
  Dtype* A = this->blobs_[1]->mutable_gpu_data();
  Dtype* B = this->blobs_[2]->mutable_gpu_data();
  int sizeA = this->blobs_[1]->count();
  int sizeB = this->blobs_[2]->count();
  Dtype* partA = this->blobs_[3]->mutable_gpu_data() + sizeA * block_idx_;
  Dtype* partB = this->blobs_[4]->mutable_gpu_data() + sizeB * block_idx_;
  int num_blocks = this->num_blocks_;
  bool do_learn_dictionary = this->do_learn_dictionary_;
  // Normalize dictionary (make sure that the norm for each column is <= 1)
  if (this->phase_ == TRAIN && do_learn_dictionary)
    normalize_dictionary_gpu(kernel_dim_, num_output_, dictionary);
  // Perform sparse coding (and optionally dictionary learning) on each input vector
  for (int i = 0; i < top.size()/2; ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* loss = top[2*i]->mutable_gpu_data();
    // Set loss value to 0
    caffe_gpu_set(1, Dtype(.0), loss);
    for (int n = 0; n < this->num_; ++n) {
      // Perform forward sparse coding
      this->forward_gpu_sparse_coding(bottom_data + bottom[i]->offset(n), dictionary,
          A, B, partA, partB, top_data + top[2*i]->offset(n), loss);
      bool initialized = cnt_init_delay_ == 0 && cnt_init_vectors_ == 0;
      if (this->phase_ == TRAIN && do_learn_dictionary_ && initialized) {
        // Update counters
        ++sample_idx_;
        ++block_pos_;
        if (block_pos_>= block_size_) {
          // Handle block swapping
          block_pos_ = 0;
          block_idx_ = (block_idx_ + 1) % num_blocks;
          partA = this->blobs_[3]->mutable_gpu_data() + sizeA * block_idx_;
          partB = this->blobs_[4]->mutable_gpu_data() + sizeB * block_idx_;
          // Clear oldest block
          caffe_gpu_set(sizeA, Dtype(0.), partA);
          caffe_gpu_set(sizeB, Dtype(0.), partB);
          // Recompute sum from partial sums
          caffe_gpu_set(sizeA, Dtype(0.), A);
          caffe_gpu_set(sizeB, Dtype(0.), B);
          for (int bi = 0; bi < num_blocks; ++bi) {
            const Dtype* pA = this->blobs_[3]->mutable_gpu_data() + sizeA * bi;
            const Dtype* pB = this->blobs_[4]->mutable_gpu_data() + sizeB * bi;
            caffe_gpu_add(sizeA, A, pA, A);
            caffe_gpu_add(sizeB, B, pB, B);
          }
          // Update block size according to configuration settings
          block_size_ = int(block_size_ * block_increase_rate_);
          if (block_size_ > max_block_size_)
            block_size_ = max_block_size_;
          LOG(INFO) << "Setting block size to " << block_size_ << " at sample index " << sample_idx_;
        }
      }
    }
    // Put objective value in second output
    Dtype* loss = top[2*i+1]->mutable_cpu_data();
    // Normalize loss

    *top_data = Dtype(loss/num_);
  }
  // Save counters to a blob, so they get reloaded if training is restarted
  save_counters_to_blob();
  */

}

template <typename Dtype>
void DictionaryLayer<Dtype>::update_dictionary_gpu(int m, int k,
      const Dtype* alpha, const Dtype* x, Dtype* D,
      Dtype* A, Dtype* B, Dtype* partA, Dtype* partB, bool do_update_dict) {
  /*
  // Update matrix A = A + alpha * alpha^T
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, k, k, 1,
       Dtype(1.), alpha, alpha, Dtype(1.), A);
  // Update matrix B = B + x * alpha^T
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, m, k, 1,
       Dtype(1.), x, alpha, Dtype(1.), B);
  // Update partial sums partA and partB
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, k, k, 1,
       Dtype(1.), alpha, alpha, Dtype(1.), partA);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, m, k, 1,
       Dtype(1.), x, alpha, Dtype(1.), partB);
  if( do_update_dict ) {
    Dtype* tmp = new Dtype[m];   // jth column of D (unnormalized)
    Dtype* Aj = new Dtype[k];    // jth column of A
    double prev_cost = objective_function();
    double conv_criteria = prev_cost * epsilon_bcd_;
    // Update dictionary
    for (int iter = 0; iter < max_iter_bcd_; ++iter) {
      for (int j = 0; j < k; ++j) {
        // Update column j
        Dtype Ajj = A[j*k+j];
        for (int i = 0; i < m; ++i)
          tmp[i] = B[i*k+j] + Ajj * D[i*k+j];
        for (int i = 0; i < k; ++i)
          Aj[i] = A[i*k+j];
        // Compute tmp = (b_j-D*a_j)+A_jj*d_j
        caffe_cpu_gemv<Dtype>(CblasNoTrans, CblasNoTrans, m, k,
             -Dtype(1.), D, Aj, Dtype(1.), tmp);
        // Update column j of D
        // d_j = uj / max(A_jj,norm(uj));
        caffe_gpu_dot<Dtype>(m, tmp, tmp, norm_tmp);
        norm_tmp = sqrt(norm_tmp);
        if (norm_tmp < Ajj)
          norm_tmp = Ajj;
        for (int i = 0; i < m; ++i)
          D[i*k+j] = tmp[i] / norm_tmp;
      }
      // Check for convergence criteria
      double cost = objective_function();
      LOG(INFO) << "Iter " << iter+1 << ": Objective = " << cost;
      if (prev_cost - cost < conv_criteria)
        break;
      prev_cost = cost;
    }
  }
  */
}

template <typename Dtype>
void DictionaryLayer<Dtype>::normalize_dictionary_gpu(int m, int k, Dtype* D) {
  // TEMPORARY CODE
  Dtype* tmp_D = new Dtype[m*k];
  caffe_copy(m*k, D, tmp_D);
  normalize_dictionary_cpu(m, k, tmp_D);
  caffe_copy(m*k, tmp_D, D);
  free(tmp_D);
  // END TEMPORARY CODE
}

template <typename Dtype>
__global__ void transpose_kernel(int n, int m, int k, const Dtype* A, Dtype* B) {
  CUDA_KERNEL_LOOP(index, n) {
    int idx_row = index / m;
    int idx_col = index % m;
    B[index] = A[idx_row + idx_col*k];
  }
}

// Perform B = A^T
template<typename Dtype>
void DictionaryLayer<Dtype>::transpose_gpu(int m, int k, const Dtype* A, Dtype* B) {
  int N = m*k;
  transpose_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, m, k, A, B);
  // TEMPORARY CODE
//  Dtype* tmp_A = new Dtype[m*k];
//  Dtype* tmp_B = new Dtype[m*k];
//  caffe_copy(m*k, A, tmp_A);
//  caffe_copy(m*k, B, tmp_B);
//  transpose_gpu(m, k, tmp_A, tmp_B);
//  caffe_copy(m*k, tmp_A, A);
//  caffe_copy(m*k, tmp_B, B);
//  free(tmp_A);
//  free(tmp_B);
  // END TEMPORARY CODE
}

// Perform sparse coding on the GPU, estimate the loss and add it to the
// previous loss value
template <typename Dtype>
void DictionaryLayer<Dtype>::forward_gpu_sparse_coding(const Dtype* input,
    Dtype* dictionary, Dtype* A, Dtype* B, Dtype* partA, Dtype* partB,
    Dtype* output, Dtype* loss, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  // Perform sparse coding for each input vector
  int m = kernel_dim_;
  int k = num_output_;
  Dtype* vec_d = vec_d_buffer_.mutable_gpu_data();      // D^T * x
  Dtype* vec_r = vec_r_buffer_.mutable_gpu_data();      // Residual vector
  Dtype* vec_p = vec_p_buffer_.mutable_gpu_data();      // Descent direction
  Dtype* vec_w = vec_w_buffer_.mutable_gpu_data();      // Vector w
  Dtype* C = C_buffer_.mutable_gpu_data();              // (2*lambda*diag(1/abs(alpha[]))+D^T*D)
  Dtype* diagDtD = diagDtD_buffer_.mutable_gpu_data();  // diag(D^T*D)
  Dtype* sparse_codes = sparse_codes_buffer_.mutable_gpu_data();

  // Precompute C = D^T * D
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, k, k, m,
       (Dtype)1., dictionary, dictionary,
       (Dtype)0., C);
  // Save diag of D^T * D
  for (int i = 0; i < k; ++i)
    diagDtD[i] = C[i*k+i];

  // Initialize loss
  for (int i = 0; i < conv_out_spatial_dim_; ++i)
  {
    const Dtype* x = col_buff + i*m;  // Input sample
    Dtype* vec_alpha = sparse_codes + i*k;   // Sparse code vector
    // Initialize sparse code
    for (int j = 0; j < num_output_; j++)
      vec_alpha[j] = 1.0;
    // Perform num_iter_irls iterations of iteratively reweighted
    // least squares using the previous result as starting value
    for (int iter_irls = 0; iter_irls < num_iter_irls_; ++iter_irls)
    {
      // Build matrix C = diag(2*lambda/fabs(alpha[]) + D^T * D
      for (int j = 0; j < k; ++j)
        C[j*k+j] = 2*lambda_/(fabs(vec_alpha[j])+EPSILON) + diagDtD[j];
      // Build vector d = D^T * x
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, k, 1, m,
           (Dtype)1., dictionary, x,
           (Dtype)0., vec_d);
      // Perform conjugate gradient descent to approximately solve
      // C * alpha = d     for alpha
      conjugate_gradient_gpu(k, C, vec_d, vec_alpha, num_iter_cg_, vec_p, vec_r, vec_w);
    }
    add_objective_gpu(m, k, dictionary, x, vec_alpha, loss);
  }

  // Perform online dictionary update
  bool initialized = cnt_init_delay_ == 0 && cnt_init_vectors_ == 0;
  if (this->phase_ == TRAIN && do_learn_dictionary_ && initialized) {
    for (int i = 0; i < conv_out_spatial_dim_; ++i) {
      const Dtype* vec_alpha = sparse_codes + i*num_output_;  // Sparse code
      const Dtype* x = col_buff + i*kernel_dim_;  // Input sample
      bool do_update_dict = (i+sample_idx_*conv_out_spatial_dim_) % dict_update_interval_== 0
          && (i+sample_idx_*conv_out_spatial_dim_) >= dict_update_delay_;
      update_dictionary_gpu(kernel_dim_, num_output_, vec_alpha, x, dictionary,
                        A, B, partA, partB, do_update_dict);
    }
  }
  // Sparse codes are in pixel-first order, we need to transpose them so they are in
  // channel-first order
  transpose_gpu(conv_out_spatial_dim_, num_output_, sparse_codes, output);

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
          col_buff = col_buffer_.cpu_data();
          const Dtype* x = col_buff + i*m;    // Input sample
          Dtype* dst = dictionary + cnt_init_vectors_;
          for (int j = 0; j < m; ++j, dst+=k) {
            // NOTE: This is a very inefficient way to do "*dst = x[j];"
            caffe_gpu_memcpy(1, &x[j], dst);
          }
        }
      }
    } else {
      // TEMPORARY CODE
      //save_to_matlab("mat_D.bin", dictionary, m, k);
      // END TEMPORARY CODE
    }
  }

  return loss/conv_out_spatial_dim_;
}

template <typename Dtype>
void DictionaryLayer<Dtype>::add_objective_gpu(int m, int k,
        const Dtype* D, const Dtype* x, const Dtype* alpha,
        Dtype* loss) {
  // Compute objective function
  // Cost(alpha) = 0.5*||x_t-D*alpha||_2^2 + lambda*||alpha||_1
  Dtype* tmp = tmp_buffer_.mutable_gpu_diff();
  caffe_gpu_memcpy(m, x, tmp);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, 1, k,
                        (Dtype)(-1.), D, alpha,
                        (Dtype)1., tmp);
  Dtype* cost = tmp + m;
  caffe_gpu_dot<Dtype>(m, tmp, tmp, cost);
  caffe_gpu_scale<Dtype>(1, 0.5, cost, cost);
  Dtype* asum = cost + 1;
  // Sum of absolute values of elements in alpha
  caffe_gpu_asum<Dtype>(k, alpha, asum);
  caffe_gpu_scale<Dtype>(1, Dtype(lambda_), asum, asum);
  // Add cost and asum to loss
  caffe_gpu_add<Dtype>(1, asum, loss, loss);
  caffe_gpu_add<Dtype>(1, cost, loss, loss);
}

template <typename Dtype>
void DictionaryLayer<Dtype>::conjugate_gradient_gpu(int k, const Dtype* C,
      const Dtype* d, Dtype* x, int num_iter, Dtype* temp_p, Dtype* temp_r, Dtype* temp_w) {
  /*
  // Initialize the residual
  caffe_gpu_memcpy(k, d, temp_r);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, k, 1, k,
       (Dtype)(-1.), C, x, (Dtype)1., temp_r);
  // Allocate scalars
  Dtype* prev_norm_r = tmp_buffer_.mutable_gpu_diff();
  Dtype* alpha = prev_norm_r+1;
  Dtype* norm = alpha+1;
  // Compute norm of the residual
  caffe_gpu_dot<Dtype>(k, temp_r, temp_r, prev_norm_r);
  if (fabs(prev_norm_r) < EPSILON) {
    return;   // Accept initial solution
  }
  // Initialize the descent direction
  memcpy(temp_p, temp_r, k*sizeof(Dtype));
  // Perform num_iter_cg iterations of conjugate gradient descent
  for (int iter_cg = 0; iter_cg < num_iter; ++iter_cg)
  {
    // w = C * p
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, k, 1, k,
         (Dtype)1., C, temp_p, (Dtype)0., temp_w);
    // alpha = norm(r) / dot(p, w)
    Dtype dot_p_w = caffe_gpu_dot<Dtype>(k, temp_p, temp_w);
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
  */
}

template <typename Dtype>
void DictionaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
  /*
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {

          // TO DO
          //this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
          //    top_diff + top[i]->offset(n), weight_diff);

        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {

          // TO DO
          //this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
          //    bottom_diff + bottom[i]->offset(n));

        }
      }
    }
  }
  */
}

INSTANTIATE_LAYER_GPU_FUNCS(DictionaryLayer);

}  // namespace caffe
