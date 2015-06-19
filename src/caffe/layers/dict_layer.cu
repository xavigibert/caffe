#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#define EPSILON (1e-6)

namespace caffe {

template <typename Dtype>
__global__ void kernel_norm(int m, int k, const Dtype* D, Dtype* diagDtD) {
  CUDA_KERNEL_LOOP(index, k) {
    Dtype res = (Dtype)0.;
    for (int j =0; j < m; ++j)
      res += D[index+j*k] * D[index+j*k];
    diagDtD[index] = res;
  }
}

template <typename Dtype>
__global__ void kernel_vector_to_column(int m, int k, int j, const Dtype* v,
      Dtype* A) {
  CUDA_KERNEL_LOOP(index, m) {
    A[j+index*k] = v[index];
  }
}

template <typename Dtype>
__global__ void kernel_column_to_vector(int m, int k, int j, const Dtype* A,
      Dtype* v) {
  CUDA_KERNEL_LOOP(index, m) {
    v[index] = A[j+index*k];
  }
}

// Versions of caffe_gpu_scal that take scale factor as reference, so we can
// pass a pointer to the GPU.
// Remember, before calling this function, we need to call
//    cublasSetPointerMode(CUBLAS_POINTER_MODE_DEVICE);
// and, afterwards, we need to call
//    cublasSetPointerMode(CUBLAS_POINTER_MODE_HOST);
template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype* alpha, Dtype *X);

template <>
void caffe_gpu_scal<float>(const int N, const float* alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double* alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, alpha, X, 1));
}

// Normalize if norm > 1
template <typename Dtype>
__global__ void kernel_conditional_normalize(const int N, const Dtype* norm, Dtype* v) {
  CUDA_KERNEL_LOOP(index, N) {
    if (*norm > (Dtype)1.)
      v[index] /= sqrt(*norm);
  }
}

// Swap 2 columns in a matrix
template <typename Dtype>
__global__ void kernel_swap_columns(int m, int k, int j0, int j1, Dtype* A) {
  CUDA_KERNEL_LOOP(index, m) {
    Dtype tmp = A[j0+index*k];
    A[j0+index*k] = A[j1+index*k];
    A[j1+index*k] = tmp;
  }
}

// Swap 2 vectors
template <typename Dtype>
__global__ void kernel_swap_vectors(const int N, Dtype* v0, Dtype* v1) {
  CUDA_KERNEL_LOOP(index, N) {
    Dtype tmp = v0[index];
    v0[index] = v1[index];
    v1[index] = tmp;
  }
}


template <typename Dtype>
void DictionaryLayer<Dtype>::normalize_dictionary_gpu(int m, int k, Dtype* D) {
  // Normalize in a temporary matrix Z (D transposed)
  Dtype* Z = Z_buffer_.mutable_gpu_data();
  transpose_gpu(m, k, D, Z);
  // Normalize columns whose norm is greater than 1
  Dtype* norm = tmp_buffer_.mutable_gpu_data();
  // Tell cuBLAS that vector "norm" is in device memory
  cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
  for (int i = 0; i < k; ++i) {
    Dtype* vi = Z + i*m;
    caffe_gpu_dot(m, vi, vi, norm);
    kernel_conditional_normalize<Dtype><<<CAFFE_GET_BLOCKS(m),
        CAFFE_CUDA_NUM_THREADS>>>(m, norm, vi);
  }
  // Switch pointer mode back to host
  cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_HOST);
  transpose_gpu(k, m, Z, D);
}

// Same as forward_preprocess_cpu, but assuming that matrix Vt has been computed
// in previous iteration
template <typename Dtype>
void DictionaryLayer<Dtype>::fast_preprocess_gpu() {
  Dtype* Dflag = this->blobs_[bias_idx_ + 1]->mutable_cpu_data();
  int m = kernel_dim_;
  int k = num_output_;
  int r = rank_;
  // Normalize dictionary (make sure that the norm for each column is <= 1)
  // Orthonormalize dictionary (make sure that D^T*D=diag(D^T*D))
  if (!is_dict_normalized_ || (*Dflag)) {
    Dtype* D = this->blobs_[0]->mutable_gpu_data();
    if (orthogonalize_)
      NOT_IMPLEMENTED; //orthogonalize_dictionary_gpu(m, k, D, &dict_order_[0]);
    else
      normalize_dictionary_gpu(m, k, D);
    // Precompute SVD and pseudoinverse of D
    // We assume that matrix D has not changed much since previous iteration,
    // so we use previous vectors in Vt and refine
    // Note: we use Ddagger as temporary storage
    Dtype* work = Ddagger_buffer_.mutable_gpu_data(); // mxk
    Dtype* tmp = tmp_buffer_.mutable_gpu_data();
    caffe_gpu_memcpy(m*k*sizeof(Dtype), D, work);
    // Low-rank approximation on D
    Dtype* W = SV_buffer_.mutable_gpu_data(); // rx1
    Dtype* U = U_buffer_.mutable_gpu_data();  // mxr
    Dtype* Vt = this->blobs_[bias_idx_ + 2]->mutable_gpu_data();  // r*k
    Dtype* u = tmp;
    Dtype* prev_u = tmp + m;
    vector<Dtype> hostW(r);
    for (int ri = 0; ri < r; ++ri) {
      // Copy column ri of U into u
      kernel_column_to_vector<Dtype><<<CAFFE_GET_BLOCKS(m),
          CAFFE_CUDA_NUM_THREADS>>>(m, r, ri, U, u);
      Dtype* v = Vt + ri*k;
      const int max_iter = 10;
      for (int iter = 0; iter < max_iter; ++iter) {
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, m, 1, k, (Dtype)1., work, v,
            (Dtype)0., u);
        // Tell cuBLAS that vector W is in device memory
        cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
        caffe_gpu_dot<Dtype>(m, u, u, &W[ri]);
        // Do W[ri] = 1/sqrt(W[ri]) on device memory
        caffe_gpu_powx(1, &W[ri], -(Dtype)0.5, &W[ri]);
        caffe_gpu_scal(m, &W[ri], u);
        // Copy u back into corresponding column in U
        kernel_vector_to_column<Dtype><<<CAFFE_GET_BLOCKS(m),
            CAFFE_CUDA_NUM_THREADS>>>(m, r, ri, u, U);
        // The constants provided to gemm are in host memory
        cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_HOST);
        caffe_gpu_gemm(CblasTrans, CblasNoTrans, k, 1, m, (Dtype)1., work, u,
            (Dtype)0., v);
        // Tell cuBLAS that vector W is in device memory
        cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
        caffe_gpu_dot<Dtype>(k, v, v, &W[ri]);
        // Do W[ri] = 1/sqrt(W[ri]) on device memory
        caffe_gpu_powx(1, &W[ri], -(Dtype)0.5, &W[ri]);
        caffe_gpu_scal(k, &W[ri], v);
        // Set W[ri] to s
        caffe_gpu_powx(1, &W[ri], -(Dtype)1., &W[ri]);
        // Switch pointer mode back to host
        cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_HOST);
        // Check for convergence
        caffe_gpu_sub(m, u, prev_u, prev_u);
        Dtype delta = (Dtype)0.;
        caffe_gpu_dot<Dtype>(m, prev_u, prev_u, &delta);
        delta /= m;
        if (delta < 2 * EPSILON || iter == max_iter-1) {
          //LOG(INFO) << "Converged after " << iter << " iterations, " <<
          //    " delta = " << delta;
          break;
        }
        caffe_gpu_memcpy(m*sizeof(Dtype), u, prev_u);
      }
      caffe_gpu_memcpy(sizeof(Dtype), &W[ri], &hostW[ri]);
      // Check that singular vectors are in non-increasing order
      int ri1 = ri;
      for (int i = 0; i < ri; ++i) {
        if (hostW[i] < hostW[ri]) {
          ri1 = i;
          break;
        }
      }
      if (ri1 != ri) {
        //LOG(INFO) << "Swapping vectors W[" << ri << "] = " << W[ri] <<
        //    " and W[" << ri1 << "] = " << W[ri1];
        std::swap(hostW[ri], hostW[ri1]);
        // Swap u[ri] and u[ri1]
        kernel_swap_columns<Dtype><<<CAFFE_GET_BLOCKS(m),
            CAFFE_CUDA_NUM_THREADS>>>(m, r, ri, ri1, U);
        kernel_swap_vectors<Dtype><<<CAFFE_GET_BLOCKS(k),
            CAFFE_CUDA_NUM_THREADS>>>(k, Vt + ri*k, Vt + ri1*k);
        // Inflate
        caffe_gpu_memcpy(m*k*sizeof(Dtype), D, work);
        // Re-deflate
        for (int i = 0; i < ri1; ++i) {
          // Copy column ri of U into u
          kernel_column_to_vector<Dtype><<<CAFFE_GET_BLOCKS(m),
              CAFFE_CUDA_NUM_THREADS>>>(m, r, i, U, u);
          Dtype* v = Vt + i*k;
          caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, m, k, 1, -(Dtype)hostW[i], u, v,
              (Dtype)1., work);
        }
        // Recompute new singular vector
        ri = ri1-1;
      }
      else {
        // Deflate
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, m, k, 1, -(Dtype)hostW[ri],
            u, v, (Dtype)1., work);
      }
    }
    // Reconstruct D from rank r approximation
    caffe_gpu_memcpy(r*k*sizeof(Dtype), Vt, tmp);
    for (int i = 0; i < r; ++i) {
      //hostW[i] = std::max(hostW[i], (Dtype)EPSILON);
      caffe_gpu_scal(k, hostW[i], tmp + i*k);
    }
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, m, k, r, (Dtype)1., U, tmp,
        (Dtype)0., D);
    // Precompute pseudoinverse of D
    Dtype* Ddagger = Ddagger_buffer_.mutable_gpu_data();
    Dtype* Vt_sn2 = Vt_sn2_buffer_.mutable_gpu_data();
    Dtype* Vt_s2 = this->blobs_[bias_idx_ + 3]->mutable_gpu_data();
    caffe_gpu_memcpy(r*k*sizeof(Dtype), Vt, Vt_sn2);
    caffe_gpu_memcpy(r*k*sizeof(Dtype), Vt, Vt_s2);
    for (int i = 0; i < r; ++i) {
      if (hostW[i] > EPSILON) {
        caffe_gpu_scal(k, (Dtype)1./(hostW[i]*hostW[i]), Vt_sn2 + i*k);
        caffe_gpu_scal(k, hostW[i]*hostW[i], Vt_s2 + i*k);
      }
      else {
        caffe_gpu_scal(k, (Dtype)0., Vt_sn2 + i*k);
        caffe_gpu_scal(k, (Dtype)0., Vt_s2 + i*k);
      }
    }
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, k, k, r, (Dtype)1., Vt, Vt_sn2,
        (Dtype)0., tmp);
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, m, k, k, (Dtype)1., D, tmp,
        (Dtype)0., Ddagger);
    is_dict_normalized_ = true;
    *Dflag = (Dtype)0.;
  }
}

template <typename Dtype>
void DictionaryLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //Forward_cpu(bottom, top);

  // Perform normalization and decomposition (if necessary)
  if (first_time_ || this->phase_ == TEST) {
    forward_preprocess_cpu();
    first_time_ = false;
  }
  else
    fast_preprocess_gpu();
  // Perform sparse coding (and optionally dictionary learning) on each input vector
  const Dtype* D = this->blobs_[0]->gpu_data();
  const Dtype* Vt = this->blobs_[bias_idx_ + 2]->gpu_data();
  const Dtype* Vt_s2 = this->blobs_[bias_idx_ + 3]->gpu_data();
  for (int i = 0; i < top.size()/2; ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[2*i]->mutable_gpu_data();
    double loss = 0.;
    //LOG(INFO) << "Input norm = " << sqrt(caffe_cpu_dot<Dtype>(bottom[i]->count(),
    //    bottom_data, bottom_data)/bottom[i]->count());
    for (int n = 0; n < this->num_; ++n) {
      // Perform forward sparse coding
      loss += this->forward_gpu_sparse_coding(bottom_data + bottom[i]->offset(n),
          D, Vt, Vt_s2, top_data + top[2*i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[bias_idx_]->gpu_data();
        this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
    // Put objective value in second output
    top_data = top[2*i+1]->mutable_cpu_data();
    *top_data = Dtype(loss/num_);
  }
}

// A is mxk, B is kxm
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
}

template <typename Dtype>
__global__ void kernel_compute_diag_weights(int n, Dtype lambda,
      const Dtype* vec_alpha, Dtype* diag) {
  CUDA_KERNEL_LOOP(index, n) {
    diag[index] = 2*lambda/(fabs(vec_alpha[index])+EPSILON);
  }
}

template <typename Dtype>
__global__ void kernel_hard_threshold(int n, Dtype lambda, Dtype* vec_alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    if (fabs(vec_alpha[index]) < lambda)
      vec_alpha[index] = (Dtype)0.;
  }
}

// Perform sparse coding on the GPU, estimate the loss and add it to the
// previous loss value
template <typename Dtype>
double DictionaryLayer<Dtype>::forward_gpu_sparse_coding(const Dtype* input,
      const Dtype* D, const Dtype* Vt, const Dtype *Vt_s2, Dtype* output,
      bool skip_im2col) {
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
  Dtype* sparse_codes = sparse_codes_buffer_.mutable_gpu_data();
  Dtype* diag = tmp_buffer_.mutable_gpu_data() + std::max(k,m);
  Dtype* tmp = tmp_buffer_.mutable_gpu_data() + 2*std::max(k,m);
  for (int i = 0; i < conv_out_spatial_dim_; ++i)
  {
    const Dtype* x = col_buff + i*m;  // Input sample
    Dtype* vec_alpha = sparse_codes + i*k;   // Sparse code vector
    // Initialize sparse code
    caffe_gpu_set<Dtype>(k, (Dtype)1., vec_alpha);
    // Perform num_iter_irls iterations of iteratively reweighted
    // least squares using the previous result as starting value
    for (int iter_irls = 0; iter_irls < num_iter_irls_; ++iter_irls)
    {
      // Build matrix w = diag(2*lambda/fabs(alpha[])
      kernel_compute_diag_weights<Dtype><<<CAFFE_GET_BLOCKS(k),
          CAFFE_CUDA_NUM_THREADS>>>(k, (Dtype)lambda_, vec_alpha, diag);
      // Build vector d = D^T * x
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, k, 1, m,
           (Dtype)1., D, x,
           (Dtype)0., vec_d);
      // Perform conjugate gradient descent to approximately solve
      // C * alpha = d     for alpha
      // Note: We do not compute matrix C explicitly, since
      //   C * alpha = diag(2*lambda/fabs(alpha[]) .* alpha + V * Vt_s2 * alpha
      //   C * alpha = tmp + V * Vt_s2 * alpha
      // is more efficient to compute
      conjugate_gradient_gpu(k, rank_, diag, Vt, Vt_s2, vec_d, vec_alpha,
          num_iter_cg_, vec_p, vec_r, vec_w, tmp);
    }
    // Apply hard threshold to sparse codes
    kernel_hard_threshold<<<CAFFE_GET_BLOCKS(k), CAFFE_CUDA_NUM_THREADS>>>(
        k, (Dtype)lambda_, vec_alpha);
    //add_objective_gpu(m, k, D, x, vec_alpha, loss);
  }
  // Sparse codes are in pixel-first order, we need to transpose them so they
  // are in channel-first order
  transpose_gpu(conv_out_spatial_dim_, num_output_, sparse_codes, output);

  return 0.;
}

template <typename Dtype>
void DictionaryLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  int k = num_output_;
  caffe_gpu_add(k, bias, output, output);
}

template <typename Dtype>
void DictionaryLayer<Dtype>::add_objective_gpu(int m, int k,
        const Dtype* D, const Dtype* x, const Dtype* alpha,
        Dtype* loss) {
  // Compute objective function
  // Cost(alpha) = 0.5*||x_t-D*alpha||_2^2 + lambda*||alpha||_1
  Dtype* tmp = tmp_buffer_.mutable_gpu_data();
  caffe_gpu_memcpy(m, x, tmp);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, 1, k,
                        (Dtype)(-1.), D, alpha,
                        (Dtype)1., tmp);
  Dtype cost = (Dtype)0.;
  //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, 1, m, (Dtype)1., tmp,
  //    tmp, (Dtype)0., cost);
  caffe_gpu_dot<Dtype>(m, tmp, tmp, &cost);
  //caffe_gpu_scale<Dtype>(1, 0.5, cost, cost);
  Dtype asum = (Dtype)0.;
  // Sum of absolute values of elements in alpha
  caffe_gpu_asum<Dtype>(k, alpha, &asum);
  //caffe_gpu_scale<Dtype>(1, Dtype(lambda_), asum, asum);
  // Add cost and asum to loss
  //caffe_gpu_add<Dtype>(1, asum, loss, loss);
  //caffe_gpu_add<Dtype>(1, cost, loss, loss);
  cost = 0.5 * cost + lambda_ * asum;
  //LOG(INFO) << "COST = " << cost;
  caffe_gpu_add_scalar(1, cost, loss);
}

// Compute C * x = w .* x + V * Vt2 * x
//         (kx1)    (kx1)  (k*r)(r*k)(k*1)
template <typename Dtype>
void DictionaryLayer<Dtype>::compute_Cx_gpu(int k, int r, const Dtype* w,
      const Dtype* Vt, const Dtype* Vt2, const Dtype* x,
      Dtype* tmp, Dtype* Cx) {
  caffe_gpu_mul(k, w, x, Cx);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, r, k, (Dtype)1., Vt2, x, (Dtype)0.,
      tmp);
  caffe_gpu_gemv<Dtype>(CblasTrans, r, k, (Dtype)1., Vt, tmp, (Dtype)1., Cx);
}

// Perform one step of CGD in a single kernel
template <typename Dtype>
__global__ void kernel_step_cg(int k, Dtype* w, Dtype* p, Dtype* r,
      Dtype* x, Dtype* ret_norm_r) {
  __shared__ Dtype prev_norm_r;
  __shared__ Dtype dot_p_w[1024];
  __shared__ Dtype norm_r[1024];
  if (threadIdx.x == 0)
    prev_norm_r = *ret_norm_r;
  // Compute dot_p_w
  dot_p_w[threadIdx.x] = (Dtype)0.;
  CUDA_KERNEL_LOOP(j, k) {
    dot_p_w[threadIdx.x] += w[j]*p[j];
  }
  __syncthreads();
  // Reduce sum and leave result in dot_p_w[0]
  if (threadIdx.x < 512) dot_p_w[threadIdx.x] += dot_p_w[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) dot_p_w[threadIdx.x] += dot_p_w[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) dot_p_w[threadIdx.x] += dot_p_w[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x < 64) dot_p_w[threadIdx.x] += dot_p_w[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x < 32) dot_p_w[threadIdx.x] += dot_p_w[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x < 16) dot_p_w[threadIdx.x] += dot_p_w[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x < 8) dot_p_w[threadIdx.x] += dot_p_w[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x < 4) dot_p_w[threadIdx.x] += dot_p_w[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x < 2) dot_p_w[threadIdx.x] += dot_p_w[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x < 1) dot_p_w[threadIdx.x] += dot_p_w[threadIdx.x + 1];
  __syncthreads();
  // Compute alpha
  Dtype alpha = prev_norm_r / dot_p_w[0];
  // Update x and r
  CUDA_KERNEL_LOOP(j, k) {
    x[j] += alpha*p[j];
    r[j] -= alpha*w[j];
  }
  __syncthreads();
  // Compute norm of r
  norm_r[threadIdx.x] = (Dtype)0.;
  CUDA_KERNEL_LOOP(j, k) {
    norm_r[threadIdx.x] += r[j]*r[j];
  }
  // Reduce sum and leave result in norm_r[0]
  if (threadIdx.x < 512) norm_r[threadIdx.x] += norm_r[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) norm_r[threadIdx.x] += norm_r[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) norm_r[threadIdx.x] += norm_r[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x < 64) norm_r[threadIdx.x] += norm_r[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x < 32) norm_r[threadIdx.x] += norm_r[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x < 16) norm_r[threadIdx.x] += norm_r[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x < 8) norm_r[threadIdx.x] += norm_r[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x < 4) norm_r[threadIdx.x] += norm_r[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x < 2) norm_r[threadIdx.x] += norm_r[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x < 1) norm_r[threadIdx.x] += norm_r[threadIdx.x + 1];
  __syncthreads();
  // Compute beta
  Dtype beta = norm_r[0] / prev_norm_r;
  // Update p
  CUDA_KERNEL_LOOP(j, k) {
    p[j] = r[j] + beta*p[j];
  }
  // Return new norm of r
  if (threadIdx.x == 0)
    *ret_norm_r = norm_r[0];
}

template <typename Dtype>
void DictionaryLayer<Dtype>::conjugate_gradient_gpu(int k, int r,
      const Dtype* weights, const Dtype* Vt, const Dtype* Vt2, const Dtype* d,
      Dtype* x, int num_iter, Dtype* temp_p, Dtype* temp_r, Dtype* temp_w,
      Dtype* tmp) {
  // Temporay scalar variables on GPU
  thrust::device_ptr<Dtype> norm_r(tmp++);
  // Initialize the residual
  compute_Cx_gpu(k, r, weights, Vt, Vt2, x, tmp, temp_r);
  caffe_gpu_sub(k, d, temp_r, temp_r);
  // Compute norm of the residual
  cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
  caffe_gpu_dot<Dtype>(k, temp_r, temp_r, norm_r.get());
  cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_HOST);
  if (fabs(*norm_r) < EPSILON) {
    return;   // Accept initial solution
  }
  // Initialize the descent direction
  caffe_gpu_memcpy(k*sizeof(Dtype), temp_r, temp_p);
  // Perform num_iter_cg iterations of conjugate gradient descent
  for (int iter_cg = 0; iter_cg < num_iter; ++iter_cg)
  {
    // w = C * p
    compute_Cx_gpu(k, r, weights, Vt, Vt2, temp_p, tmp, temp_w);
    // Invoke kernel that does
    //   dot_p_w = sum_j(p[j]*w[j])
    //   alpha = prev_norm_r / dot_p_w;
    //   x = x + alpha*p
    //   r = r - alpha*w
    //   norm_r = sum_j(r[j]*r[j])
    //   beta = norm_r / prev_norm_r
    //   p = r + beta*p
    // and returns norm_r
    // Our kernel only has 1 block to allow thread synchronization
    kernel_step_cg<Dtype><<<1, 1024>>>(k, temp_w, temp_p,
        temp_r, x, norm_r.get());
    if (fabs(*norm_r) < EPSILON) {
      return;
    }
  }
}

template <typename Dtype>
__global__ void kernel_mod_gradient(int k, const Dtype* alpha,
    const Dtype* alpha_diff, Dtype* mod_alpha_diff) {
  CUDA_KERNEL_LOOP(index, k) {
    mod_alpha_diff[index] = alpha[index] == (Dtype)0.
        ? (Dtype)0. : alpha_diff[index];
  }
}

template <typename Dtype>
void DictionaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//  Backward_cpu(top, propagate_down, bottom);
  const Dtype* D = this->blobs_[0]->gpu_data();
  Dtype* D_diff = this->blobs_[0]->mutable_gpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), D_diff);
  }
  CHECK(is_dict_normalized_);
  CHECK_EQ(conv_out_spatial_dim_, 1) << "Convolutional dictionaries not implemented, yet!";
  // Temporary storage and precomputed constants
  int m = kernel_dim_;
  int k = num_output_;
  Dtype* tmp1 = tmp_buffer_.mutable_gpu_data();
  Dtype* tmp2 = tmp_buffer_.mutable_gpu_data() + std::max(k,m);
  Dtype* tmp3 = tmp_buffer_.mutable_gpu_data() + 2*std::max(k,m);
  Dtype* tmp_dl_dx = tmp_buffer_.mutable_gpu_data() + 3*std::max(k,m);
  // Precomputed matrices
  const Dtype* Vt = this->blobs_[bias_idx_ + 2]->gpu_data();
  const Dtype* Vt_sn2 = Vt_sn2_buffer_.gpu_data();
  const Dtype* Ddagger = Ddagger_buffer_.gpu_data();
  for (int idx = 0; idx < top.size()/2; ++idx) {
    const Dtype* top_diff = top[2*idx]->gpu_diff();
    const Dtype* top_data = top[2*idx]->gpu_data();
    const Dtype* bottom_data = bottom[idx]->gpu_data();
    Dtype* bottom_diff = bottom[idx]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[bias_idx_]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[idx*2]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[idx]) {
      for (int n = 0; n < this->num_; ++n) {
        const Dtype* alpha = top_data + top[idx*2]->offset(n);
        const Dtype* alpha_diff = top_diff + top[idx*2]->offset(n);
        // Precompute modified output gradient
        Dtype* mod_alpha_diff = mod_alpha_diff_buffer_.mutable_gpu_data();
        kernel_mod_gradient<Dtype><<<CAFFE_GET_BLOCKS(k),
            CAFFE_CUDA_NUM_THREADS>>>(k, alpha, alpha_diff, mod_alpha_diff);
        // dl/dx is necessary for both gradients
        Dtype* dl_dx = propagate_down[idx] ?
              bottom_diff + bottom[idx]->offset(n) : tmp_dl_dx;
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[idx] || this->param_propagate_down_[0]) {
          this->backward_gpu_gemm(mod_alpha_diff, Ddagger, dl_dx);
        }
        // gradient w.r.t. dictionary. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->dict_gpu_backprop(bottom_data + bottom[idx]->offset(n),
              dl_dx, mod_alpha_diff, Ddagger, Vt, Vt_sn2, tmp1, tmp2, tmp3,
              D_diff);
          this->dict_gpu_optimize(bottom_data + bottom[idx]->offset(n), alpha,
              D, (Dtype)etha_, tmp1, tmp2, D_diff);
          // Mark dictionary as unnormalized
          is_dict_normalized_ = false;
          Dtype* Dflag = this->blobs_[bias_idx_ + 1]->mutable_cpu_data();
          *Dflag = (Dtype)1.;
        }
        // gradient of reconstruction error w.r.t bottom data, if necessary
        if (propagate_down[idx]) {
          this->backward_gpu_optimize(alpha, D,
              bottom_data + bottom[idx]->offset(n),
              (Dtype)(etha_ * this->layer_param_.param(0).lr_mult()), dl_dx);
        }
      }
    }
  }
}

template <typename Dtype>
void DictionaryLayer<Dtype>::dict_gpu_backprop(const Dtype* x, const Dtype* dl_dx,
      const Dtype* mod_alpha_diff, const Dtype* Dtdagger, const Dtype* Vt,
      const Dtype* Vt_sn2, Dtype* tmp1, Dtype* tmp2, Dtype* tmp3, Dtype* D_diff) {
  int m = kernel_dim_;
  int k = num_output_;
  int r = rank_;
  // Compute intermediate products
  // tmp1 = x^T * (D^dagger)^T
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, k, m, (Dtype)1., x,
      Dtdagger, (Dtype)0., tmp1);
  // tmp2 = dl_dalpha * V
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, r, k, (Dtype)1.,
      mod_alpha_diff, Vt, (Dtype)0., tmp2);
  // tmp3 = tmp2 * Vt_sn2
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, k, r, (Dtype)1.,
      tmp2, Vt_sn2, (Dtype)0., tmp3);
  // Compute gradient of dictionary and add it to D_diff
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, k, 1, -(Dtype)2., dl_dx,
      tmp1, (Dtype)1., D_diff);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, k, 1, (Dtype)1., x,
      tmp3, (Dtype)1., D_diff);
  // Result to be D_diff = D_diff - 2 * (D^dagger)^T * x^T * mod_alpha_diff^t * (D^dagger)^T
  //                              + x * mod_alpha_diff * V * Vt_sn2

}

// Gradient that decreases reconstruction loss on dictionary
template <typename Dtype>
void DictionaryLayer<Dtype>::dict_gpu_optimize(const Dtype* x,
    const Dtype* alpha, const Dtype* D, Dtype etha, Dtype* tmp1, Dtype* tmp2,
    Dtype* D_diff) {
  if (etha == (Dtype)0.)
    return;
  // Do dl/dD += etha*(D*alpha-x)*alpha^T
  int m = kernel_dim_;
  int k = num_output_;
  caffe_copy(m, x, tmp1);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, 1, k,
      (Dtype)1., D, alpha, -(Dtype)1., tmp1);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, k, 1,
      etha, tmp1, alpha, (Dtype)1., D_diff);
}

// Compute backpropagated reconstruction loss and add it to dl_dx
template <typename Dtype>
void DictionaryLayer<Dtype>::backward_gpu_optimize(const Dtype* alpha,
    const Dtype* D, const Dtype* x, Dtype etha, Dtype* dl_dx) {
  if (etha == (Dtype)0.)
    return;
  // Do dl/dx += etha*(x-D*alpha)
  int m = kernel_dim_;
  int k = num_output_;
  caffe_gpu_axpy(m, etha, x, dl_dx);
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, m, 1, k, -etha, D, alpha,
      (Dtype)1., dl_dx);
}

template <typename Dtype>
void DictionaryLayer<Dtype>::backward_gpu_gemm(const Dtype* mod_alpha_diff,
    const Dtype* D, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, kernel_dim_,
      1, num_output_,
      (Dtype)1., D, mod_alpha_diff,
      (Dtype)0., col_buff);
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void DictionaryLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  int k = num_output_;
  caffe_gpu_add(k, input, bias, bias);
}

INSTANTIATE_LAYER_GPU_FUNCS(DictionaryLayer);

}  // namespace caffe
