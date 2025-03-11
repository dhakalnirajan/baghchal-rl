#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Define the constants again to use them in the CUDA kernels.
#define BOARD_SIZE 25
#define NN_INPUT_SIZE (BOARD_SIZE * 2)
#define NN_HIDDEN_SIZE 100
#define NN_OUTPUT_SIZE BOARD_SIZE

__device__ float relu(float x) { return x > 0 ? x : 0; }
__device__ float relu_derivative(float x) { return x > 0 ? 1.0f : 0.0f; }

__global__ void cuda_forward_kernel(float *weights_ih, float *weights_ho, float *biases_h, float *biases_o,
                            float *inputs, float *hidden, float *raw_logits, float *outputs,
                            int input_size, int hidden_size, int output_size) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < hidden_size) {
        float sum = biases_h[i];
        for (int j = 0; j < input_size; j++) {
          sum += inputs[j] * weights_ih[j * hidden_size + i];
        }
        hidden[i] = relu(sum);
    }

    __syncthreads();

    if(i < output_size){
        float sum = biases_o[i];
        for(int j =0; j<hidden_size; j++){
          sum+= hidden[j] * weights_ho[j*output_size + i];
        }
          raw_logits[i]=sum;
        }

    __syncthreads();
    
     if (i < output_size) {
      // Softmax calculation in a separate kernel.
      float max_val = raw_logits[0];
        for (int j = 1; j < output_size; j++) {
          if (raw_logits[j] > max_val) {
              max_val = raw_logits[j];
          }
        }

        float sum = 0.0f;
        for (int j = 0; j < output_size; j++) {
        outputs[j] = expf(raw_logits[j] - max_val);
          sum += outputs[j];
        }


        if (sum > 0) {
          for (int j = 0; j < output_size; j++) {
            outputs[j] /= sum;
          }
        } else {
          for (int j = 0; j < output_size; j++) {
            outputs[j] = 1.0f / output_size;
          }
        }
     }


}


__global__ void cuda_backprop_kernel(float *weights_ih, float *weights_ho, float *biases_h, float *biases_o,
                                float *inputs, float *hidden, float *outputs, float *target_probs,
                                  float learning_rate, float reward_scaling, int input_size, int hidden_size, int output_size){
  int i = blockIdx.x * blockDim.x + threadIdx.x;


  __shared__ float output_deltas[BOARD_SIZE];
    if(i < output_size){
        output_deltas[i] = (outputs[i] - target_probs[i]) * fabsf(reward_scaling);
    }
  __syncthreads();


    __shared__ float hidden_deltas[NN_HIDDEN_SIZE];
    if (i < hidden_size) {
        float error = 0;
        for (int j = 0; j < output_size; j++) {
            error += output_deltas[j] * weights_ho[i * output_size + j];
        }
        hidden_deltas[i] = error * relu_derivative(hidden[i]);
    }
      __syncthreads();

    if(i< hidden_size){
        for (int j = 0; j < output_size; j++) {
            weights_ho[i * output_size + j] -= learning_rate * output_deltas[j] * hidden[i];
        }
    }
        __syncthreads();
  if(i < output_size)
    biases_o[i] -= learning_rate * output_deltas[i];

    __syncthreads();

    if(i< input_size){
        for (int j = 0; j < hidden_size; j++) {
           weights_ih[i * hidden_size + j] -= learning_rate * hidden_deltas[j] * inputs[i];
       }
    }
      __syncthreads();
    if(i<hidden_size)
        biases_h[i] -= learning_rate * hidden_deltas[i];
}

void cuda_forward_kernel_wrapper(float *weights_ih, float *weights_ho, float *biases_h, float *biases_o,
                            float *inputs, float *hidden, float *raw_logits, float *outputs,
                            int input_size, int hidden_size, int output_size){
    dim3 threadsPerBlock(128);
    dim3 numBlocks((hidden_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
     
    cuda_forward_kernel<<<numBlocks, threadsPerBlock>>>(weights_ih, weights_ho, biases_h, biases_o,
                             inputs, hidden, raw_logits, outputs,
                             input_size, hidden_size, output_size);
     cudaDeviceSynchronize();
}

void cuda_backprop_kernel_wrapper(float *weights_ih, float *weights_ho, float *biases_h, float *biases_o,
                            float *inputs, float *hidden, float *outputs, float *target_probs,
                                  float learning_rate, float reward_scaling, int input_size, int hidden_size, int output_size){
    dim3 threadsPerBlock(128);
    dim3 numBlocks((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    cuda_backprop_kernel<<<numBlocks, threadsPerBlock>>>(weights_ih, weights_ho, biases_h, biases_o, inputs, hidden, outputs, target_probs,
    learning_rate, reward_scaling, input_size, hidden_size, output_size);

        cudaDeviceSynchronize();
}

// Extern C wrapper for the C code to call cuda kernels.
extern "C" {

  void cuda_forward_kernel(float *nn, int input_size, int hidden_size, int output_size) {
       cuda_forward_kernel_wrapper(((float*)nn), ((float*)nn + (input_size * hidden_size)),
       ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size)),
       ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size),
       ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size + output_size),
       ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size + output_size + input_size),
        ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size + output_size + input_size + hidden_size),
         ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size + output_size + input_size + hidden_size + output_size),
        input_size, hidden_size, output_size
       );
  }
   void cuda_backprop_kernel(float *nn, float* target_probs, float learning_rate, float reward_scaling, int input_size, int hidden_size, int output_size)
  {
    cuda_backprop_kernel_wrapper(((float*)nn), ((float*)nn + (input_size * hidden_size)),
       ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size)),
       ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size),
       ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size + output_size),
        target_probs,
        learning_rate,
        reward_scaling,
        input_size,
        hidden_size,
        output_size
        );
  }
}