/*****************************************************************************
 *  File: alphabaghchal_cuda.cu
 *  Author: Nirajan Dhakal
 *  Date: March 12, 2025
 *  License: MIT License
 *
 *  Description:
 *  This CUDA C++ file implements kernels for an AlphaZero-like neural network
 *  architecture designed for the Baghchal game. It provides CUDA accelerated
 *  forward and backpropagation operations, utilizing a shared memory for efficient
 *  communication within thread blocks. The forward pass computes policy
 *  probabilities for each move, along with a value estimate for the current
 *  game state. Backpropagation then updates the network weights based on
 *  target probabilities and value estimates. This implementation uses a
 *  shared memory for efficient computation and optimization.
 *
 *  Fair Use Notice:
 *  The code provided in this file is intended for educational and
 *  non-commercial use. You are free to use, modify, and distribute this
 *  code, provided you retain the above copyright notice and this
 *  fair use notice. Any usage for commercial purposes requires
 *  explicit permission from the original author.
 *
 *  Note: This implementation is a simplified representation and may
 *        not be suitable for production or large-scale applications.
 *
 *****************************************************************************/

 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <stdio.h>
 
 // Define the constants again to use them in the CUDA kernels.
 #define BOARD_SIZE 25
 #define NN_INPUT_SIZE (BOARD_SIZE * 2)
 #define NN_HIDDEN_SIZE 128
 #define NN_OUTPUT_SIZE BOARD_SIZE
 #define NN_BATCH_SIZE 64
 
 __device__ float relu(float x) { return x > 0 ? x : 0; }
 __device__ float relu_derivative(float x) { return x > 0 ? 1.0f : 0.0f; }
 
 __global__ void cuda_forward_kernel(float *weights_ih, float *weights_ho, float *biases_h, float *biases_o,
                             float *inputs, float *hidden, float *raw_logits, float *policy_probs, float *value,
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
           policy_probs[j] = expf(raw_logits[j] - max_val);
           sum += policy_probs[j];
         }
 
         if (sum > 0) {
           for (int j = 0; j < output_size; j++) {
             policy_probs[j] /= sum;
           }
         } else {
           for (int j = 0; j < output_size; j++) {
             policy_probs[j] = 1.0f / output_size;
           }
         }
      }
     __syncthreads();
     if(i==0){
       float sum = 0;
        for(int j=0; j< hidden_size; j++){
             sum+=hidden[j];
        }
        *value = tanh(sum);
     }
 }
 
 
 __global__ void cuda_backprop_kernel(float *weights_ih, float *weights_ho, float *biases_h, float *biases_o,
                                 float *inputs, float *hidden, float *policy_probs, float *value, float *target_probs, float *target_value,
                                   float learning_rate, int input_size, int hidden_size, int output_size){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
 
     __shared__ float value_delta;
     if (i==0){
       value_delta = *value - *target_value;
     }
       __syncthreads();
 
   __shared__ float output_deltas[NN_OUTPUT_SIZE];
     if(i < output_size){
         output_deltas[i] = (policy_probs[i] - target_probs[i]);
     }
   __syncthreads();
 
     __shared__ float hidden_deltas[NN_HIDDEN_SIZE];
     if (i < hidden_size) {
         float error = 0;
         for (int j = 0; j < output_size; j++) {
             error += output_deltas[j] * weights_ho[i * output_size + j];
         }
         error +=  value_delta;
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
                             float *inputs, float *hidden, float *raw_logits, float *policy_probs, float *value,
                             int input_size, int hidden_size, int output_size){
     dim3 threadsPerBlock(128);
     dim3 numBlocks((hidden_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
      
     cuda_forward_kernel<<<numBlocks, threadsPerBlock>>>(weights_ih, weights_ho, biases_h, biases_o,
                              inputs, hidden, raw_logits, policy_probs, value,
                              input_size, hidden_size, output_size);
      cudaDeviceSynchronize();
 }
 
 void cuda_backprop_kernel_wrapper(float *weights_ih, float *weights_ho, float *biases_h, float *biases_o,
                             float *inputs, float *hidden, float *policy_probs, float *value, float *target_probs, float *target_value,
                                   float learning_rate, int input_size, int hidden_size, int output_size){
     dim3 threadsPerBlock(128);
     dim3 numBlocks((hidden_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
     
     cuda_backprop_kernel<<<numBlocks, threadsPerBlock>>>(weights_ih, weights_ho, biases_h, biases_o, inputs, hidden, policy_probs, value, target_probs, target_value,
     learning_rate, input_size, hidden_size, output_size);
 
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
    void cuda_backprop_kernel(float *nn, float* target_probs, float *target_value, float learning_rate, int input_size, int hidden_size, int output_size)
   {
     cuda_backprop_kernel_wrapper(((float*)nn), ((float*)nn + (input_size * hidden_size)),
        ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size)),
        ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size),
        ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size + output_size),
         ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size + output_size + input_size),
         ((float*)nn + (input_size * hidden_size) + (hidden_size * output_size) + hidden_size + output_size + input_size + hidden_size),
         target_probs, target_value,
         learning_rate,
         input_size,
         hidden_size,
         output_size
         );
   }
 }