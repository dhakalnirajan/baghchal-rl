/*****************************************************************************
 *  File: alphabaghchal.h
 *  Author: Nirajan Dhakal
 *  Date: March 12, 2025
 *  License: MIT License
 *
 *  Description:
 *  This header file defines the data structures, constants, and function
 *  prototypes for the Baghchal game implementation using an AlphaZero-like
 *  approach. It includes definitions for the game state, neural network structure,
 *  MCTS parameters, and function declarations for various operations including
 *  neural network initialization, game management, forward passes, backpropagation,
 *  and MCTS search. This file provides the necessary interfaces for the
 *  implementation, making it a central component of the project.
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


#ifndef ALPHABAGHCHAL_H
#define ALPHABAGHCHAL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

// --- 1. Game Specific Constants ---
#define BOARD_SIZE 25
#define TIGER_COUNT 4
#define GOAT_COUNT 20

// --- 2. Neural Network Parameters ---
#define NN_INPUT_SIZE (BOARD_SIZE * 2)
#define NN_HIDDEN_SIZE 128
#define NN_OUTPUT_SIZE BOARD_SIZE
#define LEARNING_RATE 0.01
#define NN_BATCH_SIZE 64

// --- 3. MCTS Parameters ---
#define MCTS_SIMULATIONS 100
#define CPUCT 1.414f

// --- 4. Game State ---
typedef struct {
    int board[BOARD_SIZE];
    int tigers_on_board;
    int goats_on_board;
    int current_player;
    int placement_phase;
} GameState;

// --- 5. Neural Network Structure ---
typedef struct {
  float *weights_ih;
  float *weights_ho;
  float *biases_h;
  float *biases_o;
  float *inputs;
  float *hidden;
  float *raw_logits;
  float *policy_probs;
  float *value;
} NeuralNetwork;


// --- 6. Helper function prototypes ----
void init_neural_network(NeuralNetwork *nn);
void softmax(float *input, float *output, int size);
void forward_pass(NeuralNetwork *nn, float *inputs);
void init_game(GameState *state);
void display_board(GameState *state);
void board_to_inputs(GameState *state, float *inputs);
int check_game_over(GameState *state, char *winner);
int get_computer_move(GameState *state, NeuralNetwork *nn, int display_probs);
void backprop(NeuralNetwork *nn, float *target_probs, float *target_value, float learning_rate);
void learn_from_game(NeuralNetwork *nn, int *move_history, int num_moves, int nn_moves_even, char winner, float *values_history);
void play_game(NeuralNetwork *nn);
int get_random_move(GameState *state);
char play_random_game(NeuralNetwork *nn, int *move_history, int *num_moves, float *values_history);
void train_against_random(NeuralNetwork *nn, int num_games);
int get_valid_moves(GameState *state, int player, int moves[BOARD_SIZE], int *num_moves);
int mcts_search(GameState *state, NeuralNetwork *nn);

#endif