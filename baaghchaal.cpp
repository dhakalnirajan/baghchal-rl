/*****************************************************************************
 *  File: baaghchaal.cpp
 *  Author: Nirajan Dhakal
 *  Date: March 12, 2025
 *  License: MIT License
 *
 *  Description:
 *  This C++ file implements a simplified version of the Baghchal game, also known as
 *  Goat and Tigers, using a neural network to control the Tiger's moves. The game
 *  allows for both computer vs. random play and human vs. computer gameplay. This
 *  implementation includes features such as a neural network with backpropagation
 *  training, ASCII art board display, and logic for move validation and game-over
 *  conditions. The neural network is trained through self-play against a random
 *  move selection.
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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <math.h>

// --- 1. Game Specific Constants ---
#define BOARD_SIZE 25    // Number  of intersections
#define TIGER_COUNT 4
#define GOAT_COUNT 20

// --- 2. Neural Network Parameters ---
#define NN_INPUT_SIZE (BOARD_SIZE * 2)  // Two bits per cell (empty, goat, tiger)
#define NN_HIDDEN_SIZE 100
#define NN_OUTPUT_SIZE (BOARD_SIZE) // Maximum moves are limited in this example.
#define LEARNING_RATE 0.1

// --- 3. Game State ---
typedef struct {
    int board[BOARD_SIZE];  // 0: empty, 1: goat, 2: tiger
    int tigers_on_board;    // Count of tigers on the board
    int goats_on_board;     // Count of goats on the board
    int current_player;     // 0: goat, 1: tiger
    int placement_phase;    // 1: during the placement phase, 0 otherwise.
} GameState;

// --- 4. Neural Network Structure ---
typedef struct {
    float weights_ih[NN_INPUT_SIZE * NN_HIDDEN_SIZE];
    float weights_ho[NN_HIDDEN_SIZE * NN_OUTPUT_SIZE];
    float biases_h[NN_HIDDEN_SIZE];
    float biases_o[NN_OUTPUT_SIZE];

    float inputs[NN_INPUT_SIZE];
    float hidden[NN_HIDDEN_SIZE];
    float raw_logits[NN_OUTPUT_SIZE];
    float outputs[NN_OUTPUT_SIZE];
} NeuralNetwork;

// --- 5. Activation Functions ---
float relu(float x) { return x > 0 ? x : 0; }
float relu_derivative(float x) { return x > 0 ? 1.0f : 0.0f; }

// --- 6. Neural Network Initialization ---
#define RANDOM_WEIGHT() (((float)rand() / RAND_MAX) - 0.5f)
void init_neural_network(NeuralNetwork *nn);

// --- 7. Softmax Function ---
void softmax(float *input, float *output, int size);

// --- 8. Forward Pass ---
void forward_pass(NeuralNetwork *nn, float *inputs);

// --- 9. Game Initialization ---
void init_game(GameState *state);

// --- 10. Display Board (ASCII Art) ---
void display_board(GameState *state);

// --- 11. Board to Input ---
void board_to_inputs(GameState *state, float *inputs);

// --- 12. Check Game Over ---
int check_game_over(GameState *state, char *winner);

// --- 13. Get Computer Move ---
int get_computer_move(GameState *state, NeuralNetwork *nn, int display_probs);

// --- 14. Backpropagation ---
void backprop(NeuralNetwork *nn, float *target_probs, float learning_rate, float reward_scaling);

// --- 15. Learn From Game ---
void learn_from_game(NeuralNetwork *nn, int *move_history, int num_moves, int nn_moves_even, char winner);

// --- 16. Play Game Against Human ---
void play_game(NeuralNetwork *nn);

// --- 17. Get Random Move ---
int get_random_move(GameState *state);

// --- 18. Play Random Game (for training) ---
char play_random_game(NeuralNetwork *nn, int *move_history, int *num_moves);

// --- 19. Train Against Random ---
void train_against_random(NeuralNetwork *nn, int num_games);

// Helper function to get valid moves.
int get_valid_moves(GameState *state, int player, int moves[BOARD_SIZE], int *num_moves);

// === Main Function ===
int main(int argc, char **argv) {
    int random_games = 50000; // Increased for better results

    if (argc > 1) random_games = atoi(argv[1]);
    srand(time(NULL));

    // Initialize neural network.
    NeuralNetwork nn;
    init_neural_network(&nn);

    // Train against random moves.
    if (random_games > 0) train_against_random(&nn, random_games);

    // Play game with human and learn more.
    while(1) {
        char play_again;
        play_game(&nn);

        printf("Play again? (y/n): ");
        scanf(" %c", &play_again);
        if (play_again != 'y' && play_again != 'Y') break;
    }
    return 0;
}

// --- 6. Neural Network Initialization ---
void init_neural_network(NeuralNetwork *nn) {
  for (int i = 0; i < NN_INPUT_SIZE * NN_HIDDEN_SIZE; i++)
    nn->weights_ih[i] = RANDOM_WEIGHT();

  for (int i = 0; i < NN_HIDDEN_SIZE * NN_OUTPUT_SIZE; i++)
    nn->weights_ho[i] = RANDOM_WEIGHT();

  for (int i = 0; i < NN_HIDDEN_SIZE; i++)
    nn->biases_h[i] = RANDOM_WEIGHT();

  for (int i = 0; i < NN_OUTPUT_SIZE; i++)
    nn->biases_o[i] = RANDOM_WEIGHT();
}

// --- 7. Softmax Function ---
void softmax(float *input, float *output, int size) {
  float max_val = input[0];
  for (int i = 1; i < size; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  if (sum > 0) {
    for (int i = 0; i < size; i++) {
      output[i] /= sum;
    }
  } else {
    for (int i = 0; i < size; i++) {
      output[i] = 1.0f / size;
    }
  }
}


// --- 8. Forward Pass ---
void forward_pass(NeuralNetwork *nn, float *inputs) {
  memcpy(nn->inputs, inputs, NN_INPUT_SIZE * sizeof(float));

  for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
    float sum = nn->biases_h[i];
    for (int j = 0; j < NN_INPUT_SIZE; j++) {
      sum += inputs[j] * nn->weights_ih[j * NN_HIDDEN_SIZE + i];
    }
    nn->hidden[i] = relu(sum);
  }

  for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
    nn->raw_logits[i] = nn->biases_o[i];
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
      nn->raw_logits[i] += nn->hidden[j] * nn->weights_ho[j * NN_OUTPUT_SIZE + i];
    }
  }

  softmax(nn->raw_logits, nn->outputs, NN_OUTPUT_SIZE);
}

// --- 9. Game Initialization ---
void init_game(GameState *state) {
    memset(state->board, 0, sizeof(state->board));
    state->tigers_on_board = 4; // Start with 4 tigers
    state->goats_on_board = 0;  // Goats enter later
    state->current_player = 0; // Goats start first
    state->placement_phase = 1; // Start in placement phase

    // Place tigers in starting positions
    state->board[0] = 2; // Top left
    state->board[4] = 2; // Top right
    state->board[20] = 2;// Bottom left
    state->board[24] = 2;// Bottom Right
}


// --- 10. Display Board (ASCII Art) ---
void display_board(GameState *state) {
    printf("Board:\n");
  for (int i = 0; i < BOARD_SIZE; i++) {
      if (state->board[i] == 0)
          printf(".  ");
      else if (state->board[i] == 1)
          printf("G  ");
      else
          printf("T  ");

      if ((i + 1) % 5 == 0)
          printf("\n");
  }
  printf("\n");

    // Print numbers for each position for the user to input.
  printf("Positions:\n");
    for (int i=0; i<BOARD_SIZE; i++){
        printf("%d  ", i);
    if ((i+1) % 5 ==0)
        printf("\n");
  }
    printf("\n");
}

// --- 11. Board to Input ---
void board_to_inputs(GameState *state, float *inputs) {
  for (int i = 0; i < BOARD_SIZE; i++) {
    if (state->board[i] == 0) {
      inputs[i * 2] = 0;
      inputs[i * 2 + 1] = 0;
    } else if (state->board[i] == 1) {
      inputs[i * 2] = 1;
      inputs[i * 2 + 1] = 0;
    } else {
      inputs[i * 2] = 0;
      inputs[i * 2 + 1] = 1;
    }
  }
}

// --- 12. Check Game Over ---
int check_game_over(GameState *state, char *winner) {
    int tiger_count = 0;
    int goat_count = 0;
    int possible_tiger_moves = 0;

    for(int i = 0; i < BOARD_SIZE; i++) {
      if(state->board[i] == 2) tiger_count++;
      if(state->board[i] == 1) goat_count++;
    }
    if (tiger_count == 0) {
        *winner = 'G';
        return 1;
    }
    if (goat_count == 0) {
        *winner = 'T';
        return 1;
    }

    // Check if tigers are blocked (goats win)
    int valid_moves[BOARD_SIZE];
    get_valid_moves(state, 1, valid_moves, &possible_tiger_moves);
    if (possible_tiger_moves == 0 && state->placement_phase == 0) {
      *winner = 'G';
      return 1;
    }
  return 0; // Game continues.
}

// --- 13. Get Computer Move ---
int get_computer_move(GameState *state, NeuralNetwork *nn, int display_probs) {
    float inputs[NN_INPUT_SIZE];
    board_to_inputs(state, inputs);
    forward_pass(nn, inputs);

    int valid_moves[BOARD_SIZE];
    int num_valid_moves = 0;
    int best_move = -1;
    float best_prob = -1.0f;

    // Get all legal moves based on current player.
    int player = state->current_player;
    get_valid_moves(state, player, valid_moves, &num_valid_moves);

    for(int i=0; i<num_valid_moves; i++){
      int move = valid_moves[i];
        if (nn->outputs[move] > best_prob) {
          best_prob = nn->outputs[move];
          best_move = move;
        }
    }

    if (display_probs) {
      printf("Neural network move probabilities:\n");
      for (int i=0; i < NN_OUTPUT_SIZE; i++){
        printf("%5.1f%% ", nn->outputs[i] * 100.0f);
        if(i%5 == 4) printf("\n");
      }
      printf("\n");
    }

  return best_move;
}

// --- 14. Backpropagation ---
void backprop(NeuralNetwork *nn, float *target_probs, float learning_rate, float reward_scaling) {
  float output_deltas[NN_OUTPUT_SIZE];
  float hidden_deltas[NN_HIDDEN_SIZE];

  for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
    output_deltas[i] =
      (nn->outputs[i] - target_probs[i]) * fabsf(reward_scaling);
  }

  for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
    float error = 0;
    for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
      error += output_deltas[j] * nn->weights_ho[i * NN_OUTPUT_SIZE + j];
    }
    hidden_deltas[i] = error * relu_derivative(nn->hidden[i]);
  }

  for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
    for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
      nn->weights_ho[i * NN_OUTPUT_SIZE + j] -=
        learning_rate * output_deltas[j] * nn->hidden[i];
    }
  }
  for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
    nn->biases_o[j] -= learning_rate * output_deltas[j];
  }

  for (int i = 0; i < NN_INPUT_SIZE; i++) {
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
      nn->weights_ih[i * NN_HIDDEN_SIZE + j] -=
        learning_rate * hidden_deltas[j] * nn->inputs[i];
    }
  }
  for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
    nn->biases_h[j] -= learning_rate * hidden_deltas[j];
  }
}

// --- 15. Learn From Game ---
void learn_from_game(NeuralNetwork *nn, int *move_history, int num_moves, int nn_moves_even, char winner) {
    float reward;
    char nn_symbol = nn_moves_even ? 'T' : 'G'; // Modified for Goat/Tiger
    if (winner == 'T') {
        reward = 1.0f;  // Reward if NN was tiger and won
    } else if( winner == 'G'){
        reward = -1.0f;  // Reward if NN was tiger and lost
    }
    else {
        reward=0.2f; // Reward if tie
    }


  GameState state;
  float target_probs[NN_OUTPUT_SIZE];

  for (int move_idx = 0; move_idx < num_moves; move_idx++) {
      if ((nn_moves_even && move_idx % 2 != 1) ||
          (!nn_moves_even && move_idx % 2 != 0))
      {
          continue;
      }

      init_game(&state);
      for (int i = 0; i < move_idx; i++) {
        int player = (i % 2 == 0) ? 0 : 1;
        int move = move_history[i];
          if(player==0){
              if(state.placement_phase==1){
                  state.board[move] = 1;
                  state.goats_on_board++;
                  if(state.goats_on_board==20) state.placement_phase = 0;

              }else{
                state.board[move]=1;
              }
          }
        else{
          state.board[move]=2;
        }
      }

      float inputs[NN_INPUT_SIZE];
      board_to_inputs(&state, inputs);
      forward_pass(nn, inputs);
      int move = move_history[move_idx];

      float move_importance = 0.5f + 0.5f * (float)move_idx/(float)num_moves;
      float scaled_reward = reward * move_importance;

    for (int i = 0; i < NN_OUTPUT_SIZE; i++)
      target_probs[i] = 0;

    if (scaled_reward >= 0) {
        target_probs[move] = 1;
    } else {
        int valid_moves[BOARD_SIZE];
        int num_valid_moves = 0;
        get_valid_moves(&state, (nn_moves_even? 1: 0),valid_moves, &num_valid_moves);
        if (num_valid_moves>0){
            float other_prob = 1.0f / num_valid_moves;
             for (int i=0; i<num_valid_moves; i++){
                int valid_move = valid_moves[i];
                 if(valid_move != move){
                   target_probs[valid_move] = other_prob;
                 }
             }
        }
    }
     backprop(nn, target_probs, LEARNING_RATE, scaled_reward);
  }
}


// --- 16. Play Game Against Human ---
void play_game(NeuralNetwork *nn) {
    GameState state;
    char winner;
    int move_history[BOARD_SIZE*BOARD_SIZE]; // maximum moves are limited here
    int num_moves = 0;
    init_game(&state);

    printf("Welcome to Baaghchaal! You are goats (G), the computer is tigers (T).\n");
    printf("Enter moves based on board positions (0-24) \n");

    while (!check_game_over(&state, &winner)) {
        display_board(&state);
        int valid_moves[BOARD_SIZE];
        int num_valid_moves = 0;

        if (state.current_player == 0) { // User as goat
            int move;
            char movec;
            while (1){
              printf("Your goat move (0-24): ");
                scanf(" %c", &movec);
              move = movec - '0';
                if (move < 0 || move >= BOARD_SIZE){
                  printf("Invalid input. Must be a number between 0-24\n");
                    continue;
                }
                get_valid_moves(&state, 0, valid_moves, &num_valid_moves);
                int valid = 0;
                for(int i =0; i<num_valid_moves; i++){
                    if(valid_moves[i]==move){
                        valid = 1;
                        break;
                    }
                }
                if(!valid){
                    printf("Invalid move, not an allowed move. Try again!\n");
                  continue;
                }
                  break;
            }
            state.board[move] = 1;
            move_history[num_moves++] = move;
            state.goats_on_board++;
            if(state.goats_on_board==20) state.placement_phase = 0;
          } else {
        // Computer (tiger) turn
        printf("Computer's move:\n");
        int move = get_computer_move(&state, nn, 1);
        state.board[move] = 2;
        move_history[num_moves++] = move;
        printf("Computer placed T at position %d\n", move);
      }
        state.current_player = !state.current_player;
    }
    display_board(&state);

    if (winner == 'G') {
      printf("You win!\n");
    }
    else if (winner == 'T'){
        printf("Computer wins!\n");
    } else {
      printf("It's a tie!\n");
    }
  learn_from_game(nn, move_history, num_moves, 1, winner);
}

// --- 17. Get Random Move ---
int get_random_move(GameState *state) {
    int valid_moves[BOARD_SIZE];
    int num_valid_moves =0;
    int player = state->current_player;

  get_valid_moves(state, player, valid_moves, &num_valid_moves);
  if(num_valid_moves>0){
      int index = rand()%num_valid_moves;
      return valid_moves[index];
  }else{
      return -1; // no move possible.
  }
}

// --- 18. Play Random Game (for training) ---
char play_random_game(NeuralNetwork *nn, int *move_history, int *num_moves) {
    GameState state;
    char winner = 0;
    *num_moves = 0;

    init_game(&state);

    while (!check_game_over(&state, &winner)) {
      int move;
      if (state.current_player == 0) {
        move = get_random_move(&state);
        if(move==-1) break; // if there is no valid move stop the game.
      } else {
        move = get_computer_move(&state, nn, 0);
      }

      char symbol = (state.current_player == 0) ? 1 : 2;
      state.board[move]=symbol;
      move_history[(*num_moves)++] = move;

      state.current_player = !state.current_player;
    }

  learn_from_game(nn,move_history, *num_moves, 1, winner);
    return winner;
}

// --- 19. Train Against Random ---
void train_against_random(NeuralNetwork *nn, int num_games) {
    int move_history[BOARD_SIZE*BOARD_SIZE]; // max moves
    int num_moves;
    int wins = 0, losses = 0, ties = 0;

    printf("Training neural network against %d random games...\n", num_games);
    for (int i = 0; i < num_games; i++) {
      char winner = play_random_game(nn, move_history, &num_moves);
    if(winner == 'T'){
      wins++;
    }
    else if (winner == 'G')
    {
        losses++;
      }
      else{
          ties++;
      }

        if ((i + 1) % 1000 == 0) {
          printf("Games: %d, Wins: %d (%.1f%%), "
                 "Losses: %d (%.1f%%), Ties: %d (%.1f%%)\n",
                 i + 1, wins, (float)wins * 100 / (i + 1),
                 losses, (float)losses * 100 / (i + 1),
                 ties, (float)ties * 100 / (i + 1));
        }
    }
    printf("\nTraining complete!\n");
}

// Helper function to get valid moves.
int get_valid_moves(GameState *state, int player, int moves[BOARD_SIZE], int *num_moves) {
    *num_moves = 0;
    int i;
  if(state->placement_phase==1 && player == 0){
      for(i=0; i<BOARD_SIZE; i++){
        if(state->board[i]==0){
              moves[*num_moves] = i;
                (*num_moves)++;
        }
      }
    return 1;
  }
  if (player == 0) { // Goats
    for(i = 0; i < BOARD_SIZE; i++) {
      if(state->board[i] == 1) continue; // Cannot move from another goat position.
      if (state->board[i]==0) { // Only move to empty positions

          int row = i / 5;
          int col = i % 5;

          // Check adjacent positions
          int positions[4][2] = {
              {row-1, col}, {row+1, col}, {row, col-1}, {row, col+1}
          };
          for(int j=0; j < 4; j++){
              int new_row = positions[j][0];
              int new_col = positions[j][1];
                if(new_row>=0 && new_row<5 && new_col>=0 && new_col<5){
                    int position = new_row*5 + new_col;
                   if(state->board[position]==0){
                        moves[*num_moves] = position;
                        (*num_moves)++;
                   }
                }
          }

      }
    }
    return 1;
    } else {  // Tigers
    for (i = 0; i < BOARD_SIZE; i++) {
        if (state->board[i] == 2) {
        int row = i/5;
        int col = i%5;
        int positions[8][2] = {
            {row-1,col}, {row+1, col}, {row, col-1}, {row, col+1},
            {row-2,col}, {row+2, col}, {row, col-2}, {row, col+2}
        };
          for (int j=0; j < 8; j++) {
              int new_row = positions[j][0];
              int new_col = positions[j][1];
                if(new_row>=0 && new_row<5 && new_col>=0 && new_col<5){
                   int pos = new_row*5 + new_col;
                   if(j<4){ // Normal step move
                        if(state->board[pos]==0){
                            moves[*num_moves]=pos;
                            (*num_moves)++;
                        }
                    }
                   else{ // Jump move.
                       int jump_row = positions[j-4][0];
                       int jump_col = positions[j-4][1];
                        int jump_pos = jump_row*5 + jump_col;

                       if (state->board[jump_pos]==1 && state->board[pos]==0) {
                           moves[*num_moves] = pos;
                            (*num_moves)++;
                        }
                   }
                }
            }
        }
    }
    return 1;
    }
}