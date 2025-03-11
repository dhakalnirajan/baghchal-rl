#include "alphabaghchal.h"
#include "alphabaghchal_cuda.h"
#include "alphabaghchal_mcts.h"

// === Main Function ===
int main(int argc, char **argv) {
    int random_games = 5000;

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

    // Free GPU memory allocated for the neural network.
    cudaFree(nn.weights_ih);
    cudaFree(nn.weights_ho);
    cudaFree(nn.biases_h);
    cudaFree(nn.biases_o);
    cudaFree(nn.inputs);
    cudaFree(nn.hidden);
    cudaFree(nn.raw_logits);
    cudaFree(nn.policy_probs);
    cudaFree(nn.value);

    return 0;
}

void init_neural_network(NeuralNetwork *nn) {

    // Allocate memory on GPU
    cudaMalloc((void**)&nn->weights_ih, NN_INPUT_SIZE * NN_HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&nn->weights_ho, NN_HIDDEN_SIZE * NN_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&nn->biases_h, NN_HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&nn->biases_o, NN_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&nn->inputs, NN_INPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&nn->hidden, NN_HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&nn->raw_logits, NN_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&nn->policy_probs, NN_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&nn->value, sizeof(float));

    float *h_weights_ih = (float *)malloc(NN_INPUT_SIZE * NN_HIDDEN_SIZE * sizeof(float));
    float *h_weights_ho = (float *)malloc(NN_HIDDEN_SIZE * NN_OUTPUT_SIZE * sizeof(float));
    float *h_biases_h = (float *)malloc(NN_HIDDEN_SIZE * sizeof(float));
    float *h_biases_o = (float *)malloc(NN_OUTPUT_SIZE * sizeof(float));


    for (int i = 0; i < NN_INPUT_SIZE * NN_HIDDEN_SIZE; i++) h_weights_ih[i] = (((float)rand() / RAND_MAX) - 0.5f) * 0.1f;
    for (int i = 0; i < NN_HIDDEN_SIZE * NN_OUTPUT_SIZE; i++) h_weights_ho[i] = (((float)rand() / RAND_MAX) - 0.5f) * 0.1f;
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) h_biases_h[i] = (((float)rand() / RAND_MAX) - 0.5f) * 0.1f;
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) h_biases_o[i] = (((float)rand() / RAND_MAX) - 0.5f) * 0.1f;

    // Copy from host to device
    cudaMemcpy(nn->weights_ih, h_weights_ih, NN_INPUT_SIZE * NN_HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nn->weights_ho, h_weights_ho, NN_HIDDEN_SIZE * NN_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nn->biases_h, h_biases_h, NN_HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nn->biases_o, h_biases_o, NN_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    free(h_weights_ih);
    free(h_weights_ho);
    free(h_biases_h);
    free(h_biases_o);
}

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

void forward_pass(NeuralNetwork *nn, float *inputs) {
    // Copy inputs from host to device
    cudaMemcpy(nn->inputs, inputs, NN_INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cuda_forward_kernel(nn, NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);

    // Copy outputs back from device to host
    cudaMemcpy(inputs, nn->policy_probs, NN_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
}


void init_game(GameState *state) {
    memset(state->board, 0, sizeof(state->board));
    state->tigers_on_board = 4;
    state->goats_on_board = 0;
    state->current_player = 0;
    state->placement_phase = 1;

    state->board[0] = 2;
    state->board[4] = 2;
    state->board[20] = 2;
    state->board[24] = 2;
}


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
  return 0;
}

int get_computer_move(GameState *state, NeuralNetwork *nn, int display_probs) {
    int move = mcts_search(state, nn);
    if(display_probs){
        float inputs[NN_INPUT_SIZE];
        board_to_inputs(state, inputs);
        forward_pass(nn, inputs);
        printf("Neural network move probabilities:\n");
            for (int i=0; i < NN_OUTPUT_SIZE; i++){
            printf("%5.1f%% ", inputs[i] * 100.0f);
            if(i%5 == 4) printf("\n");
            }
        printf("\n");
    }
  return move;
}


void backprop(NeuralNetwork *nn, float *target_probs, float *target_value, float learning_rate) {
  cuda_backprop_kernel(nn, target_probs, target_value, learning_rate, NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
}

void learn_from_game(NeuralNetwork *nn, int *move_history, int num_moves, int nn_moves_even, char winner, float *values_history) {
    float reward;
    char nn_symbol = nn_moves_even ? 'T' : 'G';
    if (winner == 'T') {
        reward = 1.0f;
    } else if( winner == 'G'){
        reward = -1.0f;
    }
    else {
        reward=0.2f;
    }

  GameState state;
  float target_probs[NN_OUTPUT_SIZE];
    float *h_target_probs = (float*)malloc(NN_OUTPUT_SIZE * sizeof(float));
  float target_value;
  float *h_target_value = &target_value;


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
      float value = values_history[move_idx];
        target_value = reward * value;

      float move_importance = 0.5f + 0.5f * (float)move_idx/(float)num_moves;
      float scaled_reward = reward * move_importance;


        for (int i = 0; i < NN_OUTPUT_SIZE; i++)
            h_target_probs[i] = 0;

        if (scaled_reward >= 0) {
            h_target_probs[move] = 1.0f;
        } else {
            int valid_moves[BOARD_SIZE];
            int num_valid_moves = 0;
            get_valid_moves(&state, (nn_moves_even? 1: 0),valid_moves, &num_valid_moves);
            if (num_valid_moves>0){
                float other_prob = 1.0f / num_valid_moves;
                for (int i=0; i<num_valid_moves; i++){
                    int valid_move = valid_moves[i];
                    if(valid_move != move){
                        h_target_probs[valid_move] = other_prob;
                    }
                 }
            }
        }

      cudaMemcpy(target_probs, h_target_probs, NN_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(nn->value, h_target_value, sizeof(float), cudaMemcpyHostToDevice);

     backprop(nn, target_probs, nn->value, LEARNING_RATE);
    }
  free(h_target_probs);
}

void play_game(NeuralNetwork *nn) {
    GameState state;
    char winner;
    int move_history[BOARD_SIZE*BOARD_SIZE];
    float values_history[BOARD_SIZE*BOARD_SIZE];
    int num_moves = 0;
    init_game(&state);

    printf("Welcome to Baaghchaal! You are goats (G), the computer is tigers (T).\n");
    printf("Enter moves based on board positions (0-24) \n");

    while (!check_game_over(&state, &winner)) {
        display_board(&state);
        int valid_moves[BOARD_SIZE];
        int num_valid_moves = 0;

        if (state.current_player == 0) {
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
            move_history[num_moves] = move;
            values_history[num_moves] = 0;
            num_moves++;
            state.goats_on_board++;
            if(state.goats_on_board==20) state.placement_phase = 0;
          } else {
        printf("Computer's move:\n");
        int move = get_computer_move(&state, nn, 1);
        state.board[move] = 2;
        move_history[num_moves] = move;
          float inputs[NN_INPUT_SIZE];
          board_to_inputs(&state, inputs);
          forward_pass(nn, inputs);
          float value;
          cudaMemcpy(&value, nn->value, sizeof(float), cudaMemcpyDeviceToHost);
        values_history[num_moves] = value;
        num_moves++;
        printf("Computer placed T at position %d, with a value of %.2f\n", move, value);
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
  learn_from_game(nn, move_history, num_moves, 1, winner, values_history);
}

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

char play_random_game(NeuralNetwork *nn, int *move_history, int *num_moves, float *values_history) {
    GameState state;
    char winner = 0;
    *num_moves = 0;

    init_game(&state);

    while (!check_game_over(&state, &winner)) {
      int move;
      if (state.current_player == 0) {
        move = get_random_move(&state);
        if(move==-1) break;
      } else {
        move = get_computer_move(&state, nn, 0);
          float inputs[NN_INPUT_SIZE];
          board_to_inputs(&state, inputs);
          forward_pass(nn, inputs);
          float value;
          cudaMemcpy(&value, nn->value, sizeof(float), cudaMemcpyDeviceToHost);
          values_history[(*num_moves)] = value;

      }

      char symbol = (state.current_player == 0) ? 1 : 2;
      state.board[move]=symbol;
      move_history[(*num_moves)++] = move;

      state.current_player = !state.current_player;
    }
  learn_from_game(nn,move_history, *num_moves, 1, winner, values_history);
    return winner;
}

void train_against_random(NeuralNetwork *nn, int num_games) {
    int move_history[BOARD_SIZE*BOARD_SIZE];
    float values_history[BOARD_SIZE*BOARD_SIZE];
    int num_moves;
    int wins = 0, losses = 0, ties = 0;

    printf("Training neural network against %d random games...\n", num_games);
    for (int i = 0; i < num_games; i++) {
      char winner = play_random_game(nn, move_history, &num_moves, values_history);
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
  if (player == 0) {
    for(i = 0; i < BOARD_SIZE; i++) {
      if(state->board[i] == 1) continue;
      if (state->board[i]==0) {

          int row = i / 5;
          int col = i % 5;

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
    } else {
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
                   if(j<4){
                        if(state->board[pos]==0){
                            moves[*num_moves]=pos;
                            (*num_moves)++;
                        }
                    }
                   else{
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