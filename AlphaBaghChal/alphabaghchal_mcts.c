#include "alphabaghchal_mcts.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

MCTSNode *create_node(GameState state, int move, MCTSNode* parent) {
    MCTSNode *node = (MCTSNode *)malloc(sizeof(MCTSNode));
    if (!node) {
        perror("Failed to allocate memory for MCTS node");
        exit(EXIT_FAILURE);
    }
    node->state = state;
    node->move = move;
    node->visit_count = 0;
    node->value_sum = 0;
    node->num_children = 0;
    node->parent = parent;
    for(int i=0; i<BOARD_SIZE; i++){
        node->children[i] = NULL;
    }
    return node;
}

void free_tree(MCTSNode *node) {
    if (node == NULL) return;
    for (int i = 0; i < node->num_children; i++) {
       if(node->children[i])
          free_tree(node->children[i]);
    }
    free(node);
}

float calculate_ucb(MCTSNode *node, float total_visits) {
    if(node->visit_count == 0){
      return FLT_MAX;
    }

    float exploitation = node->value_sum / node->visit_count;
    float exploration = CPUCT * sqrtf(logf(total_visits) / node->visit_count);
    return exploitation + exploration;
}

int select_move(MCTSNode *node) {
    float best_ucb = -FLT_MAX;
    int best_child = -1;
  
    float total_visits = node->visit_count;
    for (int i = 0; i < node->num_children; i++) {
      MCTSNode* child = node->children[i];
        if(child){
        float ucb = calculate_ucb(child, total_visits);
           if (ucb > best_ucb) {
              best_ucb = ucb;
                best_child = i;
            }
        }
    }
    return best_child;
}

void expand_node(MCTSNode *node, NeuralNetwork *nn) {
  if(node->num_children > 0) return; // already expanded.
  
    int valid_moves[BOARD_SIZE];
    int num_valid_moves;
    get_valid_moves(&node->state, node->state.current_player, valid_moves, &num_valid_moves);

  for (int i = 0; i < num_valid_moves; i++) {
      int move = valid_moves[i];
        GameState next_state = node->state;
      char symbol = (next_state.current_player == 0) ? 1 : 2;
      next_state.board[move]=symbol;
        if(next_state.current_player==0) {
          next_state.goats_on_board++;
          if(next_state.goats_on_board==20) next_state.placement_phase=0;
        }
      next_state.current_player = !next_state.current_player;
      node->children[i] = create_node(next_state, move, node);
      node->num_children++;
  }
}

float simulate(MCTSNode *node, NeuralNetwork *nn) {
    GameState state = node->state;
    char winner;
  while (!check_game_over(&state, &winner)) {
    int valid_moves[BOARD_SIZE];
    int num_valid_moves = 0;
    get_valid_moves(&state, state.current_player, valid_moves, &num_valid_moves);
        if(num_valid_moves==0){
            break;
        }
        int move_index = rand()%num_valid_moves;
    int move = valid_moves[move_index];
    char symbol = (state.current_player == 0) ? 1 : 2;
      state.board[move]=symbol;
    if(state.current_player==0) {
        state.goats_on_board++;
        if(state.goats_on_board==20) state.placement_phase=0;
    }
      state.current_player = !state.current_player;
    }

    if (winner == 'T'){
      return 1;
    } else if (winner == 'G') {
        return -1;
    } else {
        return 0;
    }
}

void backpropagate(MCTSNode *node, float value) {
    while (node != NULL) {
        node->visit_count += 1;
        node->value_sum += value;
        node = node->parent;
        value = -value; // Alternate perspective
    }
}

int mcts_search(GameState state, NeuralNetwork *nn) {
    MCTSNode *root = create_node(state, -1, NULL);

    for (int i = 0; i < MCTS_SIMULATIONS; i++) {
      MCTSNode *current = root;

        // Selection: Find the best child
        while (current->num_children > 0) {
          int best_child = select_move(current);
          if(best_child == -1) break;
            current = current->children[best_child];
        }

        // Expansion
        expand_node(current, nn);

        // Simulation or rollout
      float value;
        if(current->num_children>0){
            int child_idx = rand()%current->num_children;
             value = simulate(current->children[child_idx], nn);
        }
        else{
           value = simulate(current, nn);
        }

        // Backpropagation
        backpropagate(current, value);
    }

    // Choose the best action after simulation is complete
    int best_action = -1;
    float best_visit = -1;
  for(int i =0; i< root->num_children; i++){
    if(root->children[i] && root->children[i]->visit_count > best_visit){
      best_visit = root->children[i]->visit_count;
      best_action = root->children[i]->move;
    }
  }

  free_tree(root);
  return best_action;
}