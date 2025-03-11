#ifndef ALPHABAGHCHAL_MCTS_H
#define ALPHABAGHCHAL_MCTS_H

#include "alphabaghchal.h"

// MCTS Node Structure
typedef struct MCTSNode {
  GameState state;
  int move;
  float visit_count;
  float value_sum;
  struct MCTSNode *children[BOARD_SIZE];
  int num_children;
  struct MCTSNode *parent;
} MCTSNode;


MCTSNode *create_node(GameState state, int move, MCTSNode* parent);
void free_tree(MCTSNode *node);
int select_move(MCTSNode *node);
void expand_node(MCTSNode *node, NeuralNetwork *nn);
float simulate(MCTSNode *node, NeuralNetwork *nn);
void backpropagate(MCTSNode *node, float value);
int mcts_search(GameState state, NeuralNetwork *nn);

#endif