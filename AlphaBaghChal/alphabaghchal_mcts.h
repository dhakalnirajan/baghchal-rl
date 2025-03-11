/*****************************************************************************
 *  File: alphabaghchal_mcts.h
 *  Author: Nirajan Dhakal
 *  Date: March 12, 2025
 *  License: MIT License
 *
 *  Description:
 *  This header file defines the data structures and function prototypes for the
 *  Monte Carlo Tree Search (MCTS) implementation used in the Baghchal game. It
 *  includes the definition of the MCTSNode structure, which represents a node in
 *  the MCTS tree, along with the function declarations for operations such as node
 *  creation, tree traversal, move selection, node expansion, simulation,
 *  backpropagation, and the main MCTS search function. This file serves as the
 *  interface for the MCTS logic, providing the necessary declarations for use
 *  in other parts of the program.
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