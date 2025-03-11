# AlphaBaghChal: Baagh Chaal with Monte Carlo Tree Search and Neural Network AI (CUDA Accelerated)

This project implements the traditional Baagh Chaal (Tiger and Goat) board game, enhanced with a sophisticated AI using Monte Carlo Tree Search (MCTS) guided by a neural network for both policy and value estimation. The game logic and MCTS are written in C, and the neural network calculations are accelerated using NVIDIA's CUDA for GPU computing.

## Table of Contents

- [AlphaBaghChal: Baagh Chaal with Monte Carlo Tree Search and Neural Network AI (CUDA Accelerated)](#alphabaghchal-baagh-chaal-with-monte-carlo-tree-search-and-neural-network-ai-cuda-accelerated)
  - [Table of Contents](#table-of-contents)
  - [1. Game Overview](#1-game-overview)
  - [2. Code Structure](#2-code-structure)
  - [3. How to Play](#3-how-to-play)
  - [4. How to Compile and Run](#4-how-to-compile-and-run)
  - [5. Code Explanation](#5-code-explanation)
    - [alphabaghchal.c Explanation](#alphabaghchalc-explanation)
    - [alphabaghchal\_cuda.cu Explanation](#alphabaghchal_cudacu-explanation)
    - [alphabaghchal\_mcts.c Explanation](#alphabaghchal_mctsc-explanation)
  - [6. Neural Network Details](#6-neural-network-details)
  - [7. Training](#7-training)
  - [8. Limitations](#8-limitations)

## 1. Game Overview

Baagh Chaal is a two-player abstract strategy game played on a 5x5 grid.

- **Players:** One player controls four tigers, and the other controls twenty goats.
- **Objective:**
  - **Tigers:** Capture all the goats, or block them so they cannot move.
  - **Goats:** Block all the tigers from moving.
- **Game Phases:**
    1. **Placement Phase:** Goats are placed one at a time on the board until all 20 are placed.
    2. **Movement Phase:** Players take turns moving their pieces.
- **Movement:**
  - **Goats:** Move one step orthogonally (up, down, left, right) to an empty space.
  - **Tigers:**
    - Move one step orthogonally to an empty space.
    - Capture a goat by jumping over it orthogonally to an empty space.
- **Initial Setup:** The game starts with 4 tigers placed on the corners of the board.

## 2. Code Structure

This project consists of the following source files:

- `alphabaghchal.h`: Header file containing definitions and function prototypes.
- `alphabaghchal.c`: Contains the main game logic, user interface, training, and integration with MCTS.
- `alphabaghchal_cuda.cu`: Contains the CUDA kernels for the neural network computations on the GPU.
- `alphabaghchal_mcts.h`: Header file for the MCTS implementation.
- `alphabaghchal_mcts.c`: Implements the Monte Carlo Tree Search algorithm.

## 3. How to Play

1. **Goat Player (You):**
    - When it's your turn, the program will display the board with numbers for positions (0-24).
    - Enter your move by typing the number of a board position and pressing enter.
    - During the placement phase, place the goats on empty positions.
    - During the movement phase, move goats one step to a valid position.
2. **Tiger Player (Computer AI):**
    - The computer AI, using MCTS guided by the trained neural network, will make the tiger's moves.
    - The computer's move and the network's evaluation of the move will be displayed on the console.
3. **Game End:** The game ends when either:
    - The tigers capture all the goats.
    - The goats block all the tigers from moving.
    - A tie happens if a tiger does not have a valid move but cannot be blocked.

## 4. How to Compile and Run

You need the following to compile and run the game:

- An NVIDIA GPU with CUDA support and a correctly installed CUDA toolkit.
- A C compiler (such as GCC).
- The files `alphabaghchal.h`, `alphabaghchal.c`, `alphabaghchal_cuda.cu`, `alphabaghchal_mcts.h`, and `alphabaghchal_mcts.c` must be in the same directory.

**Compilation Steps:**

1. Compile `alphabaghchal_cuda.cu` into an object file:

    ```bash
    nvcc -c alphabaghchal_cuda.cu -o alphabaghchal_cuda.o
    ```

2. Compile `alphabaghchal.c` and `alphabaghchal_mcts.c` into object files:

    ```bash
    gcc -c alphabaghchal.c -o alphabaghchal.o
    gcc -c alphabaghchal_mcts.c -o alphabaghchal_mcts.o
    ```

3. Link the object files to create the executable:

    ```bash
    gcc alphabaghchal.o alphabaghchal_mcts.o alphabaghchal_cuda.o -o alphabaghchal -lm -lcuda
    ```

**Running the Game:**

1. Execute the compiled program:

    ```bash
    ./alphabaghchal
    ```

2. The game will first train the neural network, which can take some time depending on the training games count. You can specify the number of training games as a command line argument.  For example: `./alphabaghchal 10000` will train for 10000 games. The default is 5000 if no arguments are provided.
3. Once training is complete, you will play against the computer AI, which uses MCTS guided by the trained neural network.
4. Follow the prompts in the console to play.

## 5. Code Explanation

### alphabaghchal.c Explanation

This file handles the game logic, user interaction, and training for the AlphaBaghChal project:

- **Data Structures:**
  - `GameState`: Stores the state of the game, including the board, players' turns, placement phase status, and the number of tigers and goats on the board.
  - `NeuralNetwork`: Defines the structure of the neural network, including weights, biases, inputs, hidden layer, outputs (policy probabilities), and value estimation.
- **Game Logic:**
  - `init_game()`: Initializes the game board with starting positions.
  - `display_board()`: Prints the game board to the console.
  - `board_to_inputs()`: Converts the board state into neural network input.
  - `check_game_over()`: Determines if the game has ended and who the winner is.
  - `get_valid_moves()`: Returns all the valid moves for each player depending on the game state.
  - `get_computer_move()`: Calls the MCTS to get the best move for the tiger.
  - `get_random_move()`: Used in training mode, gives a random move depending on the valid moves.
  - `play_game()`: Manages the game loop and interacts with the user and AI.
  - `play_random_game()`: Plays random games for training.
  - `train_against_random()`: Trains the neural network against random moves.
- **Neural Network Integration:**
  - `init_neural_network()`: Allocates memory for the network on the GPU and initializes the weights and biases with random values.
  - `forward_pass()`: Performs the forward pass of the neural network on the GPU.
  - `backprop()`: Performs backpropagation on the GPU to update network weights and biases.
  - `learn_from_game()`: Updates the network weights based on the outcomes of a game.
- **Main Function:**
  - Parses command-line arguments for the number of training games.
  - Initializes the neural network.
  - Trains the network against random moves.
  - Enters the interactive game loop.
  - Frees GPU memory allocated for the neural network.

### alphabaghchal\_cuda.cu Explanation

This file contains the CUDA kernel functions for GPU acceleration of the neural network:

- **CUDA Kernels:**
  - `cuda_forward_kernel()`: Performs the feedforward calculation of the neural network on the GPU using CUDA. It includes the ReLU activation, the hidden and the output layer, and outputs both the policy and the value of the given board state.
  - `cuda_backprop_kernel()`: Performs the backpropagation calculation of the neural network on the GPU, updating the weights and biases based on the policy and value errors.  It uses shared memory to communicate between threads.
- **Helper Functions:**
  - `cuda_forward_kernel_wrapper()`: A wrapper function to call the `cuda_forward_kernel`.
  - `cuda_backprop_kernel_wrapper()`: A wrapper function to call the `cuda_backprop_kernel`.
- **Extern C Wrapper**: The `extern C` block exports the functions with C calling conventions for use by `alphabaghchal.c`
- **Device Functions:**
  - `relu()`: Computes the ReLU activation on the GPU.
  - `relu_derivative()`: Computes the derivative of ReLU for backpropagation on the GPU.

### alphabaghchal\_mcts.c Explanation

This file implements the Monte Carlo Tree Search (MCTS) algorithm:

- **Data Structures:**
    -`MCTSNode`: Represents a node in the MCTS tree, storing the game state, the move that led to that state, visit counts, accumulated values, child nodes, and the parent node.
- **MCTS Functions:**
  - `create_node()`: Creates a new node for the MCTS tree.
  - `free_tree()`: Frees the memory allocated for the MCTS tree.
  - `select_move()`: Traverses the MCTS tree to find the most promising node to explore using the Upper Confidence Bound (UCB) formula.
  - `expand_node()`: Expands the current node by creating child nodes for all the possible valid moves.
  - `simulate()`: Performs a rollout simulation from the current node to estimate a value. It currently simulates random moves until the end of the game, but ideally the neural network should be used for rollouts too.
  - `backpropagate()`: Updates the visit counts and accumulated values of all nodes in the path from the simulated node to the root.
  - `mcts_search()`: Orchestrates the MCTS algorithm, performing multiple simulations, and returns the best move based on the accumulated statistics.

## 6. Neural Network Details

- **Architecture:** The neural network has one hidden layer, outputting both a policy distribution and a value.
  - **Input Layer:** `NN_INPUT_SIZE` (50 nodes). Each cell on the 5x5 board is represented by two nodes, indicating its state (empty, goat, or tiger).
  - **Hidden Layer:** `NN_HIDDEN_SIZE` (128 nodes).
  - **Output Layer:** `NN_OUTPUT_SIZE` (25 nodes). Each node represents a board position for the policy output. It also has a single output for the value of the position.
- **Softmax**: The softmax function is used to convert the policy output to probabilities for each cell.
- **Activation:** ReLU is used as the activation function for the hidden layer, and tanh for the value output.
- **Training:**  The network is trained through self-play against a random player, with an optimized backpropagation based on the reward of the current game, incorporating the value prediction error.

## 7. Training

- The neural network is trained through self-play against a random player, with backpropagation on both policy and value estimations. This process starts when you execute the program.
- You can control the number of training games by providing a command-line argument when running the program. For example, `./alphabaghchal 10000` will train for 10000 games. The default is 5000.
- The program reports training progress and win/loss statistics in the console.
- Training with more games can significantly enhance the AI's strength.

## 8. Limitations

- The neural network's AI strength depends heavily on the amount of training.
- The MCTS parameters (`MCTS_SIMULATIONS`, `CPUCT`) are hardcoded and could be optimized.
- The use of random playouts in the MCTS simulation could be enhanced by using the neural network to guide the simulations, and by using more sophisticated exploration methods.
- The program currently has some basic error checking, it is recommended to add more for a better experience.
- The current neural network architecture may be simplistic and could be further refined.
