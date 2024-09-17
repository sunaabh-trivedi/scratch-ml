#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "matrix.h"

typedef enum    {
    sigmoid, 
    relu,
    tanh
} activation_t;

typedef struct  {
    size_t input_dim;
    size_t output_dim;
    Matrix *weights;
    Matrix *biases;
    activation_t activation_func;
} Layer;

typedef struct  {
    Layer *layers;
    size_t num_layers;
} NeuralNetwork;

/* Activation Functions */
double sigmoid(double x);
double relu(double x);
double tanh(double x);

int activation(Matrix *input, Matrix *output, activation_t activation_func);
int dactivation(Matrix *input, Matrix *output, activation_t activation_func);

/* Layer Operations */
Layer* create_layer(size_t input_dim, size_t output_dim, activation_t activation_func);
void free_layer(Layer *layer);
int layer_forward(Layer *layer, Matrix *input, Matrix *output);

/* Neural Network Operations */
NeuralNetwork* create_neural_network(size_t num_layers, size_t *layer_dims, activation_t *activation_funcs);
void free_neural_network(NeuralNetwork *nn);
int nn_forward(NeuralNetwork *nn, Matrix *input, Matrix *output);

/* Loss Functions */
double mse_loss(Matrix *predicted, Matrix *target);
double kl_divergence(Matrix *predicted, Matrix *target);
double vae_loss(Matrix *predicted, Matrix *target);

/* Training Functions */
void update_weights(NeuralNetwork *nn, Matrix *gradient, double learning_rate);

/* Utility Functions */
int initialise_weights(Matrix *matrix);

#endif
