#include "neural_net.h"
#include "math.h"

NeuralNetwork* create_neural_network(size_t num_layers, size_t *layer_dims, activation_t *activation_funcs) {

    if (num_layers <= 1 || num_layers > 1000 || layer_dims == NULL || activation_funcs == NULL) return NULL;

    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    if (nn == NULL) return NULL;

    nn->num_layers = num_layers - 1;
    nn->layers = (Layer *)malloc(sizeof(Layer) * nn->num_layers);
    if (nn->layers == NULL) {
        free(nn);
        return NULL;
    }

    for (size_t i = 0; i < nn->num_layers; i++) {
        nn->layers[i] = *create_layer(layer_dims[i], layer_dims[i + 1], activation_funcs[i]);
        if (nn->layers[i].weights == NULL || nn->layers[i].biases == NULL) {
            for (size_t j = 0; j < i; j++) {
                free_layer(&(nn->layers[j]));
            }
            free(nn->layers);
            free(nn);
            return NULL;
        }
    }

    return nn;
}

void free_neural_network(NeuralNetwork *nn) {

    if (nn) 
    {
        for (size_t i = 0; i < nn->num_layers; i++) 
        {
            free_layer(&(nn->layers[i]));
        }

        free(nn->layers);
        free(nn);
    }
}

Layer* create_layer(size_t input_dim, size_t output_dim, activation_t activation_func)
{
    Layer *layer = (Layer *)malloc(sizeof(Layer));
    if (layer == NULL)  {
        return NULL;
    }

    layer->weights = initialise_matrix(output_dim, input_dim);
    if(layer->weights == NULL)  {
        return NULL;
    }
    layer->biases = initialise_matrix(output_dim, 1);
    if(layer->biases == NULL)   {
        return NULL;
    }

    initialise_weights(layer->weights);

    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    layer->activation_func = activation_func;

    return layer;
}

void free_layer(Layer *layer)
{
    if(layer == NULL) return;

    free_matrix(layer->weights);
    free_matrix(layer->biases);
    free(layer);
}

int layer_forward(Layer *layer, Matrix *input, Matrix *output)
{
    if(layer == NULL || input == NULL || output == NULL) return 1;
    if (input->cols != layer->weights->rows) return 2;

    Matrix *linear_output = initialise_matrix(input->rows, layer->output_dim);
    if(linear_output == NULL) return 4;

    int ret = matrix_multiply_opt(input, layer->weights, linear_output);
    if(ret) {
        free_matrix(linear_output);
        return ret;
    }

    Matrix *bias_bd =  initialise_matrix(linear_output->rows, linear_output->cols);
    ret = broadcast_matrix(layer->biases, bias_bd);
    if(ret) {
        free_matrix(linear_output);
        return ret;
    }

    ret = matrix_add(linear_output, bias_bd, linear_output);
    if(ret) {
        free_matrix(linear_output);
        return ret;
    }
    
    ret = activation(linear_output, output, layer->activation_func);
    free_matrix(linear_output);
    return ret;
}

int nn_forward(NeuralNetwork *nn, Matrix *input, Matrix *output)
{
    if(nn == NULL || input == NULL || output == NULL) return 1;

    size_t num_layers = nn->num_layers;

    int ret;
    for(size_t i = 0; i < num_layers; i++)
    {
        ret = layer_forward(&nn->layers[i], input, output);
        if(ret) return ret;

        memcpy(input, output, sizeof(Matrix));
        memcpy(input->data, output->data, input->rows * input->cols * sizeof(double));
    }

    return 0;
}

double __attribute__((noinline)) sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double __attribute__((noinline)) relu(double x) {
    return (x > 0.0) ? x : 0.0;
}

double __attribute__((noinline)) tanh(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

static double dsigmoid(double x) {
    double sig = sigmoid(x);
    return sig*(1-sig);
}

static double drelu(double x)   {
    return (x > 0) ? 1.0 : 0.0;
}

static double dtanh(double x)   {
    double t = tanh(x);
    return 1 - t*t;
}

int initialise_weights(Matrix *matrix) {

    if(matrix == NULL) return 1;

    double stddev = 1.0 / sqrt(matrix->rows);
    for (size_t i = 0; i < matrix->rows * matrix->cols; i++) {
        matrix->data[i] = stddev * (2.0 * (rand() / (double)RAND_MAX) - 1.0); // Xavier method
    }

    return 0;
}

int activation(Matrix *input, Matrix *output, activation_t activation_func)
{

    if(input == NULL || output == NULL) return 1;

    if(activation_func == sigmoid) {
        for(int i = 0; i < input->rows*input->cols; i++) output->data[i] = sigmoid(input->data[i]);
    } else if(activation_func == relu)  {
        for(int i = 0; i < input->rows*input->cols; i++) output->data[i] = relu(input->data[i]);
    } else if(activation_func == tanh)  {
        for(int i = 0; i < input->rows*input->cols; i++) output->data[i] = tanh(input->data[i]);
    } else  {
        return 5;
    }

    return 0;
}

int dactivation(Matrix *input, Matrix *output, activation_t activation_func)
{
    if(input == NULL || output == NULL) return 1;

    if(activation_func == sigmoid) {
        for(int i = 0; i < input->rows * input->cols; i++) output->data[i] = dsigmoid(input->data[i]);
    } else if(activation_func == relu) {
        for(int i = 0; i < input->rows * input->cols; i++) output->data[i] = drelu(input->data[i]);
    } else if(activation_func == tanh) {
        for(int i = 0; i < input->rows * input->cols; i++) output->data[i] = dtanh(input->data[i]);
    } else {
        return 5;
    }

    return 0;
}
