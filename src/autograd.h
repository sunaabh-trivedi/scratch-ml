#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "matrix.h"

// Forward declaration
typedef struct gNode gNode_t;

typedef enum {
    ADD,
    SUB,
    MUL,
    SIGMOID,
    RELU,
    TANH
} op_t;

struct gNode {
    Matrix *val;
    Matrix *grad;
    gNode_t *parents[2];
    op_t op;
    void (*backward)(gNode_t *gNode);
    int visited;
};

typedef struct {
    gNode_t *root;
    size_t num_nodes;
} gDAG_t;

gNode_t* create_node(Matrix *val, op_t op, gNode_t *parent1, gNode_t *parent2);
void free_gNode(gNode_t *gNode);
void free_gDAG(gDAG_t *gDAG);

void backward_add(gNode_t *gNode);
void backward_sub(gNode_t *gNode);
void backward_mul(gNode_t *gNode);
void backward_sigmoid(gNode_t *gNode);
void backward_relu(gNode_t *gNode);
void backward_tanh(gNode_t *gNode);
void free_gNode(gNode_t *gNode);

#endif // AUTOGRAD_H