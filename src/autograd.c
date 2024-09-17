#include "autograd.h"
#include "queue.h"
#include "vector.h"
#include <stdlib.h>
#include <math.h>

gNode_t* create_node(Matrix *val, op_t op, gNode_t *parent1, gNode_t *parent2)
{
    gNode_t *gNode = (gNode_t*)malloc(sizeof(gNode_t));
    if(gNode == NULL) return NULL;

    gNode->val = val;
    gNode->op = op;
    gNode->parents[0] = parent1;
    gNode->parents[1] = parent2;
    gNode->grad = 0;
    gNode->visited = 0;


    switch(op)
    {
        case ADD:
            gNode->backward = backward_add;
            break;
        case MUL:
            gNode->backward = backward_mul;
            break;
        case SUB:
            gNode->backward = backward_sub;
            break;
        case SIGMOID:
            gNode->backward = backward_sigmoid;
            break;
        case RELU:
            gNode->backward = backward_relu;
            break;
        case TANH:
            gNode->backward = backward_tanh;
            break;
        default:
            gNode->backward = NULL;
            break;
    }

    return gNode;
}

void backward_add(gNode_t *gNode)
{
    if(gNode == NULL) return;

    if(gNode->parents[0]) gNode->parents[0]->grad += gNode->grad;
    if(gNode->parents[1]) gNode->parents[1]->grad += gNode->grad;
}

void backward_sub(gNode_t *gNode)
{
    if(gNode == NULL) return;

    if(gNode->parents[0]) gNode->parents[0]->grad += gNode->grad;
    if(gNode->parents[1]) gNode->parents[1]->grad -= gNode->grad;

}

void backward_mul(gNode_t *gNode)
{
    if(gNode == NULL) return;

    if(gNode->parents[0]) gNode->parents[0]->grad += gNode->parents[1]->val->data[0] * gNode->grad;
    if(gNode->parents[1]) gNode->parents[1]->grad += gNode->parents[0]->val->data[0] * gNode->grad;

}

void backward_sigmoid(gNode_t *gNode)
{
    if(gNode == NULL) return;

    double sigmoid = 1.0f / (1.0f + exp(-gNode->val->data[0]));
    gNode->parents[0]->grad += sigmoid * (1 - sigmoid) * gNode->grad;
}

void backward_relu(gNode_t *gNode)
{
    if(gNode == NULL) return;

    double relu = gNode->val->data[0] > 0 ? 1 : 0;
    gNode->parents[0]->grad += relu * gNode->grad;
}

void backward_tanh(gNode_t *gNode)
{
    if(gNode == NULL) return;

    double th = tanh(gNode->val->data[0]);
    gNode->parents[0]->grad += (1 - th * th) * gNode->grad;
}

void free_gNode(gNode_t *gNode) 
{
    if(gNode == NULL) return;

    free_gNode(gNode->parents[0]);
    free_gNode(gNode->parents[1]);
    free(gNode->val);
    free(gNode->grad);
    free(gNode);
}

void free_gDAG(gDAG_t *gDAG)
{
    if(gDAG == NULL) return;

    free_gNode(gDAG->root);
    free(gDAG);
}

gVector_t* topological_sort_gDAG(gDAG_t *gDAG)
{
    gVector_t *gVector = gVector_create();
    if(gVector == NULL) return NULL;
    
    dfs(gDAG->root, gVector);

    return gVector;
}

static void dfs(gNode_t *gNode, gVector_t *gVector)
{   
    if(gNode == NULL) return;
    if(gNode->visited) return;

    gNode->visited = 1;

    dfs(gNode->parents[0], gVector);
    dfs(gNode->parents[1], gVector);

    int ret = gVector_push(gVector, (gNode_t*)gNode);
    if(ret) {
        printf("Topo-sort: Could not allocate enough space in vector\n");
        return;
    }
}

