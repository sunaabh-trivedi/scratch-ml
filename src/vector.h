#ifndef VECTOR_H
#define VECTOR_H

#include <stdlib.h>

typedef struct {
    void **data;
    size_t size;
    size_t capacity;
} gVector_t;

gVector_t* gVector_create();
void gVector_free(gVector_t *vec);
int gVector_push(gVector_t *vec, void *item);
void* gVector_pop(gVector_t *vec);
void* gVector_get(gVector_t *vec, size_t index);
size_t gVector_size(gVector_t *vec);
int gVector_resize(gVector_t *vec, size_t new_capacity);

#endif // VECTOR_H