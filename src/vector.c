#include "vector.h"
#include <stdlib.h>

gVector_t* gVector_create() {
    gVector_t* vec = (gVector_t*)malloc(sizeof(gVector_t));
    if (vec == NULL) return NULL;

    vec->data = NULL;
    vec->size = 0;
    vec->capacity = 0;

    return vec;
}

void gVector_free(gVector_t *vec) {
    if (vec == NULL) return;
    free(vec->data);
    free(vec);
}

int gVector_push(gVector_t *vec, void *item) {
    if (vec == NULL) return 1;

    if (vec->size == vec->capacity) {
        size_t new_capacity = vec->capacity == 0 ? 1 : vec->capacity * 2;
        if (gVector_resize(vec, new_capacity) != 0) return 1;
    }

    vec->data[vec->size++] = item;
    return 0;
}

void* gVector_pop(gVector_t *vec) {
    if (vec == NULL || vec->size == 0) return NULL;
    return vec->data[--vec->size];
}

void* gVector_get(gVector_t *vec, size_t index) {
    if (vec == NULL || index >= vec->size) return NULL;
    return vec->data[index];
}

size_t gVector_size(gVector_t *vec) {
    return vec ? vec->size : 0;
}

int gVector_resize(gVector_t *vec, size_t new_capacity) {
    if (vec == NULL) return 1;

    void **new_data = (void**)realloc(vec->data, new_capacity * sizeof(void*));
    if (new_data == NULL) return 1;

    vec->data = new_data;
    vec->capacity = new_capacity;
    return 0;
}