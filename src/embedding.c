// src/embedding.c
#include "embedding.h"
#include <stdlib.h>
#include <stdio.h>
// embedding.c
#include <string.h>
#include "tensor.h"

Embedding *embedding_create(int vocab_size, int hidden_size) {
    Embedding *e = (Embedding*)malloc(sizeof(Embedding));
    e->weights = tensor_create(vocab_size, hidden_size);
    tensor_fill_random(e->weights, -0.01f, 0.01f);
    return e;
}

void embedding_free(Embedding *e) {
    if (!e) return;
    tensor_free(e->weights);
    free(e);
}

// Convert token IDs -> embedded vectors
// void embedding_forward(Embedding *e, int *input_ids, int seq_len, Tensor *out) {
//     printf(" inside embedding forward hi\n");
//     int hidden = e->weights->cols;
//     printf(" befor for loop inside embedding forward hi\n");
//     for (int i = 0; i < seq_len; ++i) {
//         int idx = input_ids[i];
//         printf("%dth iteration in outer for loop of embedding forward hi\n",i);
//         for (int j = 0; j < hidden; ++j) {
//             printf("%dth iteration in inner for loop of embedding forward hi\n",i);
//             out->data[i*hidden + j] = e->weights->data[idx*hidden + j];
//         }
//     }
    
// }

void embedding_forward(Embedding *e, int *input_ids, int seq_len, Tensor *out) {
    // printf(" inside embedding forward hi\n");
    int hidden = e->weights->cols;
    int vocab  = e->weights->rows;
    // printf(" befor for loop inside embedding forward hi\n");
    for (int i = 0; i < seq_len; ++i) {
        int idx = input_ids[i];

        if (idx < 0 || idx >= vocab) {
            idx = 0;   // UNK token
        }

        memcpy(
            out->data + i * hidden,
            e->weights->data + idx * hidden,
            hidden * sizeof(float)
        );
    }
    // printf(" ending inside embedding forward hi\n");
}

void embedding_backward(
    Embedding *e,
    int *input_ids,
    int seq_len,
    Tensor *grad_x,
    Tensor *grad_emb
) {
    int hidden = e->weights->cols;

    for (int i = 0; i < seq_len; i++) {
        int idx = input_ids[i];

        if (idx < 0 || idx >= e->weights->rows)
            continue;

        for (int j = 0; j < hidden; j++) {
            grad_emb->data[idx * hidden + j] +=
                grad_x->data[i * hidden + j];
        }
    }
}
