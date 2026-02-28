//evaluation.h
#ifndef EVALUATION_H
#define EVALUATION_H

// typedef struct {
//     float precision;
//     float recall;
//     float f1;
// } Metrics;

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
// #include <time.h>

// typedef struct Model Model;
// typedef enum InferenceMode InferenceMode;

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include "tensor.h"
#include "ops.h"
#include "linear.h"
#include "qtensor.h"
#include "layers.h"
#include "layernorm.h"
#include "activations.h"
#include "attention.h"
#include "ffn.h"
#include "transformer_block.h"
#include "loss.h"
#include "optimizer.h"
#include "tokenizer.h"
#include "embedding.h"
#include "metrics.h"

#include "model.h"

#include "config.h"
#include "../data/data.h"
// #define MAX_SEQ 128
#include <math.h>


typedef struct {
    long TP, FP, FN, TN;
} Metrics;

typedef struct {
    int start;   // token index
    int end;     // inclusive
    int type;    // PER / ORG / LOC
} Entity;


int argmax(const Tensor *logits, int row);
// void eval(const Tensor *logits, const int *target, int seq_len);
// void test_model(
//     Embedding *emb,
//     Tokenizer *tok,
//     Tensor *Wq, Tensor *Wk, Tensor *Wv, Tensor *Wo,
//     Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2,
//     Tensor *W_cls, Tensor *b_cls,
//     char **val_texts,
//     int  **val_labels
// );

void test_model(
    Embedding *emb,
    Tokenizer *tok,
    Tensor *Wq, Tensor *Wk, Tensor *Wv, Tensor *Wo,
    Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2,
    Tensor *W_cls, Tensor *b_cls,
    char **val_texts,
    int  **val_labels,
    int    num_samples
);

void save_dataset(
    const char *filename,
    char texts[][128],
    int  labels[][MAX_SEQ_LEN],
    int  num_samples
);

void test_model_q8(
    Embedding *emb,
    Tokenizer *tok,
    Tensor *Wq, Tensor *Wk, Tensor *Wv, Tensor *Wo,
    Tensor *W1, Tensor *b1,
    Tensor *W2, Tensor *b2,
    Tensor *W_cls, Tensor *b_cls,
    char **texts,
    int **labels,
    int n_samples
);

// evaluation.h
float evaluate_model(
    Model     *model,
    Embedding *emb,
    char      **val_texts,
    int       **val_labels,
    int         val_samples,
    InferenceMode mode
);

#endif