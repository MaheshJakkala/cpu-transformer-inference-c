// src/model.h
#ifndef MODEL_H
#define MODEL_H
#include "tensor.h"
#include "qtensor.h"

typedef enum {
    MODE_FP32 = 0,
    MODE_INT8  = 1
} InferenceMode;

typedef struct {
    Tensor  *Wq, *Wk, *Wv, *Wo;
    Tensor  *W1, *b1, *W2, *b2;
    Tensor  *Wcls, *bcls;
    QTensor *Wq_q, *Wk_q, *Wv_q, *Wo_q;
    QTensor *W1_q, *W2_q, *Wcls_q;
} Model;

#endif