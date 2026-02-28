// ffn.h
#ifndef FFN_H
#define FFN_H

#include "tensor.h"
#include "linear.h"
#include "activations.h"
#include "qtensor.h"
#include <stdio.h>

void ffn_forward(
    const Tensor *input,
    const Tensor *W1, const Tensor *b1,
    const Tensor *W2, const Tensor *b2,
    Tensor *out, Tensor *temp1
);

// ff1_post: POST-GELU activations (for linear_backward w.r.t W2)
// ff1_pre:  PRE-GELU activations  (for correct gelu_backward derivative)
void ffn_backward(
    const Tensor *attn_out,
    const Tensor *ff1_post,
    const Tensor *ff1_pre,
    const Tensor *grad_out,
    const Tensor *W2,
    const Tensor *W1,
    Tensor *grad_W2,
    Tensor *grad_b2,
    Tensor *grad_W1,
    Tensor *grad_b1,
    Tensor *grad_attn_out
);

void ffn_forward_q8(
    const QTensor *input_q,
    const QTensor *W1_q,
    const Tensor  *b1,
    const QTensor *W2_q,
    const Tensor  *b2,
    Tensor *out,
    Tensor *temp1
);

#endif