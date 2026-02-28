#ifndef QTENSOR_H
#define QTENSOR_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

typedef struct Tensor Tensor;

typedef struct {
    int rows;
    int cols;
    int8_t *data;
    float scale;
} QTensor;

/* Activation quantization */
QTensor* quantize_activation_symmetric(const Tensor *src);

/* Weight quantization (transposed for GEMM) */
QTensor* quantize_weight_symmetric_transposed(const Tensor *W);

/* Dequantize to FP32 */
Tensor* dequantize_tensor(const QTensor *qt);

/* Free */
void qtensor_free(QTensor *qt);

/* INT8 matmul */
void matmul_q8(
    const QTensor *A,
    const QTensor *B,
    Tensor *out
);

#endif