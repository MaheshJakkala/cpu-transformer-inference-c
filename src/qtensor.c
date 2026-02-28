// //qtensor.c
// #include "qtensor.h"
// #include "tensor.h"

// QTensor* quantize_activation_symmetric(const Tensor *src) {
//     QTensor *qt = malloc(sizeof(QTensor));
//     qt->rows = src->rows;
//     qt->cols = src->cols;
//     qt->data = malloc(src->rows * src->cols * sizeof(int8_t));

//     float max = 0.0f;
//     for (int i = 0; i < src->rows * src->cols; i++)
//         if (fabsf(src->data[i]) > max)
//             max = fabsf(src->data[i]);

//     qt->scale = max / 127.0f + 1e-8f;

//     for (int i = 0; i < src->rows * src->cols; i++) {
//         int32_t v = (int32_t)roundf(src->data[i] / qt->scale);
//         if (v > 127) v = 127;
//         if (v < -128) v = -128;
//         qt->data[i] = (int8_t)v;
//     }

//     return qt;
// }

// QTensor* quantize_weight_transpose(const Tensor *W)
// {
//     int K = W->rows;   // in_dim
//     int N = W->cols;   // out_dim

//     QTensor *q = (QTensor*)malloc(sizeof(QTensor));
//     q->rows = N;   // TRANSPOSED
//     q->cols = K;
//     q->data = (int8_t*)malloc(N * K);

//     float max_abs = 0.0f;
//     for (int i = 0; i < K * N; i++)
//         if (fabsf(W->data[i]) > max_abs)
//             max_abs = fabsf(W->data[i]);

//     q->scale = max_abs / 127.0f;

//     for (int k = 0; k < K; k++) {
//         for (int n = 0; n < N; n++) {

//             float val = W->data[k*N + n];

//             int qv = (int)(val / q->scale);
//             if (qv > 127) qv = 127;
//             if (qv < -127) qv = -127;

//             // STORE TRANSPOSED
//             q->data[n*K + k] = (int8_t)qv;
//         }
//     }

//     return q;
// }

// void qtensor_free(QTensor *qt) {
//     free(qt->data);
//     free(qt);
// }

// void matmul_q8(
//     const QTensor *A,     // [M x K]
//     const QTensor *B,     // [K x N]
//     Tensor *out           // [M x N] float
// ) {
//     int M = A->rows;
//     int K = A->cols;
//     int N = B->cols;

//     float scale = A->scale * B->scale;

//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             int32_t acc = 0;
//             for (int k = 0; k < K; k++) {
//                 acc += (int32_t)A->data[i*K + k] *
//                        (int32_t)B->data[k*N + j];
//             }
//             out->data[i*N + j] = acc * scale;
//         }
//     }
// }



#include "qtensor.h"
#include "tensor.h"

/* ---------------- Activation Quant ---------------- */

QTensor* quantize_activation_symmetric(const Tensor *src)
{
    QTensor *qt = malloc(sizeof(QTensor));
    qt->rows = src->rows;
    qt->cols = src->cols;
    qt->data = malloc(src->rows * src->cols);

    float max_abs = 0.0f;
    int size = src->rows * src->cols;

    for (int i = 0; i < size; i++)
        if (fabsf(src->data[i]) > max_abs)
            max_abs = fabsf(src->data[i]);

    qt->scale = max_abs / 127.0f + 1e-8f;

    for (int i = 0; i < size; i++) {
        int qv = (int)roundf(src->data[i] / qt->scale);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        qt->data[i] = (int8_t)qv;
    }

    return qt;
}

/* ---------------- Weight Quant (TRANSPOSED) ---------------- */

QTensor* quantize_weight_symmetric_transposed(const Tensor *W)
{
    int K = W->rows;
    int N = W->cols;

    QTensor *qt = malloc(sizeof(QTensor));
    qt->rows = N;
    qt->cols = K;
    qt->data = malloc(N * K);

    float max_abs = 0.0f;
    for (int i = 0; i < K*N; i++)
        if (fabsf(W->data[i]) > max_abs)
            max_abs = fabsf(W->data[i]);

    qt->scale = max_abs / 127.0f + 1e-8f;

    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            float val = W->data[k*N + n];
            int qv = (int)roundf(val / qt->scale);
            if (qv > 127) qv = 127;
            if (qv < -128) qv = -128;

            qt->data[n*K + k] = (int8_t)qv;  // transpose
        }
    }

    return qt;
}

/* ---------------- Dequantize ---------------- */

Tensor* dequantize_tensor(const QTensor *qt)
{
    Tensor *t = tensor_create(qt->rows, qt->cols);

    int size = qt->rows * qt->cols;
    for (int i = 0; i < size; i++)
        t->data[i] = qt->data[i] * qt->scale;

    return t;
}

/* ---------------- INT8 Matmul ---------------- */

void matmul_q8(const QTensor *A, const QTensor *B, Tensor *out)
{
    int M = A->rows;
    int K = A->cols;
    int N = B->rows;  // because transposed

    float scale = A->scale * B->scale;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            int32_t acc = 0;

            for (int k = 0; k < K; k++) {
                acc += (int32_t)A->data[i*K + k] *
                       (int32_t)B->data[j*K + k];
            }

            out->data[i*N + j] = acc * scale;
        }
    }
}

void qtensor_free(QTensor *qt)
{
    free(qt->data);
    free(qt);
}