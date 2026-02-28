//metrics.c
#include "metrics.h"

// size_t model_size_fp32(Model *m) {
//     size_t total = 0;

//     total += m->Wq->rows * m->Wq->cols * sizeof(float);
//     total += m->Wk->rows * m->Wk->cols * sizeof(float);
//     total += m->Wv->rows * m->Wv->cols * sizeof(float);
//     total += m->Wo->rows * m->Wo->cols * sizeof(float);
//     total += m->Wcls->rows * m->Wcls->cols * sizeof(float);

//     return total;
// size_t model_size_fp32(Model *m) {
//     size_t total = 0;

//     total += m->Wq->rows * m->Wq->cols * sizeof(float);
//     total += m->Wk->rows * m->Wk->cols * sizeof(float);
//     total += m->Wv->rows * m->Wv->cols * sizeof(float);
//     total += m->Wo->rows * m->Wo->cols * sizeof(float);
//     total += m->Wcls->rows * m->Wcls->cols * sizeof(float);

//     return total;
// }

// size_t model_size_int8(Model *m) {
//     size_t total = 0;

//     total += m->Wq_q->rows * m->Wq_q->cols * sizeof(int8_t);
//     total += m->Wk_q->rows * m->Wk_q->cols * sizeof(int8_t);
//     total += m->Wv_q->rows * m->Wv_q->cols * sizeof(int8_t);
//     total += m->Wo_q->rows * m->Wo_q->cols * sizeof(int8_t);
//     total += m->Wcls_q->rows * m->Wcls_q->cols * sizeof(int8_t);

//     return total;
// }

// }

// size_t model_size_int8(Model *m) {
//     size_t total = 0;

//     total += m->Wq_q->rows * m->Wq_q->cols * sizeof(int8_t);
//     total += m->Wk_q->rows * m->Wk_q->cols * sizeof(int8_t);
//     total += m->Wv_q->rows * m->Wv_q->cols * sizeof(int8_t);
//     total += m->Wo_q->rows * m->Wo_q->cols * sizeof(int8_t);
//     total += m->Wcls_q->rows * m->Wcls_q->cols * sizeof(int8_t);

//     return total;
// }

size_t tensor_size_fp32(const Tensor *t) {
    return t->rows * t->cols * sizeof(float);
}

size_t qtensor_size_int8(const QTensor *t) {
    return t->rows * t->cols * sizeof(int8_t)
           + sizeof(float); // scale
}

size_t model_size_fp32(Model *m) {
    size_t total = 0;

    total += tensor_size_fp32(m->Wq);
    total += tensor_size_fp32(m->Wk);
    total += tensor_size_fp32(m->Wv);
    total += tensor_size_fp32(m->Wo);
    total += tensor_size_fp32(m->W1);
    total += tensor_size_fp32(m->W2);
    total += tensor_size_fp32(m->Wcls);

    total += tensor_size_fp32(m->b1);
    total += tensor_size_fp32(m->b2);
    total += tensor_size_fp32(m->bcls);

    return total;
}

size_t model_size_int8(Model *m) {
    size_t total = 0;

    total += qtensor_size_int8(m->Wq_q);
    total += qtensor_size_int8(m->Wk_q);
    total += qtensor_size_int8(m->Wv_q);
    total += qtensor_size_int8(m->Wo_q);
    total += qtensor_size_int8(m->W1_q);
    total += qtensor_size_int8(m->W2_q);
    total += qtensor_size_int8(m->Wcls_q);

    total += tensor_size_fp32(m->b1);
    total += tensor_size_fp32(m->b2);
    total += tensor_size_fp32(m->bcls);

    return total;
}

double measure_latency(Model *model, Tensor *input, Tensor *output, InferenceMode mode, int runs) {

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < runs; i++) {
        inference_forward(model, input, output, mode);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_sec =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    return (time_sec / runs) * 1000.0; // ms
}

float quantization_mse(Tensor *original, Tensor *dequantized) {

    float mse = 0.0f;
    int size = original->rows * original->cols;

    for (int i = 0; i < size; i++) {
        float diff = original->data[i] - dequantized->data[i];
        mse += diff * diff;
    }

    return mse / size;
}

void log_quantization_results(
    const char *version,
    double latency_fp32,
    double latency_int8,
    float f1_fp32,
    float f1_int8,
    size_t size_fp32,
    size_t size_int8
) {
    FILE *fp = fopen("benchmarks/quantization_results.csv", "a");

    double reduction = 100.0 * (1.0 - ((double)size_int8 / size_fp32));

    //append the results to the file
    fprintf(fp,
        "%s,%.4f,%.4f,%.4f,%.4f,%zu,%zu\n",
        version,
        latency_fp32,
        latency_int8,
        f1_fp32,
        f1_int8,
        size_fp32,
        size_int8
    );

    fclose(fp);
}

