//metrics.h
#ifndef METRICS_H
#define METRICS_H

#include <time.h>
#include <stddef.h>
#include "tensor.h"

// metrics.h — replace the two typedef forward-decls with:
#include "model.h"

// typedef struct Model Model;
// typedef enum InferenceMode InferenceMode;

size_t model_size_fp32(Model *m);
size_t model_size_int8(Model *m);
double measure_latency(Model *model, Tensor *input, Tensor *output, InferenceMode mode, int runs);
float quantization_mse(Tensor *original, Tensor *dequantized);

void log_quantization_results(
    const char *version,
    double latency_fp32,
    double latency_int8,
    float f1_fp32,
    float f1_int8,
    size_t size_fp32,
    size_t size_int8
);

#endif