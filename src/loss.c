// #include "tensor.h"
// #include "loss.h"
// #include <math.h>
// #include <stdio.h>

// float cross_entropy_loss(const Tensor *logits, const int *target){
//     int rows = logits->rows;
//     int cols = logits->cols;
//     float loss = 0.0f;

//     for(int i=0;i<rows;i++){
//         if (target[i] == -1) continue;
//         int t = target[i];  // correct class index
//         float max_val = logits->data[i*cols];
//         for(int j=1;j<cols;j++)
//             if(logits->data[i*cols + j] > max_val) max_val = logits->data[i*cols + j];

//         float sum_exp = 0.0f;
//         for(int j=0;j<cols;j++)
//             sum_exp += expf(logits->data[i*cols + j] - max_val);

//         float log_prob = logits->data[i*cols + t] - max_val - logf(sum_exp);
//         loss -= log_prob;
//     }

//     return loss / rows;
// }

// void cross_entropy_loss_grad(const Tensor *logits, const int *target, Tensor *grad){
//     int rows = logits->rows, cols = logits->cols;
//     for(int i=0;i<rows;i++){
//         float sum_exp = 0.0f;
//         float max_val = logits->data[i*cols];
//         for(int j=1;j<cols;j++)
//             if(logits->data[i*cols + j] > max_val) max_val = logits->data[i*cols + j];

//         for(int j=0;j<cols;j++)
//             sum_exp += expf(logits->data[i*cols + j] - max_val);

//         for(int j=0;j<cols;j++){
//             float s = expf(logits->data[i*cols+j]-max_val)/sum_exp;
//             grad->data[i*cols+j] = s - (j==target[i]?1.0f:0.0f);
//             grad->data[i*cols+j] /= rows; // normalize
//         }
//     }
// }



#include "tensor.h"
#include "loss.h"
#include <math.h>
#include <stdio.h>

/*
  logits: [seq_len, num_classes]
  target: [seq_len]
  target[i] == -1  → ignore (PAD)
*/
// Class weights to counter CoNLL-2003 imbalance (~85% O tokens)
// O gets 1.0, all entity tags get higher weight so the model
// is penalized more for missing them.
// static const float CLASS_WEIGHTS[9] = {
//     0.3f,  // 0: O       ← DOWN-weighted heavily
//     8.0f,  // 1: B-PER
//     8.0f,  // 2: I-PER
//     8.0f,  // 3: B-ORG
//     8.0f,  // 4: I-ORG
//     8.0f,  // 5: B-LOC
//     8.0f,  // 6: I-LOC
//     6.0f,  // 7: B-MISC
//     6.0f   // 8: I-MISC
// };
static const float CLASS_WEIGHTS[9] = {
    0.2f,  // O        ← slightly higher than before
    5.0f,  // B-PER    ← slightly lower than before
    5.0f,  // I-PER
    5.0f,  // B-ORG
    5.0f,  // I-ORG
    5.0f,  // B-LOC
    5.0f,  // I-LOC
    4.0f,  // B-MISC
    4.0f   // I-MISC
};

float cross_entropy_loss(const Tensor *logits, const int *target) {
    int rows = logits->rows;
    int cols = logits->cols;

    float loss = 0.0f;
     
    float weight_sum = 0.0f;
    int valid_count = 0;

    for (int i = 0; i < rows; i++) {
        if (target[i] == -1) continue;

        int t = target[i];

        if (t < 0 || t >= cols) {
    printf("BAD TARGET: %d (cols=%d)\n", t, cols);
    exit(1);
}
        float w = CLASS_WEIGHTS[t];

        /* numerical stability */
        float max_val = logits->data[i * cols];
        for (int j = 1; j < cols; j++) {
            float v = logits->data[i * cols + j];
            if (v > max_val) max_val = v;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum_exp += expf(logits->data[i * cols + j] - max_val);
        }

        float log_prob =
            logits->data[i * cols + t] - max_val - logf(sum_exp);

        loss      -= w * log_prob;  // ← weighted loss
        weight_sum += w;
        valid_count++;
    }
    if (weight_sum == 0.0f || valid_count == 0) return 0.0f;
        if (weight_sum == 0.0f) return loss / weight_sum; 
        else return loss / valid_count; // normalize by valid tokens
    
          
    
    // if (valid_count == 0) return 0.0f;
    // return loss / valid_count;
}


void cross_entropy_loss_grad(
    const Tensor *logits,
    const int *target,
    Tensor *grad
) {
    int rows = logits->rows;
    int cols = logits->cols;

    tensor_zero(grad);

      float weight_sum = 0.0f;
    int valid_count = 0;
    for (int i = 0; i < rows; i++)
        if (target[i] != -1){
            weight_sum += CLASS_WEIGHTS[target[i]];
            valid_count++;
        }
            
    if (weight_sum == 0.0f) return;


    // if (valid_count == 0) return;

    for (int i = 0; i < rows; i++) {
        if (target[i] == -1) continue;

        int   t = target[i];
        float w = CLASS_WEIGHTS[t];

        float max_val = logits->data[i * cols];
        for (int j = 1; j < cols; j++) {
            float v = logits->data[i * cols + j];
            if (v > max_val) max_val = v;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum_exp += expf(logits->data[i * cols + j] - max_val);
        }

        for (int j = 0; j < cols; j++) {
            float softmax =
                expf(logits->data[i * cols + j] - max_val) / sum_exp;

             // gradient of weighted cross-entropy:
            // dL/dz_j = w * (softmax_j - 1{j==t}) / weight_sum
            grad->data[i * cols + j] = w * (softmax - (float)(j == t)) / weight_sum;
            // grad->data[i * cols + j] =
            //     (softmax - (j == target[i])) / valid_count;
        }
    }
}
