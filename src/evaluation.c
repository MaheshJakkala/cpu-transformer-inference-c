//evaluation.c
#include "evaluation.h"

// src/labels.c or utils.c
const char* label_to_bio(int label){
    switch (label) {
        case O_TAG: return "O";
        case B_PER: return "B-PER";
        case I_PER: return "I-PER";
        case B_ORG: return "B-ORG";
        case I_ORG: return "I-ORG";
        case B_LOC: return "B-LOC";
        case I_LOC: return "I-LOC";
        case B_MISC: return "B-MISC";
        case I_MISC: return "I-MISC";
        default:    return "UNK";
    }
}


int argmax(const Tensor *logits, int row) {
    int cols = logits->cols;
    int best = 0;
    float maxv = logits->data[row * cols];
    for (int j = 1; j < cols; j++) {
        float v = logits->data[row * cols + j];
        if (v > maxv) {
            maxv = v;
            best = j;
        }
    }
    return best;
}
// void CPUFirstProof()
// {
//     struct timespec t1, t2;
//     clock_gettime(CLOCK_MONOTONIC, &t1);

//     // forward pass only
//     embedding_forward(...);
//     transformer_block_forward(...);
//     linear_forward(...);

//     clock_gettime(CLOCK_MONOTONIC, &t2);

//     double latency_ms =
//         (t2.tv_sec - t1.tv_sec) * 1000.0 +
//         (t2.tv_nsec - t1.tv_nsec) / 1e6;

//     printf("Latency: %.3f ms\n", latency_ms);

//     double tokens_per_sec = MAX_SEQ_LEN / (latency_ms / 1000.0);
//     printf("Throughput: %.2f tokens/sec\n", tokens_per_sec);


// }

void memoryfootprint(Embedding *emb)
{
    size_t param_bytes = 0;

    param_bytes += emb->weights->rows * emb->weights->cols * sizeof(float);
    param_bytes += HIDDEN_SIZE * HIDDEN_SIZE * 4 * sizeof(float); // Q,K,V,O
    param_bytes += HIDDEN_SIZE * HIDDEN_SIZE * 2 * sizeof(float); // FFN
    param_bytes += HIDDEN_SIZE * NUM_CLASSES * sizeof(float);

    printf("Model parameters: %.2f MB\n", param_bytes / (1024.0*1024.0));

    size_t activation_bytes =
    MAX_SEQ_LEN * HIDDEN_SIZE * sizeof(float) * 6; // x, Q,K,V,out,ffn

    printf("Peak activation memory: %.2f MB\n",activation_bytes / (1024.0*1024.0));
}

// }
// void eval(const Tensor *logits, const int *target, int seq_len)
// {   
//     long TP = 0, FP = 0, FN = 0, TN = 0;

//     for (int i = 0; i < seq_len; i++)
//     {
//         if (target[i] == -1) continue;

//         int pred = argmax(logits, i);
//         int gold = target[i];

//         if (pred == gold && gold != 0) TP++;
//         else if (pred != gold && pred != 0) FP++;
//         else if (pred != gold && gold != 0) FN++;
//         else TN++;
//     }

//     float precision = TP / (float)(TP + FP + 1e-8);
//     float recall    = TP / (float)(TP + FN + 1e-8);
//     float f1        = 2 * precision * recall / (precision + recall + 1e-8);

//     printf("Precision: %.4f\n", precision);
//     printf("Recall   : %.4f\n", recall);
//     printf("F1-score : %.4f\n", f1);


// }

void eval_accumulate(
    const Tensor *logits,
    const int *target,
    int seq_len,
    Metrics *m
){
    for (int i = 0; i < seq_len; i++) {
        if (target[i] == -1) continue;

        int pred = argmax(logits, i);
        int gold = target[i];

        if (pred == gold && gold != O_TAG) m->TP++;
        else if (pred != gold && pred != O_TAG) m->FP++;
        else if (pred != gold && gold != O_TAG) m->FN++;
        else m->TN++;
    }
}
int extract_entities(
    const int *labels,
    int seq_len,
    Entity *ents,
    int max_ents
){
    int count = 0;

    for (int i = 0; i < seq_len; i++) {
        int lbl = labels[i];

        if (lbl == O_TAG || lbl == PAD) continue;

        int type = lbl;  // store full label (B-PER etc.)

        if (lbl == B_PER || lbl == B_ORG || lbl == B_LOC || lbl == B_MISC ||
           ((lbl == I_PER || lbl == I_ORG || lbl == I_LOC || lbl == I_MISC) &&
            (i == 0 || labels[i-1] == O_TAG)))
        {
            int start = i;
            int j = i + 1;

            while (j < seq_len) {
                int next = labels[j];
                if (next == O_TAG || next == PAD) break;
                if (next == B_PER || next == B_ORG || next == B_LOC || lbl == B_MISC) break;
                j++;
            }

            if (count < max_ents) {
                ents[count++] = (Entity){start, j - 1, type};
            }

            i = j - 1;
        }
    }
    return count;
}

int entity_match(Entity a, Entity b){
    return a.start == b.start &&
           a.end   == b.end &&
           a.type  == b.type;
}
void eval_entities(
    const Tensor *logits,
    const int *gold,
    int seq_len,
    long *TP,
    long *FP,
    long *FN
){
    int pred_labels[MAX_SEQ_LEN];

    for (int i = 0; i < seq_len; i++)
        pred_labels[i] = argmax(logits, i);

    Entity gold_ents[32];
    Entity pred_ents[32];

    int gcount = extract_entities(gold, seq_len, gold_ents, 32);
    int pcount = extract_entities(pred_labels, seq_len, pred_ents, 32);

    int matched[32] = {0};

    // True positives
    for (int i = 0; i < pcount; i++) {
        for (int j = 0; j < gcount; j++) {
            if (!matched[j] && entity_match(pred_ents[i], gold_ents[j])) {
                (*TP)++;
                matched[j] = 1;
                goto next_pred;
            }
        }
        (*FP)++;
    next_pred:;
    }

    // False negatives
    for (int j = 0; j < gcount; j++) {
        if (!matched[j]) (*FN)++;
    }
}


void test_model(
    Embedding *emb,
    Tokenizer *tok,
    Tensor *Wq, Tensor *Wk, Tensor *Wv, Tensor *Wo,
    Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2,
    Tensor *W_cls, Tensor *b_cls,
    char **val_texts,
    int  **val_labels,
    int  num_samples
) {
    // Metrics m = {0};
    double latency_ms = 0.0;
    long TP = 0, FP = 0, FN = 0;


    for (int s = 0; s < num_samples; ++s)
    {
        int input_ids[MAX_SEQ_LEN];
        int seq_len;

        printf("sentence: %s\n",val_texts[s]);
        encode_word(val_texts[s], input_ids, MAX_SEQ_LEN, &seq_len);

        Tensor *x      = tensor_create(seq_len, HIDDEN_SIZE);
        Tensor *out    = tensor_create(seq_len, HIDDEN_SIZE);
        Tensor *logits = tensor_create(seq_len, NUM_CLASSES);

        int padding_mask[MAX_SEQ_LEN] = {0};
        for (int i = 0; i < seq_len; i++) padding_mask[i] = 1;

              /* Transformer */
        AttentionCache cache = {0};
        Tensor *ff1 = NULL;

            struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        /* Embedding */
        embedding_forward(emb, input_ids, seq_len, x);
        //  QTensor *x_q = quantize_tensor(x);
        QTensor *x_q = quantize_weight_symmetric_transposed(x);
        /* Positional signal */
        for (int i = 0; i < seq_len; i++)
            x->data[i * HIDDEN_SIZE + (i % HIDDEN_SIZE)] += 0.01f;

        // printf("\nhellowww\n");
        // printf("x shape: (%d,%d) , wq shape: (%d,%d)\n",x->rows,x->cols,Wq->rows,Wq->cols);
        Tensor *ff1_pre = NULL;

        transformer_block_forward(
            x, Wq, Wk, Wv, Wo,
            W1, b1, W2, b2,
            out, &ff1, &ff1_pre, &cache, padding_mask
        );

        printf("\nhellowww\n");
        /* Classifier */
        // linear_forward(out, W_cls, b_cls, logits);
        //  QTensor *out_q = quantize_tensor(out);
        QTensor *out_q = quantize_weight_symmetric_transposed(out);

        //   QTensor *Wcls_q = quantize_tensor(W_cls);
        QTensor *Wcls_q = quantize_weight_symmetric_transposed(W_cls);

        linear_forward_int8(out_q, Wcls_q, b_cls, logits);
        // linear_forward_q8(out_q, Wcls_q, logits);

        // optional: bias
        for (int i = 0; i < logits->rows; i++)
            for (int j = 0; j < logits->cols; j++)
                logits->data[i*logits->cols+j] += b_cls->data[j];


        printf("hello in eval\n");
        clock_gettime(CLOCK_MONOTONIC, &t2);

        latency_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 +(t2.tv_nsec - t1.tv_nsec) / 1e6;
        
        char tokens[MAX_SEQ_LEN][32];

        encode_word_with_tokens(
            val_texts[s],
            input_ids,
            tokens,
            MAX_SEQ_LEN,
            &seq_len
        );


       int target[MAX_SEQ_LEN];

        for (int i = 0; i < seq_len; i++) {
            int lbl = val_labels[s][i];
            target[i] = (lbl >= 0 && lbl < NUM_CLASSES) ? lbl : -1;
        }

        printf("\n=== Inference ===\n");
        for (int i = 0; i < seq_len; i++) {
            int pred = argmax(logits, i);
            printf(
                "token=\"%s\" (id=%d) → %s , target:%s\n",
                tokens[i],
                input_ids[i],
                label_to_bio(pred),
                label_to_bio(target[i])
            );
        }
        eval_entities(logits, target, seq_len, &TP, &FP, &FN);




        /* Cleanup */
        tensor_free(x);
        tensor_free(out);
        tensor_free(logits);
        if (ff1) tensor_free(ff1);
        // if (ff1) tensor_free(ff1);
        if (ff1_pre) tensor_free(ff1_pre);

        tensor_free(cache.Q);
        tensor_free(cache.K);
        tensor_free(cache.V);
        tensor_free(cache.scores);
        tensor_free(cache.attn);
        tensor_free(cache.attn_out);

        qtensor_free(x_q);
        qtensor_free(out_q);
        qtensor_free(Wcls_q);

    }

    float precision = TP / (float)(TP + FP + 1e-8);
    float recall    = TP / (float)(TP + FN + 1e-8);
    float f1        = 2 * precision * recall / (precision + recall + 1e-8);

    printf("\n=== CoNLL Entity-Level Evaluation ===\n");
    printf("Precision: %.4f\n", precision);
    printf("Recall   : %.4f\n", recall);
    printf("F1-score : %.4f\n", f1);


    printf("Latency: %.3f ms\n", latency_ms);

    double tokens_per_sec = MAX_SEQ_LEN / (latency_ms / 1000.0);
    printf("Throughput: %.2f tokens/sec\n", tokens_per_sec);

    memoryfootprint(emb);
}

void print_metrics(const Metrics *m){
    float precision = m->TP / (float)(m->TP + m->FP + 1e-8);
    float recall    = m->TP / (float)(m->TP + m->FN + 1e-8);
    float f1        = 2 * precision * recall / (precision + recall + 1e-8);

    printf("Precision: %.4f\n", precision);
    printf("Recall   : %.4f\n", recall);
    printf("F1-score : %.4f\n", f1);
}


// void save_dataset(
//     const char *filename,
//     char texts[][128],
//     int  labels[][MAX_SEQ_LEN],
//     int  num_samples
// ){
//     FILE *fp = fopen(filename, "w");
//     if (!fp) {
//         perror("fopen");
//         return;
//     }

//     for (int s = 0; s < num_samples; s++) {
//         fprintf(fp, "SENTENCE: %s\n", texts[s]);
//         fprintf(fp, "LABELS: ");

//         for (int i = 0; i < MAX_SEQ_LEN; i++) {
//             if (labels[s][i] == -1) break;
//             fprintf(fp, "%d ", labels[s][i]);
//         }
//         fprintf(fp, "\n\n");
//     }

//     fclose(fp);
// }

// void test_model_q8(
//     Embedding *emb, Tokenizer *tok,
//     Tensor *Wq, Tensor *Wk, Tensor *Wv, Tensor *Wo,
//     Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2,
//     Tensor *W_cls, Tensor *b_cls,
//     char **texts,
//     int **labels,
//     int n_samples
// ){
//     /* ---------------- Quantize ONCE ---------------- */
//     // QTensor *Wq_q  = quantize_tensor(Wq);
//     // QTensor *Wk_q  = quantize_tensor(Wk);
//     // QTensor *Wv_q  = quantize_tensor(Wv);
//     // QTensor *Wo_q  = quantize_tensor(Wo);
//     // QTensor *W1_q  = quantize_tensor(W1);
//     // QTensor *W2_q  = quantize_tensor(W2);
//     // QTensor *Wcls_q = quantize_tensor(W_cls);

//     // QTensor *Wq_q = quantize_weight_transpose(Wq);
//     // QTensor *Wk_q = quantize_weight_transpose(Wk);
//     // QTensor *Wv_q = quantize_weight_transpose(Wv);
//     // QTensor *Wo_q = quantize_weight_transpose(Wo);
//     // QTensor *W1_q = quantize_weight_transpose(W1);
//     // QTensor *W2_q = quantize_weight_transpose(W2);
//     // QTensor *Wcls_q = quantize_weight_transpose(W_cls);

//     QTensor *Wq_q = quantize_weight_symmetric_transposed(Wq);
//     QTensor *Wk_q = quantize_weight_symmetric_transposed(Wk);
//     QTensor *Wv_q = quantize_weight_symmetric_transposed(Wv);
//     QTensor *Wo_q = quantize_weight_symmetric_transposed(Wo);
//     QTensor *W1_q = quantize_weight_symmetric_transposed(W1);
//     QTensor *W2_q = quantize_weight_symmetric_transposed(W2);
//     QTensor *Wcls_q = quantize_weight_symmetric_transposed(W_cls);

//     Metrics m = {0};

//     struct timespec t1, t2;
//     clock_gettime(CLOCK_MONOTONIC, &t1);
//     Tensor *W_deq = dequantize_tensor(model.Wq_q);
// float mse = quantization_mse(model.Wq, W_deq);
//     // printf("\\-----------hi--------------\\");
//     for (int s = 0; s < n_samples; s++) {
//         printf("sentence: %s\n",texts[s]);
//         int input_ids[MAX_SEQ_LEN];
//         int seq_len;
//         encode_word(texts[s], input_ids, MAX_SEQ_LEN, &seq_len);

//         Tensor *x = tensor_create(seq_len, HIDDEN_SIZE);
//         embedding_forward(emb, input_ids, seq_len, x);

//         QTensor *x_q = quantize_activation_symmetric(x);
//         // QTensor *x_q = quantize_weight_transpose(x);

//         Tensor *out = tensor_create(seq_len, HIDDEN_SIZE);
//         AttentionCache cache = {0};

//         int padding_mask[MAX_SEQ_LEN] = {0};
//         for (int i = 0; i < seq_len; i++) padding_mask[i] = 1;

//         /* -------- QUANTIZED ATTENTION -------- */
//         // printf("\\-----------before att_for_q8--------------\\");
//         attention_forward_q8(
//             x_q,
//             Wq_q, Wk_q, Wv_q, Wo_q,
//             padding_mask,
//             out,
//             &cache
//         );

//         Tensor *logits = tensor_create(seq_len, NUM_CLASSES);

//         QTensor *out_q = quantize_activation_symmetric(out);
//         // QTensor *out_q = quantize_weight_transpose(out);

//         // printf("\\-----------before last_linear_for_q8--------------\\");
//         linear_forward_q8(out_q, Wcls_q, b_cls, logits);

//         free_qtensor(out_q);

//         eval_accumulate(logits, labels[s], seq_len, &m);
// //  printf("\n=== Inference ===\n");
// //         for (int i = 0; i < seq_len; i++) {
// //             int pred = argmax(logits, i);
// //             printf(
// //                 "token=\"%s\" (id=%d) → %s , target:%s\n",
// //                 tokens[i],
// //                 input_ids[i],
// //                 label_to_bio(pred),
// //                 label_to_bio(target[i])
// //             );
// //         }


//         tensor_free(x);
//         tensor_free(out);
//         tensor_free(logits);
//         free_qtensor(x_q);
//         attention_cache_free(&cache);
//     }

//     clock_gettime(CLOCK_MONOTONIC, &t2);

//     double ms =
//         (t2.tv_sec - t1.tv_sec) * 1000.0 +
//         (t2.tv_nsec - t1.tv_nsec) / 1e6;

//     double tokens = n_samples * MAX_SEQ_LEN;

//     double latency_fp32 = measure_latency(... MODE_FP32 ...);
// double latency_int8 = measure_latency(... MODE_INT8 ...);

//     printf("\n=== Quantized Inference ===\n");
//     printf("Latency: %.3f ms\n", ms);
//     printf("Throughput: %.2f tokens/sec\n", tokens / (ms / 1000.0));

//     print_metrics(&m);

//     /* ---------------- Memory ---------------- */
//     size_t qbytes =
//         Wq_q->rows * Wq_q->cols +
//         Wk_q->rows * Wk_q->cols +
//         Wv_q->rows * Wv_q->cols +
//         Wo_q->rows * Wo_q->cols;

//     double speedup = latency_fp32 / latency_int8;
// double reduction = 100 * (1 - (double) size_int8 / size_fp32);
//     // double reduction = 100.0 * (1.0 - ((double)size_int8 / size_fp32));
//     printf("INT8 parameter memory: %.2f MB\n",
//            qbytes / (1024.0 * 1024.0));

    

//     free_qtensor(Wq_q); free_qtensor(Wk_q);
//     free_qtensor(Wv_q); free_qtensor(Wo_q);
//     free_qtensor(W1_q); free_qtensor(W2_q);
//     free_qtensor(Wcls_q);
// }


// float quant_error(const Tensor *f, const QTensor *q){
//     float err = 0;
//     for (int i = 0; i < f->rows * f->cols; i++) {
//         float dq = q->data[i] * q->scale;
//         err += fabsf(f->data[i] - dq);
//     }
//     return err / (f->rows * f->cols);
// }

/* ---------- Evaluate Model ---------- */

// evaluation.h
float evaluate_model(
    Model     *model,
    Embedding *emb,
    char      **val_texts,
    int       **val_labels,
    int         val_samples,
    InferenceMode mode
){
    long TP = 0, FP = 0, FN = 0;

    for (int s = 0; s < val_samples; s++) {

        int input_ids[MAX_SEQ_LEN];
        int seq_len;

        encode_word(val_texts[s], input_ids, MAX_SEQ_LEN, &seq_len);

        Tensor *x = tensor_create(seq_len, HIDDEN_SIZE);
        Tensor *out = tensor_create(seq_len, HIDDEN_SIZE);
        Tensor *logits = tensor_create(seq_len, NUM_CLASSES);

        embedding_forward(emb, input_ids, seq_len, x);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                float angle = i / powf(10000.0f, (2.0f * (j / 2)) / HIDDEN_SIZE);
                if (j % 2 == 0)
                    x->data[i * HIDDEN_SIZE + j] += 0.1f * sinf(angle);
                else
                    x->data[i * HIDDEN_SIZE + j] += 0.1f * cosf(angle);
            }
        }


        
        int padding_mask[MAX_SEQ_LEN] = {0};
        for (int i = 0; i < seq_len; i++) padding_mask[i] = 1;
        
        // Add positional signal HERE, before forward pass:
        for (int i = 0; i < seq_len; i++)
            x->data[i * HIDDEN_SIZE + (i % HIDDEN_SIZE)] += 0.01f;

        // Then do transformer_block_forward...
        AttentionCache cache = {0};
        Tensor *ff1 = NULL;
        Tensor *ff1_pre = NULL;

        if (mode == MODE_FP32) {

            // Tensor *ff1_pre = NULL;
            

            transformer_block_forward(
                x,
                model->Wq, model->Wk,
                model->Wv, model->Wo,
                model->W1, model->b1,
                model->W2, model->b2,
                out, &ff1,&ff1_pre, &cache, padding_mask
            );

            linear_forward(out, model->Wcls, model->bcls, logits);

        } else {

            QTensor *x_q = quantize_activation_symmetric(x);

            transformer_block_forward_q8(
                x_q,
                model->Wq_q, model->Wk_q,
                model->Wv_q, model->Wo_q,
                model->W1_q, model->b1,   // ← fixed
                model->W2_q, model->b2,   // ← fixed
                out, &ff1, &cache, padding_mask
            );

            QTensor *out_q = quantize_activation_symmetric(out);

            linear_forward_int8(
                out_q,
                model->Wcls_q,
                model->bcls,
                logits
            );

            qtensor_free(x_q);
            qtensor_free(out_q);
        }


        eval_entities(logits, val_labels[s], seq_len, &TP, &FP, &FN);
        // In evaluate_model, after linear_forward / linear_forward_int8, before eval_entities:

        // ===== DEBUG: print first 3 samples =====
        if (s < 3) {
            printf("\n--- Sample %d (mode=%s) ---\n", s, mode == MODE_FP32 ? "FP32" : "INT8");
            
            // Print raw logit stats
            float max_logit = -1e9, min_logit = 1e9;
            for (int i = 0; i < seq_len * NUM_CLASSES; i++) {
                if (logits->data[i] > max_logit) max_logit = logits->data[i];
                if (logits->data[i] < min_logit) min_logit = logits->data[i];
            }
            printf("Logit range: [%.4f, %.4f]\n", min_logit, max_logit);
            
            // Count predicted classes
            int class_counts[NUM_CLASSES] = {0};
            for (int i = 0; i < seq_len; i++) {
                int pred = argmax(logits, i);
                class_counts[pred]++;
            }
            printf("Predicted class distribution: ");
            for (int c = 0; c < NUM_CLASSES; c++)
                if (class_counts[c] > 0)
                    printf("class%d=%d ", c, class_counts[c]);
            printf("\n");
            
            // Per-token output
            for (int i = 0; i < seq_len; i++) {
                int pred = argmax(logits, i);
                int gold = val_labels[s][i];
                
                // Print logit row
                printf("  tok[%d] gold=%-8s pred=%-8s | logits: ", 
                    i, label_to_bio(gold), label_to_bio(pred));
                for (int c = 0; c < NUM_CLASSES; c++)
                    printf("%.2f ", logits->data[i * NUM_CLASSES + c]);
                printf("\n");
            }
        }

        tensor_free(x);
        tensor_free(out);
        tensor_free(logits);

        // Add this before the end of the for-loop in evaluate_model:
        if (ff1) tensor_free(ff1);
        if (ff1_pre) tensor_free(ff1_pre);
        tensor_free(cache.Q); tensor_free(cache.K); tensor_free(cache.V);
        tensor_free(cache.scores); tensor_free(cache.attn); tensor_free(cache.attn_out);
    }

    float precision = TP / (float)(TP + FP + 1e-8);
    float recall    = TP / (float)(TP + FN + 1e-8);
    float f1        = 2 * precision * recall / (precision + recall + 1e-8);

    printf("Precision: %.4f\n", precision);
    printf("Recall   : %.4f\n", recall);
    printf("F1-score : %.4f\n", f1);


    return f1;
}



