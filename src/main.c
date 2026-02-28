// main.c
#include "main.h"


void inference_forward(Model *model, Tensor *input, Tensor *output, InferenceMode mode)
{
    int seq_len = input->rows;
    Tensor *logits = tensor_create(seq_len, NUM_CLASSES);
    AttentionCache cache = {0};
    Tensor *ff1 = NULL;

    int padding_mask[MAX_SEQ_LEN];
    for (int i = 0; i < seq_len; i++) padding_mask[i] = 1;

    if (mode == MODE_FP32) {
        Tensor *ff1_pre_inf = NULL;  // not needed for inference
        transformer_block_forward(
            input,
            model->Wq, model->Wk, model->Wv, model->Wo,
            model->W1, model->b1, model->W2, model->b2,
            output, &ff1, &ff1_pre_inf, &cache, padding_mask
        );
        if (ff1_pre_inf) tensor_free(ff1_pre_inf);
        linear_forward(output, model->Wcls, model->bcls, logits);
    } else {
        QTensor *x_q = quantize_activation_symmetric(input);
        transformer_block_forward_q8(
            x_q,
            model->Wq_q, model->Wk_q, model->Wv_q, model->Wo_q,
            model->W1_q, model->b1,
            model->W2_q, model->b2,
            output, &ff1, &cache, padding_mask
        );
        QTensor *out_q = quantize_activation_symmetric(output);
        linear_forward_int8(out_q, model->Wcls_q, model->bcls, logits);
        qtensor_free(x_q);
        qtensor_free(out_q);
    }

    // cleanup
    tensor_free(logits);
    if (ff1) tensor_free(ff1);
    tensor_free(cache.Q); tensor_free(cache.K); tensor_free(cache.V);
    tensor_free(cache.scores); tensor_free(cache.attn); tensor_free(cache.attn_out);
}

void train(
    char train_texts[TRAIN_SAMPLES][128],
    int  train_labels[TRAIN_SAMPLES][MAX_SEQ_LEN],
    char val_texts[VAL_SAMPLES][128],
    int  val_labels[VAL_SAMPLES][MAX_SEQ_LEN]
)
{
    // srand(time(NULL));
    //   int seq = 5;        // number of samples
    // // int hidden = 8;     // feature size
    int hidden = HIDDEN_SIZE;
    int num_classes = NUM_CLASSES;            
    
    Tokenizer *tok = tokenizer_create();
    if (!tok) {
        printf("Failed to create tokenizer\n");
        return;
    }

    /* --------------------------------------------------
    2. Create embedding
    -------------------------------------------------- */
    Embedding *emb = embedding_create(tok->vocab_size, hidden);
    Tensor *grad_emb = tensor_create(tok->vocab_size, hidden);
    tensor_zero(grad_emb);

    if (!emb) {
        printf("Failed to create embedding\n");
        return;
    }
     


    // // --- INCREASED TOY DATASET ---
    // int seq = 6;      // sequence length (was 2)
    // int hidden = 8;   // hidden size / feature dimension (was 4)

    // Tensor *x = tensor_create(seq, hidden);
    // tensor_fill_random(x, -1.0f, 1.0f);

    // int target[6] = {0, 1, 2, 3, 4, 5};  // updated target


    // ================= DATA =================
    // const char *text ="The Amazon is the largest rainforest and Amazon is expanding rapidly";


        /* ---------- Tokenize ---------- */
        // int input_ids[MAX_SEQ_LEN];
        // int seq_len;
        // encode_word(texts[s], input_ids, MAX_SEQ_LEN, &seq_len);

        /* ---------- Embedding ---------- */
        // Tensor *x = tensor_create(seq_len, hidden);
        // embedding_forward(emb, input_ids, seq_len, x);

        /* ---------- Positional signal (VERY IMPORTANT) ---------- */
        // printf("%d before for loop hi\n",epoch);
    // const char *texts[NUM_SAMPLES] = {
    //     "Amazon is hiring",
    //     "Google is a company",
    //     "I love India",
    //     "Microsoft builds software"
    // };

    // int labels[NUM_SAMPLES][MAX_SEQ_LEN] = {
    //     {1,0,0},      // Amazon → ORG
    //     {1,0,0,0},    // Google → ORG
    //     {0,0,2},      // India → LOC
    //     {1,0,0,0}     // Microsoft → ORG
    // };

    // texts = train_texts;
    // labels =train_labels;

    // int token_ids[MAX_SEQ];
    // int seq_len = 0;

    // encode_word(text, token_ids, MAX_SEQ, &seq_len);
    
//    int seq = seq_len;   // THIS is seq_len
//     if (seq != 11) {
//         printf("Token/target mismatch: seq=%d\n", seq);
//         exit(1);
//     }


//    Tensor *x = tensor_create(seq, hidden);
//     tensor_zero(x);

//     for (int i = 0; i < seq; i++) {
//         int tid = token_ids[i];
//         for (int j = 0; j < hidden; j++) {
//             x->data[i * hidden + j] = emb->weights->data[tid * hidden + j];
//         }
//     }

//     // O=0, LOCATION=1, ORGANIZATION=2
//     int target[] = {
//         0,  // The
//         1,  // Amazon (LOCATION)
//         0,  // is
//         0,  // the
//         0,  // largest
//         0,  // rainforest
//         0,  // and
//         2,  // Amazon (ORGANIZATION)
//         0,  // is
//         0,  // expanding
//         0   // rapidly
//     };

    // int target[5] = {0, 1, 2, 3, 4};



    // Transformer weights
    Tensor *Wq = tensor_create(hidden, hidden); tensor_fill_random(Wq, -0.01f, 0.01f);
    Tensor *Wk = tensor_create(hidden, hidden); tensor_fill_random(Wk, -0.01f, 0.01f);
    Tensor *Wv = tensor_create(hidden, hidden); tensor_fill_random(Wv, -0.01f, 0.01f);
    Tensor *Wo = tensor_create(hidden, hidden); tensor_fill_random(Wo, -0.01f, 0.01f);

    Tensor *W1 = tensor_create(hidden, hidden*2);
    Tensor *b1 = tensor_create(1, hidden*2);
    tensor_fill_random(W1, -0.01f, 0.01f);
    tensor_fill_random(b1, -0.01f, 0.01f);

    Tensor *W2 = tensor_create(hidden*2, hidden);
    Tensor *b2 = tensor_create(1, hidden);
    tensor_fill_random(W2, -0.01f, 0.01f);
    tensor_fill_random(b2, -0.01f, 0.01f);

    
    //quantized tensors
    // QTensor *Wq_q = quantize_tensor(Wq);
    // QTensor *Wk_q = quantize_tensor(Wk);
    // QTensor *Wv_q = quantize_tensor(Wv);
    // QTensor *Wo_q = quantize_tensor(Wo);

    // QTensor *W1_q = quantize_tensor(W1);
    // QTensor *W2_q = quantize_tensor(W2);

    // QTensor *Wcls_q = quantize_tensor(W_cls);

    // Tensor *out = tensor_create(seq, hidden);

    // Adam optimizers
    AdamState *opt_Wq = adam_create(LR, 0.9, 0.999, 1e-8, hidden, hidden);
    AdamState *opt_Wk = adam_create(LR, 0.9, 0.999, 1e-8, hidden, hidden);
    AdamState *opt_Wv = adam_create(LR, 0.9, 0.999, 1e-8, hidden, hidden);
    AdamState *opt_Wo = adam_create(LR, 0.9, 0.999, 1e-8, hidden, hidden);
    AdamState *opt_W1 = adam_create(LR, 0.9, 0.999, 1e-8, hidden, hidden*2);
    AdamState *opt_b1 = adam_create(LR, 0.9, 0.999, 1e-8, 1, hidden*2);
    AdamState *opt_W2 = adam_create(LR, 0.9, 0.999, 1e-8, hidden*2, hidden);
    AdamState *opt_b2 = adam_create(LR, 0.9, 0.999, 1e-8, 1, hidden);

    //
    AdamState *opt_W_cls = adam_create(LR, 0.9, 0.999, 1e-8, hidden, num_classes);
    AdamState *opt_b_cls = adam_create(LR, 0.9, 0.999, 1e-8, 1, num_classes);
    //for embeddings
    AdamState *opt_emb = adam_create(
    LR, 0.9, 0.999, 1e-8,
    tok->vocab_size, hidden
);


    // Gradients
    // Tensor *grad_out = tensor_create(seq, hidden);
    // Tensor *grad_x   = tensor_create(seq, hidden);

    Tensor *grad_Wq = tensor_create(hidden, hidden);
    Tensor *grad_Wk = tensor_create(hidden, hidden);
    Tensor *grad_Wv = tensor_create(hidden, hidden);
    Tensor *grad_Wo = tensor_create(hidden, hidden);

    Tensor *grad_W1 = tensor_create(hidden, hidden*2);
    Tensor *grad_b1 = tensor_create(1, hidden*2);
    Tensor *grad_W2 = tensor_create(hidden*2, hidden);
    Tensor *grad_b2 = tensor_create(1, hidden);

    // Tensor *ff1 = NULL;


    Tensor *W_cls = tensor_create(hidden, num_classes);
    Tensor *b_cls = tensor_create(1, num_classes);
    
    tensor_fill_random(W_cls, -0.01f, 0.01f);
    tensor_zero(b_cls);

    // Tensor *logits = tensor_create(seq, num_classes);

    // Tensor *grad_logits = tensor_create(seq, num_classes);
    Tensor *grad_W_cls = tensor_create(hidden, num_classes);
    Tensor *grad_b_cls = tensor_create(1, num_classes);

    Model model = {
    .Wq = Wq, .Wk = Wk, .Wv = Wv, .Wo = Wo,
    .W1 = W1, .b1 = b1, .W2 = W2, .b2 = b2,
    .Wcls = W_cls, .bcls = b_cls,
    .Wq_q = NULL, .Wk_q = NULL, .Wv_q = NULL, .Wo_q = NULL,
    .W1_q = NULL, .W2_q = NULL,
    .Wcls_q = NULL
};
    

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0;
        
        for (int s = 0; s < TRAIN_SAMPLES; ++s) {


        int input_ids[MAX_SEQ_LEN] = {0};

        int seq_len;
        
        
        // printf("%d before encode word hi\n",epoch);
        encode_word(train_texts[s], input_ids, MAX_SEQ_LEN, &seq_len);
        int padding_mask[MAX_SEQ_LEN] = {0};

        for (int i = 0; i < seq_len; i++)
            padding_mask[i] = 1;   // real tokens

        // if(epoch<=5)
        //     printf("encode word: %d\n",*input_ids);
        // encode_word(texts[s], input_ids, MAX_SEQ_LEN, &seq_len);
        // printf("Sentence: \"%s\" | seq_len=%d\n", texts[s], seq_len);


if (seq_len > MAX_SEQ_LEN) {
    seq_len = MAX_SEQ_LEN;
}

        Tensor *x = tensor_create(seq_len, hidden);
        Tensor *out = tensor_create(seq_len, hidden);
        Tensor *logits = tensor_create(seq_len, num_classes);
        Tensor *grad_logits = tensor_create(seq_len, num_classes);
        Tensor *grad_out = tensor_create(seq_len, hidden);
        Tensor *grad_x = tensor_create(seq_len, hidden);
        
        Tensor *ff1 = tensor_create(seq_len, W1->cols);
        Tensor *ff1_pre = tensor_create(seq_len, W1->cols);
        // printf("%d before embedding forward hi\n",epoch);
        embedding_forward(emb, input_ids, seq_len, x);

        // QTensor *x_q = quantize_tensor(x);
        // QTensor *x_q = quantize_weight_symmetric_transposed(x);

//         if(epoch<=5){
//             printf("Token ID: %d\n", *input_ids);
// printf("Embedding vector: ");
//             for (int j = 0; j < 5; j++) {
//                 printf("%f ", x->data[j]);
//             }
//             printf("...\n");
//         }
        // printf("%d after embedding forward hi\n",epoch);
            AttentionCache cache = {0};
        // for (int i = 0; i < seq_len; i++)
        //     x->data[i*hidden + (i % hidden)] += 0.01f;

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < hidden; j++) {
                float angle = i / powf(10000.0f, (2.0f * (j / 2)) / hidden);
                if (j % 2 == 0)
                    x->data[i * hidden + j] += 0.1f * sinf(angle);
                else
                    x->data[i * hidden + j] += 0.1f * cosf(angle);
            }
        }
        // printf("%d after for loop hi\n",epoch);
        // if (ff1) {
        //     tensor_free(ff1);
        //     ff1 = NULL;
        // }

                // ================== FORWARD ==================
        // printf("%d before transformer block forward hi\n",epoch);
        // transformer_block_forward(
        //     x, Wq, Wk, Wv, Wo,
        //     W1, b1, W2, b2,
        //     out, &ff1, &cache,padding_mask
        // );
        // ← HERE (line ~after embedding_forward in the training loop)
        
        transformer_block_forward(
            x, Wq, Wk, Wv, Wo,
            W1, b1, W2, b2,
            out, &ff1, &ff1_pre, &cache, padding_mask
        );

        if (out->cols != hidden) {
            printf("FATAL: out->cols=%d hidden=%d\n", out->cols, hidden);
            exit(1);
        }
        // printf("hi\n");
        /* ---------- Classifier ---------- */
        // printf("%d before linear forward in main hi\n",epoch);
        linear_forward(out, W_cls, b_cls, logits);

        // float loss = cross_entropy_loss(logits, target);
//           float loss = cross_entropy_loss(logits, labels[s]);
// cross_entropy_loss_grad(logits, labels[s], grad_logits);
        int target[MAX_SEQ_LEN];

        for (int i = 0; i < seq_len; i++) {
            int lbl = train_labels[s][i];
            target[i] = (lbl >= 0 && lbl < num_classes) ? lbl : -1;
        }
        // real labels
        
        // for (int i = 0; i < seq_len; i++) {
        //     if (i < label_len[s])
        //         target[i] = labels[s][i];
        //     else
        //         target[i] = -1;  // ignore extra tokens
        // }

        for (int i = seq_len; i < MAX_SEQ_LEN; i++)
            target[i] = -1;             // PAD → ignored

// O tag
        // for (int i = 0; i < MAX_SEQ_LEN; i++)
        //     labels[s][i] = 0; // default O

// for (int i = 0; i < label_len[s]; i++)
//     target[i] = labels[s][i];
        
// if (epoch == 0) {
// printf("Tokens vs labels:\n"); // printf("Epoch %d, Loss=%.4f\n", epoch, loss);
// for (int i = 0; i < seq_len; i++) {
//     printf("token_id=%d label=%d\n",
//             input_ids[i],
//             train_labels[s][i]);
// }
// }


        // printf("%d before cross entropy loss hi\n",epoch);
        float loss = cross_entropy_loss(logits, target);
        // printf("%d before cross entropy loss grad hi\n",epoch);
        cross_entropy_loss_grad(logits, target, grad_logits);

        epoch_loss += loss;

        // float loss = cross_entropy_loss(out, target);
        // printf("Epoch %d, Loss=%.4f\n", epoch, loss);


        // cross_entropy_loss_grad(out, target, grad_out);

        // cross_entropy_loss_grad(logits, target, grad_logits);
        tensor_zero(grad_W_cls);
        tensor_zero(grad_b_cls);


        /* ---------- Backward ---------- */
        linear_backward(out, W_cls, grad_logits,grad_out, grad_W_cls, grad_b_cls);


        // ================== ZERO GRADS ==================
        tensor_zero(grad_x);
        tensor_zero(grad_Wq); tensor_zero(grad_Wk);
        tensor_zero(grad_Wv); tensor_zero(grad_Wo);
        tensor_zero(grad_W1); tensor_zero(grad_b1);
        tensor_zero(grad_W2); tensor_zero(grad_b2);

        // ================== BACKWARD ==================
        // Tensor *grad_attn_out = tensor_create(seq, hidden);
        Tensor *grad_attn_out = tensor_create(seq_len, hidden);

        
        // ffn_backward(
        //     cache.attn_out,  // FFN input
        //     ff1,             // GELU output
        //     grad_out,        // dL/d(out)
        //     W2, W1,
        //     grad_W2, grad_b2,
        //     grad_W1, grad_b1,
        //     grad_attn_out
        // );
        ffn_backward(
            cache.attn_out,  // FFN input (post-residual+LN, fixed in transformer_block)
            ff1,             // POST-GELU intermediate (for linear_backward w.r.t W2)
            ff1_pre,         // PRE-GELU intermediate (for correct gelu_backward)
            grad_out,        // dL/d(out)
            W2, W1,
            grad_W2, grad_b2,
            grad_W1, grad_b1,
            grad_attn_out
        );

        attention_backward(
            x, Wq, Wk, Wv, Wo,
            grad_attn_out, &cache,
            grad_x, grad_Wq, grad_Wk, grad_Wv, grad_Wo
        );

        tensor_free(grad_attn_out);
        tensor_zero(grad_emb);
        embedding_backward(
            emb,
            input_ids,
            seq_len,
            grad_x,
            grad_emb
        );

        adam_step(emb->weights, grad_emb, opt_emb);


        /* ---------- Optimizer ---------- */
        // ================== ADAM UPDATE ==================
        adam_step(Wq, grad_Wq, opt_Wq);
        adam_step(Wk, grad_Wk, opt_Wk);
        adam_step(Wv, grad_Wv, opt_Wv);
        adam_step(Wo, grad_Wo, opt_Wo);
        adam_step(W1, grad_W1, opt_W1);
        adam_step(b1, grad_b1, opt_b1);
        adam_step(W2, grad_W2, opt_W2);
        adam_step(b2, grad_b2, opt_b2);
        /* inside training loop */
        adam_step(W_cls, grad_W_cls, opt_W_cls);
        adam_step(b_cls, grad_b_cls, opt_b_cls);


        //  // ================== PRINT GRADIENTS ==================
        // printf("grad_Wq[0..5]: ");
        // for (int i = 0; i < 6 && i < grad_Wq->rows*grad_Wq->cols; ++i)
        //     printf("%.6f ", grad_Wq->data[i]);
        // printf("\n");

        // printf("grad_Wk[0..5]: ");
        // for (int i = 0; i < 6 && i < grad_Wk->rows*grad_Wk->cols; ++i)
        //     printf("%.6f ", grad_Wk->data[i]);
        // printf("\n");

        // printf("grad_Wv[0..5]: ");
        // for (int i = 0; i < 6 && i < grad_Wv->rows*grad_Wv->cols; ++i)
        //     printf("%.6f ", grad_Wv->data[i]);
        // printf("\n");

        // printf("grad_x[0..5]: ");
        // for (int i = 0; i < 6 && i < grad_x->rows*grad_x->cols; ++i)
        //     printf("%.6f ", grad_x->data[i]);
        // printf("\n");

        // printf("Attention weights (cache.attn first row): ");
        // for (int j = 0; j < seq; ++j)
        //     printf("%.4f ", cache.attn->data[j]);
        // printf("\n-----------------------------------\n");

        /* ---------- Cleanup ---------- */
        // ================== FREE CACHE ==================
tensor_free(cache.Q);        cache.Q = NULL;
tensor_free(cache.K);        cache.K = NULL;
tensor_free(cache.V);        cache.V = NULL;
tensor_free(cache.scores);   cache.scores = NULL;
tensor_free(cache.attn);     cache.attn = NULL;
tensor_free(cache.attn_out); cache.attn_out = NULL;


        
        tensor_free(x);
tensor_free(out);
tensor_free(logits);
tensor_free(grad_logits);
tensor_free(grad_out);
tensor_free(grad_x);

    // if (ff1){ tensor_free(ff1);
    //     ff1 = NULL;}
    // }
    // if (ff1_pre){ tensor_free(ff1_pre);
    //     ff1_pre = NULL;}
        if (ff1) {
            tensor_free(ff1);
            ff1 = NULL;
        }

        if (ff1_pre) {
            tensor_free(ff1_pre);
            ff1_pre = NULL;
        }

        }
    printf("Epoch %d | Avg Loss = %.4f\n",epoch, epoch_loss / TRAIN_SAMPLES);
    
    // printf("Hi\n");
    }

    char *val_text_ptrs[VAL_SAMPLES];
    int  *val_label_ptrs[VAL_SAMPLES];

    for (int i = 0; i < VAL_SAMPLES; i++) {
        val_text_ptrs[i]  = val_texts[i];
        val_label_ptrs[i] = val_labels[i];
    }

    
    // test_model(
    //     emb, tok,
    //     Wq, Wk, Wv, Wo,
    //     W1, b1, W2, b2,
    //     W_cls, b_cls,
    //     val_text_ptrs,
    //     val_label_ptrs,
    //     VAL_SAMPLES
    // );
    model.Wq_q  = quantize_weight_symmetric_transposed(model.Wq);
    model.Wk_q  = quantize_weight_symmetric_transposed(model.Wk);
    model.Wv_q  = quantize_weight_symmetric_transposed(model.Wv);
    model.Wo_q  = quantize_weight_symmetric_transposed(model.Wo);
    model.Wcls_q = quantize_weight_symmetric_transposed(model.Wcls);
    model.W1_q = quantize_weight_symmetric_transposed(model.W1);
    model.W2_q = quantize_weight_symmetric_transposed(model.W2);

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    float f1_fp32 = evaluate_model(&model, emb, val_text_ptrs, val_label_ptrs, VAL_SAMPLES, MODE_FP32);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double latency_fp32 = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_nsec - t1.tv_nsec)/1e6;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    float f1_int8 = evaluate_model(&model, emb, val_text_ptrs, val_label_ptrs, VAL_SAMPLES, MODE_INT8);

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double latency_int8 = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_nsec - t1.tv_nsec)/1e6;

    size_t size_fp32 = model_size_fp32(&model);
    size_t size_int8 = model_size_int8(&model);



    log_quantization_results(
        "baseline_int8",
        latency_fp32,
        latency_int8,
        f1_fp32,
        f1_int8,
        size_fp32,
        size_int8
    );



    // Free all tensors
    // tensor_free(x); tensor_free(out);
    tensor_free(Wq); tensor_free(Wk); tensor_free(Wv); tensor_free(Wo);
    tensor_free(W1); tensor_free(b1); tensor_free(W2); tensor_free(b2);
    // tensor_free(grad_out); tensor_free(grad_x);
    tensor_free(grad_Wq); tensor_free(grad_Wk); tensor_free(grad_Wv); tensor_free(grad_Wo);
    tensor_free(grad_W1); tensor_free(grad_b1); tensor_free(grad_W2); tensor_free(grad_b2);

    // Free Adam states
    adam_free(opt_Wq); adam_free(opt_Wk); adam_free(opt_Wv); adam_free(opt_Wo);
    adam_free(opt_W1); adam_free(opt_b1); adam_free(opt_W2); adam_free(opt_b2);

    embedding_free(emb);
    tokenizer_free(tok);

}

// void generate_sentence1(
//     char *buffer,
//     int *labels,
//     int max_len,
//     int *seq_len,
//     int is_org,
//     int *input_ids
// ) {
//     const char *entity = is_org ?
//         ORG[rand() % 8] : LOC[rand() % 8];

//     const char *tpl = TEMPLATES[rand() % 5];
//     snprintf(buffer, 128, tpl, entity);

//     encode_word(buffer, input_ids, max_len, seq_len);

//     for (int i = 0; i < *seq_len; i++)
//         labels[i] = 0;          // O

//     labels[0] = is_org ? 1 : 2; // ORG or LOC
// }

// void generate_data(
//     char train_texts[TRAIN_SAMPLES][128],
//     int  train_labels[TRAIN_SAMPLES][MAX_SEQ_LEN],
//     char val_texts[VAL_SAMPLES][128],
//     int  val_labels[VAL_SAMPLES][MAX_SEQ_LEN]
// ){
//     for (int i = 0; i < TRAIN_SAMPLES; i++)
//     {
//         int is_org = rand() % 2;
//         int ids[MAX_SEQ_LEN], len;

//         generate_sentence1(
//             train_texts[i],
//             train_labels[i],
//             MAX_SEQ_LEN,
//             &len,
//             is_org,
//             ids
//         );
//     }

//     for (int i = 0; i < VAL_SAMPLES; i++) 
//     {
//         int is_org = rand() % 2;
//         int ids[MAX_SEQ_LEN], len;

//         generate_sentence1(
//             val_texts[i],
//             val_labels[i],
//             MAX_SEQ_LEN,
//             &len,
//             is_org,
//             ids
//         );
//     }
// }


int main(){
    srand(42);

    char train_texts[TRAIN_SAMPLES][128];
    int  train_labels[TRAIN_SAMPLES][MAX_SEQ_LEN];

    char val_texts[VAL_SAMPLES][128];
    int  val_labels[VAL_SAMPLES][MAX_SEQ_LEN];

    int train_n = 0, val_n = 0;

    load_conll_dataset(
        "data/conll2003/train.json",
        train_texts,
        train_labels,
        TRAIN_SAMPLES,
        &train_n
    );

    load_conll_dataset(
        "data/conll2003/test.json",
        val_texts,
        val_labels,
        VAL_SAMPLES,
        &val_n
    );

    printf("Loaded %d train, %d val samples\n", train_n, val_n);

    printf("Sentences: \n");
    for(int i =0;i<3;i++){
        for(int j=0;j<5;j++)
        printf("%s, labels: %d\n",train_texts[i],train_labels[i][j]);
    }

    train(train_texts,train_labels,val_texts,val_labels);

    // test();
    // eval(logits,target);
    // CPUFirstProof();
    // memoryfootprint();
    return 0;
}


