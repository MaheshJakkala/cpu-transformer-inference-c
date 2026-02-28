  //main.h
#ifndef MAIN_H
#define MAIN_H

  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  #include<stdint.h>
  #include <time.h>
  #include<math.h>
  #include "tensor.h"
  #include "ops.h"
  #include "linear.h"
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
  #include "evaluation.h"
  #include "metrics.h"
  #include "config.h"
  #include "qtensor.h"

  #include "model.h"

  #if defined(__has_include)
    #if __has_include("data/data.h")
      #include "data/data.h"
    #elif __has_include("../data/data.h")
      #include "../data/data.h"
    #elif __has_include("data.h")
      #include "data.h"
    #else
      #error "data/data.h not found; add the data directory to your include path or place data.h next to this file"
    #endif
  #else
    /* Fallback: try the original include and rely on the build include paths */
    #include "data/data.h"
  #endif


// typedef struct {
//     // FP32 weights
//     Tensor *Wq, *Wk, *Wv, *Wo;
//     Tensor *W1, *b1, *W2, *b2;
//     Tensor *Wcls, *bcls;

//     // INT8 weights
//     QTensor *Wq_q, *Wk_q, *Wv_q, *Wo_q;
//     QTensor *W1_q, *W2_q;
//     QTensor *Wcls_q;
// } Model;

// typedef enum {
//     MODE_FP32 = 0,
//     MODE_INT8 = 1
// } InferenceMode;
// ypedef struct {
//     // FP32 weights
//     Tensor *Wq, *Wk, *Wv, *Wo;
//     Tensor *W1, *b1, *W2, *b2;
//     Tensor *Wcls, *bcls;

//     // INT8 weights
//     QTensor *Wq_q, *Wk_q, *Wv_q, *Wo_q;
//     QTensor *W1_q, *W2_q;
//     QTensor *Wcls_q;
// } Model;

// typedef enum {
//     MODE_FP32 = 0,
//     MODE_INT8 = 1
// } InferenceMode;

void inference_forward(Model *model, Tensor *input, Tensor *output, InferenceMode mode);


  // void generate_sentence1(
  //     char *buffer,
  //     int *labels,
  //     int max_len,
  //     int *seq_len,
  //     int is_org,
  //     int *input_ids
  // );
  // void generate_data(
  //     char train_texts[TRAIN_SAMPLES][128],
  //     int  train_labels[TRAIN_SAMPLES][MAX_SEQ_LEN],
  //     char val_texts[VAL_SAMPLES][128],
  //     int  val_labels[VAL_SAMPLES][MAX_SEQ_LEN]
  // );

  #endif