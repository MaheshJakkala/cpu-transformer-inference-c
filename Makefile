# Makefile

CC = gcc
CFLAGS = -O3 -Wall -std=c11
LDFLAGS = -lm

CFLAGS += -fopenmp
LDFLAGS += -fopenmp

CFLAGS += -mavx2 -mfma

SRC = src/tensor.c \
      src/qtensor.c\
      src/ops.c \
      src/layers.c \
      src/layernorm.c \
      src/activations.c \
      src/linear.c \
      src/attention.c \
      src/transformer_block.c \
      src/loss.c \
      src/optimizer.c \
      src/ffn.c \
      src/tokenizer.c\
      src/embedding.c\
      data/data.c\
      data/cJSON.c\
      src/evaluation.c\
      src/metrics.c\
      src/main.c

OBJ = $(SRC:.c=.o)
TARGET = llm

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
