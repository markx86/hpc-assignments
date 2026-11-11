VER ?= ref
DATASET ?= mini
DUMP_ARRAYS ?=
NTHREADS ?= $(shell nproc)

CC = gcc

CFLAGS = \
	-O2 \
	-fopenmp \
	-Icommon/ \
	-D$(shell echo $(DATASET) | tr '[:lower:]' '[:upper:]')_DATASET \
	-DPOLYBENCH_TIME \
	-DPOLYBENCH_NTHREADS=$(NTHREADS) \
	-lm

ifneq ($(DUMP_ARRAYS),)
	CFLAGS += -DPOLYBENCH_DUMP_ARRAYS
endif

TARGET = adi_$(VER)

ALL_SRCS = $(shell find . -type f -name "adi_*.c")
ALL_TARGETS = $(ALL_SRCS:.c=)

.PHONY: build
build: $(TARGET)

.PHONY: run
run: $(TARGET)
	@echo "Running $<"
	@./$<

.PHONY: all
all: $(ALL_TARGETS)

%: %.c common/polybench.c
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean
clean:
	find . -type f -name "adi_*" -and -executable -delete
