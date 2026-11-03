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
	-DNTHREADS=$(NTHREADS) \
	-lm

ifneq ($(DUMP_ARRAYS),)
	CFLAGS += -DPOLYBENCH_DUMP_ARRAYS
endif

TARGET = adi_$(VER)

.PHONY: run
run: $(TARGET)
	@echo "Running $<"
	@./$<

build: $(TARGET)

$(TARGET): $(TARGET).c common/polybench.c
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean
clean:
	find . -type f -name "adi_*" -and -executable -delete
