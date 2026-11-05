#!/bin/sh

set -e

if test $# -ne 1; then
  echo "Usage: $0 VERSION"
  echo
  echo "Where VERSION is the version of the program to test."
  echo "For example, to test version 1, run '$0 v1'."
  exit -1
fi

if test "$DATASET" = ""; then
  DATASET=standard
fi

version="$1"

ref_file=/tmp/hpc-lab-ref_$DATASET.txt
ver_file=$(mktemp)

if test ! -e "$ref_file"; then
  make clean VER=ref
  make build VER=ref DATASET=$DATASET DUMP_ARRAYS=1
  ./adi_ref 2> "$ref_file"
fi

make clean VER=$version
make build VER=$version DATASET=$DATASET DUMP_ARRAYS=1
./adi_$version 2> "$ver_file"

set +e

diff -q $ref_file $ver_file
echo "File tested: ref: $ref_file and ver: $ver_file"
if test $? -ne 0; then
  echo "------- Test failed -------"
else
  echo "------- Test passed -------"
fi
