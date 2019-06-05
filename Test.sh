#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Test.sh
# Usage      ./Test.sh

echo Excuting  Build.sh  ...

# ===== Initialisation ======================================================

BINARIES=./Binaries

CUDA=/usr/local/cuda-10.0

# ===== Verification ========================================================

if [ ! -d $CUDA ]
then
    echo FATAL ERROR  $CUDA  does not exist
    echo Install CUDA Toolkit 10.0
    exit 1
fi

# ===== Execution ===========================================================

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BINARIES:$CUDA/lib64::$CUDA/lib64/stubs
export LD_LIBRARY_PATH

$BINARIES/OpenNet_Test
if [ 0 != $? ] ; then
    echo ERROR  $BINARIES/OpenNet_Test  failed
    exit 1
fi

$BINARIES/ONK_Test
if [ 0 != $? ] ; then
    echo ERROR  $BINARIES/ONK_Test  failed
    exit 2
fi

# ===== End =================================================================

echo OK
