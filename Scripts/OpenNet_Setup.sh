!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Scripts/OpenNet_Setup.sh
# Usage      See OpenNet_Setup

echo  Executing  OpenNet_Setup.sh $*  ...

# ===== Configuration =======================================================

OPEN_NET_INSTALL=/usr/local/OpenNet_1.0

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPEN_NET_INSTALL/bin:/usr/local/cuda-10.0/lib64

export LD_LIBRARY_PATH

# ===== Execution ===========================================================

$OPEN_NET_INSTALL/bin/OpenNet_Setup $*
if [ 0 != $? ] ; then
    echo ERROR  $OPEN_NET_INSTALL/bin/OpenNet_Setup $*  failed
    exit 1
fi

# ===== End =================================================================

echo OK
