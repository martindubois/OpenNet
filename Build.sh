#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Build.sh
# Usage      ./Build.sh

echo Excuting  Build.sh  ...

# ===== Execution ===========================================================

./Clean.sh

./Make.sh

if [ 0 != $? ] ; then
    echo ERROR  ./Make.sh  failed - $?
    exit 1
fi

# ===== End =================================================================

echo OK
exit 0
