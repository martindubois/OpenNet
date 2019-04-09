#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Build.sh
# Usage      ./Build.sh {Major.Minor.Build_Type}

echo Excuting  Build.sh $1  ...

# ===== Execution ===========================================================

./Clean.sh

./Make.sh
if [ 0 != $? ] ; then
    echo ERROR  ./Make.sh  failed
    exit 1
fi

./CreatePackages.sh
if [ 0 != $? ] ; then
    echo ERROR  ./CreatePackages.sh  failed
    exit 1
fi

./Export.sh $1
if [ 0 != $? ] ; then
    echo ERROR  ./Export.sh $1  failed
    exit 1
fi

# ===== End =================================================================

echo OK
exit 0
