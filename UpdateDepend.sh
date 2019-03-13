#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       UpdateDepend.sh
# Usage      ./UpdateDepend.sh

echo Excuting  UpdateDepend.sh  ...

# ===== Initialisation ======================================================

RESULT=0

# ===== Execution ===========================================================

cd ONK_Lib
make depend > /dev/null 2> /dev/null
if [ 0 != $? ] ; then
    echo ERROR  ONK_Lib - make depend  failed - $?
    RESULT=1
fi
cd ..

# ONK_Pro1000 makefile do not support makedepend

cd ONK_Test
make depend > /dev/null 2> /dev/null
if [ 0 != $? ] ; then
    echo ERROR  ONK_Test - make depend  failed - $?
    RESULT=2
fi
cd ..

cd OpenNet
make depend > /dev/null 2> /dev/null
if [ 0 != $? ] ; then
    echo ERROR  OpenNet - make depend  failed - $?
    RESULT=3
fi
cd ..

cd OpenNet_Test
make depend > /dev/null 2> /dev/null
if [ 0 != $? ] ; then
    echo ERROR  OpenNet_Test - make depend  failed - $?
    RESULT=4
fi
cd ..

cd OpenNet_Tool
make depend > /dev/null 2> /dev/null
if [ 0 != $? ] ; then
    echo ERROR  OpenNet_Tool - make depend  failed - $?
    RESULT=5
fi
cd ..

cd TestLib
make depend > /dev/null 2> /dev/null
if [ 0 != $? ] ; then
    echo ERROR  TestLib - make depend  failed - $?
    RESULT=6
fi
cd ..

# ===== End =================================================================

if [ 0 = $RESULT ] ; then
    echo OK
fi

exit $RESULT
