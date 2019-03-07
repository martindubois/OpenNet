#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Make.sh
# Usage      ./Make.sh

echo Excuting  Make.sh  ...

# ===== Execution ===========================================================

cd ONK_Lib
make
if [ 0 != $? ] ; then
    echo ERROR  ONK_Lib - make  failed - $?
    exit 1
fi
cd ..

cd ONK_Pro1000
make
if [ 0 != $? ] ; then
    echo ERROR  ONK_Pro1000 - make  failed - $?
    exit 2
fi
cd ..

cd ONK_Test
make
if [ 0 != $? ] ; then
    echo ERROR  ONK_Test - make  failed - $?
    exit 3
fi
cd ..

cd OpenNet
make
if [ 0 != $? ] ; then
    echo ERROR  OpenNet - make  failed - $?
    exit 4
fi
cd ..

cd OpenNet_Test
make
if [ 0 != $? ] ; then
    echo ERROR  OpenNet_Test - make  failed - $?
    exit 5
fi
cd ..

cd TestLib
make
if [ 0 != $? ] ; then
    echo ERROR  TestLib - make  failed - $?
    exit 6
fi
cd ..

cd OpenNet_Tool
make
if [ 0 != $? ] ; then
    echo ERROR  OpenNet_Tool - make  failed - $?
    exit 7
fi
cd ..

# ===== End =================================================================

echo OK
exit 0
