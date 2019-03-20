#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Make.sh
# Usage      ./Make.sh

echo Excuting  Make.sh  ...

# ===== Execution ===========================================================

echo Building ONK_Lib ...
cd ONK_Lib
make
if [ 0 != $? ] ; then
    echo ERROR  ONK_Lib - make  failed
    exit 1
fi
cd ..

echo Building ONK_Pro1000 ...
cd ONK_Pro1000
make
if [ 0 != $? ] ; then
    echo ERROR  ONK_Pro1000 - make  failed
    exit 2
fi
cd ..

echo Building ONK_Test ...
cd ONK_Test
make
if [ 0 != $? ] ; then
    echo ERROR  ONK_Test - make  failed
    exit 3
fi
cd ..

echo Building ONK_Tunnel_IO ...
cd ONK_Tunnel_IO
make
if [ 0 != $? ] ; then
    echo ERROR  ONK_Tunnel_IO - make  failed
    exit 4
fi
cd ..

echo Building OpenNet ...
cd OpenNet
make
if [ 0 != $? ] ; then
    echo ERROR  OpenNet - make  failed
    exit 5
fi
cd ..

echo Building OpenNet_Test ...
cd OpenNet_Test
make
if [ 0 != $? ] ; then
    echo ERROR  OpenNet_Test - make  failed
    exit 6
fi
cd ..

echo Building TestLib ...
cd TestLib
make
if [ 0 != $? ] ; then
    echo ERROR  TestLib - make  failed
    exit 7
fi
cd ..

echo Building OpenNet_Tool ...
cd OpenNet_Tool
make
if [ 0 != $? ] ; then
    echo ERROR  OpenNet_Tool - make  failed
    exit 8
fi
cd ..

# ===== End =================================================================

echo OK
exit 0
