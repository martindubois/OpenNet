#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       CopyToTestComputer.sh
# Usage      ./CopyToTestComputer.sh

echo Excuting  CopyToTestComputer.sh  ...

# ===== Configuration =======================================================

TEST_COMPUTER=192.168.0.198

# ===== Initialisation ======================================================

DST_FOLDER=~/OpenNet

RESULT=0

# ===== Execution ===========================================================

scp Binaries/ONK_Test $TEST_COMPUTER:$DST_FOLDER/Binaries/ONK_Test
if [ 0 != $? ] ; then
    echo ERROR  scp Binaries/ONK_Test $TEST_COMPUTER:$DST_FOLDER/Binaries/ONK_Test  failed - $?
    RESULT=1
fi

scp Binaries/OpenNet_Test $TEST_COMPUTER:$DST_FOLDER/Binaries/OpenNet_Test
if [ 0 != $? ] ; then
    echo ERROR  scp Binaries/OpenNet_Test $TEST_COMPUTER:$DST_FOLDER/Binaries/OpenNet_Test  failed - $?
    RESULT=2
fi

scp Packages/kms-opennet_0.0-1.deb $TEST_COMPUTER:$DST_FOLDER/Packages
if [ 0 != $? ] ; then
    echo ERROR  scp Packages/kms-opennet-rt_0.0-0.deb $TEST_COMPUTER:$DST_FOLDER/Packages  failed - $?
    RESULT=3
fi

scp Scripts/OpenNet_Setup.sh $TEST_COMPUTER:$DST_FOLDER/Scripts
if [ 0 != $? ] ; then
    echo ERROR  scp Scripts/OpenNet_Setup.sh $TEST_COMPUTER:$DST_FOLDER/Scripts  failed - $?
    RESULT=4
fi

scp Scripts/OpenNet_Tool/*.txt $TEST_COMPUTER:$DST_FOLDER/Scripts/OpenNet_Tool
if [ 0 != $? ] ; then
    echo ERROR  scp Scripts/OpenNet_Tool/A00RRU_*_18.04.txt $TEST_COMPUTER:$DST_FOLDER/Scripts/OpenNet_Tool  failed - $?
    RESULT=5
fi

# ===== End =================================================================

if [ 0 = $RESULT ] ; then
    echo OK
fi

exit $RESULT
