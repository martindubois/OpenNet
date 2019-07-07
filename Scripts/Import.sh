#!/bin/sh

# Author   KMS - Martin Dubois, ing.
# Product  OpenNet
# File     Import.sh
# Usage    ./Import.sh {Destination}

echo Executing  Import.sh $1  ...

# ===== Verification =========================================================

if [ ! -d $1 ]
then
    echo ERROR  $1  does not exist
    exit 1
fi

# ===== Execution ============================================================

mkdir $1/Binaries
mkdir $1/Includes
mkdir $1/Includes/OpenNet
mkdir $1/Includes/OpenNetK
mkdir $1/Libraries
mkdir $1/Modules

cp Binaries/*.so          $1/Binaries
cp Binaries/OpenNet_Tool  $1/Binaries
cp Includes/OpenNet/*.h   $1/Includes/OpenNet
cp Includes/OpenNetK/*.h  $1/Includes/OpenNetK
cp Libraries/ONK_Lib.a    $1/Libraries
cp Modules/ONK_Pro1000.ko $1/Modules

# ===== End ==================================================================

echo OK
