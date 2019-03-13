#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Export.sh
# Usage      ./Export.sh {Major.Minor.Build_Type}

echo Executing  Export.sh $1  ...

# ===== Initialisation ======================================================

DST_FOLDER=~/Export/OpenNet/$1_Linux_$(uname -r)

# ===== Execution ===========================================================

mkdir $DST_FOLDER
if [ 0 != $? ] ; then
    echo ERROR  mkdir $DST_FOLDER  failed
    exit 1
fi

mkdir $DST_FOLDER/Binaries
mkdir $DST_FOLDER/Includes
mkdir $DST_FOLDER/Includes/OpenNet
mkdir $DST_FOLDER/Includes/OpenNetK
mkdir $DST_FOLDER/Libraries
mkdir $DST_FOLDER/Modules
mkdir $DST_FOLDER/ONK_Lib
mkdir $DST_FOLDER/OpenNet
mkdir $DST_FOLDER/OpenNet_Tool
mkdir $DST_FOLDER/Packages
mkdir $DST_FOLDER/Scripts
mkdir $DST_FOLDER/Scripts/OpenNet_Tool

cp _DocUser/ReadMe.txt              $DST_FOLDER
cp Binaries/ONK_Test                $DST_FOLDER/Binaries
cp Binaries/OpenNet_Test            $DST_FOLDER/Binaries
cp Binaries/OpenNet_Tool            $DST_FOLDER/Binaries
cp Binaries/OpenNet.so              $DST_FOLDER/Binaries
cp Includes/OpenNet/*.h             $DST_FOLDER/Includes/OpenNet
cp Includes/OpenNetK/*.h            $DST_FOLDER/Includes/OpenNetK
cp Libraries/ONK_Lib.a              $DST_FOLDER/Libraries
cp Libraries/TestLib.a              $DST_FOLDER/Libraries
cp ONK_Lib/_DocUser/ReadMe.txt      $DST_FOLDER/ONK_Lib
cp ONK_Pro1000/_DocUser/ReadMe.txt  $DST_FOLDER/ONK_Intel
cp ONK_Pro1000/ONK_Pro1000.ko       $DST_FOLDER/Modules
cp OpenNet/_DocUser/ReadMe.txt      $DST_FOLDER/OpenNet
cp OpenNet_Tool/_DocUser/ReadMe.txt $DST_FOLDER/OpenNet_Tool
cp Packages/*.deb                   $DST_FOLDER/Packages
cp Scripts/OpenNet_Tool/*.txt       $DST_FOLDER/Scripts/OpenNet_Tool
cp DoxyFile_*.txt                   $DST_FOLDER

# ===== End =================================================================

echo OK
