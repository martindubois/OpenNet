#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       CreatePackages.sh
# Usage      ./CreatePackages.sh

echo Excuting  CreatePackages.sh  ...

# ===== Version =============================================================

KERNEL=$(uname -r)

PACKAGE_VERSION=1.0-10

VERSION=1.0

# ===== Initialisation ======================================================

PACKAGE_NAME=kms-opennet_$PACKAGE_VERSION
PACKAGE_NAME_DDK=kms-opennet-ddk_$PACKAGE_VERSION

# ===== Execution ===========================================================

echo Creating  kms-opennet  ...

mkdir Packages
mkdir Packages/$PACKAGE_NAME
mkdir Packages/$PACKAGE_NAME/lib
mkdir Packages/$PACKAGE_NAME/lib/modules
mkdir Packages/$PACKAGE_NAME/lib/modules/$KERNEL
mkdir Packages/$PACKAGE_NAME/lib/modules/$KERNEL/kernel
mkdir Packages/$PACKAGE_NAME/lib/modules/$KERNEL/kernel/drivers
mkdir Packages/$PACKAGE_NAME/lib/modules/$KERNEL/kernel/drivers/pci
mkdir Packages/$PACKAGE_NAME/usr
mkdir Packages/$PACKAGE_NAME/usr/local
mkdir Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION
mkdir Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/bin
mkdir Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc
mkdir Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNet
mkdir Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNetK
mkdir Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/ONK_Hardware
mkdir Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/OpenNet
mkdir Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/OpenNet_Tool
mkdir Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/OpenNet_Setup

cp _DocUser/ReadMe.txt                       Packages/$PACKAGE_NAME
cp Binaries/libOpenNet.so                    Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/bin
cp Binaries/OpenNet_Setup                    Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/bin
cp Binaries/OpenNet_Tool                     Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/bin
cp DoxyFile_*.txt                            Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION
cp Includes/OpenNet/*.h                      Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNet
cp Includes/OpenNetK/Adapter_Types.h         Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/ARP.h                   Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Ethernet.h              Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/IPv4.h                  Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Kernel_CUDA.h           Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Kernel.h                Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/PacketGenerator_Types.h Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Types.h                 Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp ONK_Pro1000/_DocUser/ReadMe.txt           Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/ONK_Hardware
cp ONK_Pro1000/ONK_Pro1000.ko                Packages/$PACKAGE_NAME/lib/modules/$KERNEL/kernel/drivers/pci
cp OpenNet/_DocUser/ReadMe.txt               Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/OpenNet
cp OpenNet_Setup/_DocUser/ReadMe.txt         Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/OpenNet_Setup
cp OpenNet_Tool/_DocUser/ReadMe.txt          Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/OpenNet_Tool
cp Scripts/OpenNet_Setup.sh                  Packages/$PACKAGE_NAME/usr/local/OpenNet_$VERSION/bin

mkdir Packages/$PACKAGE_NAME/DEBIAN

cp Scripts/control     Packages/$PACKAGE_NAME/DEBIAN/control
cp Scripts/postinst.sh Packages/$PACKAGE_NAME/DEBIAN/postinst
cp Scripts/prerm.sh    Packages/$PACKAGE_NAME/DEBIAN/prerm

dpkg-deb --build Packages/$PACKAGE_NAME

rm -r Packages/$PACKAGE_NAME

echo Creating  kms-opennet-ddk  ...

mkdir Packages/$PACKAGE_NAME_DDK
mkdir Packages/$PACKAGE_NAME_DDK/usr
mkdir Packages/$PACKAGE_NAME_DDK/usr/local
mkdir Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION
mkdir Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc
mkdir Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
mkdir Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/lib
mkdir Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/ONK_lib

cp Includes/OpenNetK/Adapter.h               Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Adapter_Linux.h         Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Constants.h             Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Hardware.h              Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Hardware_Linux.h        Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Hardware_Statistics.h   Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/IoCtl.h                 Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Linux.h                 Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/LinuxCpp.h              Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/OS.h                    Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/OSDep.h                 Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Packet.h                Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/SpinLock.h              Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Includes/OpenNetK/Types.h                 Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/inc/OpenNetK
cp Libraries/ONK_Lib.a                       Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/lib
cp ONK_Lib/_DocUser/ReadMe.txt               Packages/$PACKAGE_NAME_DDK/usr/local/OpenNet_$VERSION/ONK_Lib

mkdir Packages/$PACKAGE_NAME_DDK/DEBIAN

cp Scripts/control_DDK Packages/$PACKAGE_NAME_DDK/DEBIAN/control

dpkg-deb --build Packages/$PACKAGE_NAME_DDK

rm -r Packages/$PACKAGE_NAME_DDK

# ===== End =================================================================

echo OK
