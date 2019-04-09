#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Clean.sh
# Usage      ./Clean.sh

echo Excuting  Clean.sh  ...

# ===== Execution ===========================================================

cd ONK_Lib
./Clean.sh
cd ..

cd ONK_Pro1000
./Clean.sh
cd ..

cd ONK_Test
./Clean.sh
cd ..

cd ONK_Tunnel_IO
./Clean.sh
cd ..

cd OpenNet
./Clean.sh
cd ..

cd OpenNet_Setup
./Clean.sh
cd ..

cd OpenNet_Test
./Clean.sh
cd ..

cd OpenNet_Tool
./Clean.sh
cd ..

cd TestLib
./Clean.sh
cd ..

# ===== End =================================================================

echo OK
exit 0
