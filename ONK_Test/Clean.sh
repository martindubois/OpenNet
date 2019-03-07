#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       ONK_Test/Clean.sh
# Usage      ./Clean.sh

echo Excuting  ONK_Test/Clean.sh  ...

# ===== Execution ===========================================================

rm *.o 2> /dev/null

rm ../Binaries/ONK_Test 2> /dev/null

# ===== End =================================================================

echo OK
