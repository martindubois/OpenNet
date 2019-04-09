#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet_Setup/Clean.sh
# Usage      ./Clean.sh

echo Excuting  OpenNet_Setup/Clean.sh  ...

# ===== Execution ===========================================================

rm *.o 2> /dev/null

rm ../Binaries/OpenNet_Setup 2> /dev/null

# ===== End =================================================================

echo OK
