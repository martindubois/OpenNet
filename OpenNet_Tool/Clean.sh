#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet_Tool/Clean.sh
# Usage      ./Clean.sh

echo Excuting  OpenNet_Tool/Clean.sh  ...

# ===== Execution ===========================================================

rm *.o 2> /dev/null

rm ../Binaries/OpenNet_Tool 2> /dev/null

# ===== End =================================================================

echo OK
