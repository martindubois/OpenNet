#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       ONK_Lib/Clean.sh
# Usage      ./Clean.sh

echo Excuting  ONK_Lib/Clean.sh  ...

# ===== Execution ===========================================================

rm *.o 2> /dev/null

rm ../Libraries/ONK_Lib.a 2> /dev/null

# ===== End =================================================================

echo OK
