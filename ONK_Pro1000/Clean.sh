#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       ONK_Pro1000/Clean.sh
# Usage      ./Clean.sh

echo Excuting  ONK_Pro1000/Clean.sh  ...

# ===== Execution ===========================================================

rm *.o 2> /dev/null

rm ONK_Pro1000.ko 2> /dev/null

# ===== End =================================================================

echo OK
