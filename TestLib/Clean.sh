#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       TestLib/Clean.sh
# Usage      ./Clean.sh

echo Excuting  TestLib/Clean.sh  ...

# ===== Execution ===========================================================

rm *.o 2> /dev/null

rm ../Libraries/TestLib.a 2> /dev/null

# ===== End =================================================================

echo OK
