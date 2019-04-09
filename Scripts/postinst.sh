#!/bin/sh

# Auhtor     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Scripts/postinst.sh

echo  Executing  postinst.sh  ...

# ===== Configuration =======================================================

OPEN_NET_INSTALL=/usr/local/OpenNet_0.0

# ===== Execution ===========================================================

depmod
if [ 0 != $? ] ; then
    echo ERROR  depmod  failed
    exit 1
fi

$OPEN_NET_INSTALL/bin/OpenNet_Setup.sh install
if [ 0 != $? ] ; then
    echo ERROR  $OPEN_NET_INSTALL/bin/OpenNet_Setup.sh install  failed
    exit 2
fi

# ===== End =================================================================

echo OK
