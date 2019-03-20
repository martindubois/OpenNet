#!/bin/sh

# Auhtor     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Scripts/postinst_RT.sh

echo  Executing  postinst  ...

# ===== Execution ===========================================================

cp -f /etc/modules /etc/modules.bak
if [ 0 != $? ] ; then
    echo ERROR  cp -f /etc/modules /etc/modules.bak  failed
    exit 1
fi

echo ONK_Pro1000 >> /etc/modules
if [ 0 != $? ] ; then
    echo ERROR  Cannot edit /etc/modules
    exit 2
fi

echo ONK_Tunnel_IO >> /etc/modules
if [ 0 != $? ] ; then
    echo ERROR  Cannot edit /etc/modules
    exit 3
fi

depmod
if [ 0 != $? ] ; then
    echo ERROR  depmod  failed
    exit 4
fi

# ===== End =================================================================

echo OK
