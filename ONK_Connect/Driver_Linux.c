
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Connect/Driver_Linux.c

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Linux ==============================================================

// ===== ONK_Connect ========================================================
#include "Device_Linux.h"

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static void __exit Exit(void);
static int  __init Init(void);

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================

void Exit()
{

}

module_exit(Exit);

int Init()
{
    Device_Create();
}

module_init(Init);

// License
/////////////////////////////////////////////////////////////////////////////

MODULE_LICENSE("GPL");

MODULE_AUTHOR("KMS - Martin Dubois <mdubois@kms-quebec.com>");
MODULE_DESCRIPTION("ONK_Connect");
