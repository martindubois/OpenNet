
// Author     KMS - Martin Dubois, ing.
// Copyright  Copyright (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/Driver_Linux.c

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Linux ==============================================================
#include <linux/device.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>

// ===== Includes ===========================================================
#include <OpenNetK/Constants.h>
#include <OpenNetK/Linux.h>

// ===== ONK_Tunnel_IO ======================================================
#include "Device_Linux.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define MODULE_NAME "ONK_Tunnel_IO"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    dev_t  mChrDev;
    void * mDevice;
}
DriverContext;

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static void __exit Exit( void );
static int  __init Init( void );

// Static variables
/////////////////////////////////////////////////////////////////////////////

static DriverContext sThis;

// Functions
/////////////////////////////////////////////////////////////////////////////

// aMinor
//
// Return  The instance address
void * Driver_FindDevice( unsigned char aMinor )
{
    // printk( KERN_DEBUG "%s( %u )\n", __FUNCTION__, aMinor );

    return sThis.mDevice;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================

void Exit()
{
    // printk( KERN_DEBUG "%s()\n", __FUNCTION__ );

    if ( NULL != sThis.mDevice )
    {
        Device_Delete( sThis.mDevice );
    }

    // alloc_chrdev_region ==> unregister_chrdev_region  See Init
    unregister_chrdev_region( sThis.mChrDev, 1 );
}

module_exit( Exit );

int Init()
{
    int lRet;

    // printk( KERN_DEBUG "%s()\n", __FUNCTION__ );

    ASSERT( NULL != gOpenNet_Class );

    // alloc_chrdev_region ==> unregister_chrdev_region  See Exit
    lRet = alloc_chrdev_region( & sThis.mChrDev, 0, 1, MODULE_NAME );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - alloc_chrdev_region( , , ,  ) failed - %d\n", __FUNCTION__, lRet );

        return ( - __LINE__ );
    }

    sThis.mDevice = Device_Create( MAJOR( sThis.mChrDev ), MINOR( sThis.mChrDev ), gOpenNet_DeviceCount, gOpenNet_Class );
    if ( NULL == sThis.mDevice )
    {
        printk( KERN_ERR "%s - Device_Create( , , ,  ) failed\n", __FUNCTION__ );
    }
    else
    {
        gOpenNet_DeviceCount ++;
    }

    return 0;
}

module_init( Init );

// License
/////////////////////////////////////////////////////////////////////////////

MODULE_LICENSE( "GPL" );

MODULE_AUTHOR( "KMS - Martin Dubois <mdubois@kms-quebec.com>" );
MODULE_DESCRIPTION( "ONK_Tunnel_IO" );
