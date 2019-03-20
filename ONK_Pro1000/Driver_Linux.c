
// Author     KMS - Martin Dubois, ing.
// Copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Driver_Linux.c

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Linux ==============================================================
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/pci.h>

// ===== Includes ===========================================================
#include <OpenNetK/Constants.h>
#include <OpenNetK/Linux.h>

// ===== ONK_Pro1000 ========================================================
#include "Device_Linux.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define DEVICE_COUNT_MAX ( 32 )

static struct pci_device_id ID_TABLE[] =
{
    { PCI_DEVICE( 0x8086, 0x10c9 ), },
    { PCI_DEVICE( 0x8086, 0x10fb ), },

    { 0, }
};

MODULE_DEVICE_TABLE( pci, ID_TABLE );

#define MODULE_NAME "ONK_Pro1000"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    void           * mDevice;
    struct pci_dev * mPciDev;
}
DeviceInfo;

typedef struct
{
    dev_t        mChrDev     ;
    unsigned int mDeviceCount;
    DeviceInfo   mDevices[ DEVICE_COUNT_MAX ];
}
DriverContext;

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static char      * DevNode( struct device * aDev, umode_t * aMode );
static void __exit Exit  ( void );
static int  __init Init  ( void );
static int         Probe ( struct pci_dev * aDev, const struct pci_device_id * aId );
static void        Remove( struct pci_dev * aDev );

// Global variables
/////////////////////////////////////////////////////////////////////////////

struct class * gOpenNet_Class       = NULL;
unsigned int   gOpenNet_DeviceCount =    0;

EXPORT_SYMBOL( gOpenNet_Class       );
EXPORT_SYMBOL( gOpenNet_DeviceCount );

// Static variables
/////////////////////////////////////////////////////////////////////////////

static struct pci_driver sPciDriver =
{
    .name     = MODULE_NAME,
    .id_table = ID_TABLE   ,
    .probe    = Probe      ,
    .remove   = Remove     ,
};

static DriverContext sThis;

// Functions
/////////////////////////////////////////////////////////////////////////////

// aMinor
//
// Return  The instance address
void * Driver_FindDevice( unsigned char aMinor )
{
    // printk( KERN_DEBUG "%s( %u )\n", __FUNCTION__, aMinor );

    return sThis.mDevices[ aMinor - MINOR( sThis.mChrDev ) ].mDevice;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================

char * DevNode( struct device * aDev, umode_t * aMode )
{
    // printk( KERN_DEBUG "%s( ,  )\n", __FUNCTION__ );

    if ( NULL != aMode )
    {
        ( * aMode ) = 0666;
    }

    return NULL;
}

void Exit()
{
    // printk( KERN_DEBUG "%s()\n", __FUNCTION__ );

    ASSERT( NULL != gOpenNet_Class );

    // pci_register_driver ==> pci_unregister_driver  See Init
    pci_unregister_driver( & sPciDriver );

    // class_create ==> class_destroy  See Init
    class_destroy( gOpenNet_Class );

    // alloc_chrdev_region ==> unregister_chrdev_region  See Init
    unregister_chrdev_region( sThis.mChrDev, DEVICE_COUNT_MAX );
}

module_exit( Exit );

int Init()
{
    int lRet;

    // printk( KERN_DEBUG "%s()\n", __FUNCTION__ );

    ASSERT( NULL == gOpenNet_Class );

    sThis.mDeviceCount = 0;

    // class_create ==> class_destroy  See Exit
    gOpenNet_Class = class_create( THIS_MODULE, "OpenNet" );
    if ( NULL == gOpenNet_Class )
    {
        printk( KERN_ERR "%s - class_create( ,  ) failed", __FUNCTION__ );
        return ( - __LINE__ );
    }

    gOpenNet_Class->devnode = DevNode;

    // alloc_chrdev_region ==> unregister_chrdev_region  See Exit
    lRet = alloc_chrdev_region( & sThis.mChrDev, 0, DEVICE_COUNT_MAX, MODULE_NAME );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - alloc_chrdev_region( , , ,  ) failed - %d\n", __FUNCTION__, lRet );
        goto Error0;
    }

    // pci_register_driver ==> pci_unregister_driver  See Exit
    lRet = pci_register_driver( & sPciDriver );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - pci_register_driver(  ) failed - %d\n", __FUNCTION__, lRet );

        unregister_chrdev_region( sThis.mChrDev, DEVICE_COUNT_MAX );
        
        goto Error0;
    }

    return 0;

Error0:
    class_destroy( gOpenNet_Class );
    return ( - __LINE__ );
}

module_init( Init );

int Probe( struct pci_dev * aDev, const struct pci_device_id * aId )
{
    // printk( KERN_DEBUG "%s( ,  )\n", __FUNCTION__ );

    ASSERT( NULL != aDev );
    ASSERT( NULL != aId  );

    ASSERT( NULL != gOpenNet_Class );
    
    ASSERT( DEVICE_COUNT_MAX > sThis->mDeviceCount );

    sThis.mDevices[ sThis.mDeviceCount ].mDevice = Device_Create( aDev, MAJOR( sThis.mChrDev ), MINOR( sThis.mChrDev ) + sThis.mDeviceCount, gOpenNet_DeviceCount, gOpenNet_Class );
    sThis.mDevices[ sThis.mDeviceCount ].mPciDev =                aDev;

    if ( NULL == sThis.mDevices[ sThis.mDeviceCount ].mDevice )
    {
        printk( KERN_ERR "%s - Device_Create( , , , ,  ) failed\n", __FUNCTION__ );
        return ( - __LINE__ );
    }

    gOpenNet_DeviceCount ++;
    sThis  .mDeviceCount ++;

    return 0;
}

static void Remove( struct pci_dev * aDev )
{
    unsigned int i;

    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    for ( i = 0; i < sThis.mDeviceCount; i ++ )
    {
        if ( aDev == sThis.mDevices[ i ].mPciDev )
        {
            Device_Delete( sThis.mDevices[ i ].mDevice );

            sThis.mDevices[ i ].mDevice = NULL;
            sThis.mDevices[ i ].mPciDev = NULL;

            return;
        }
    }

    printk( KERN_ERR "%s - Invalid pci_dev\n", __FUNCTION__ );
}

// License
/////////////////////////////////////////////////////////////////////////////

MODULE_LICENSE( "GPL" );

MODULE_AUTHOR( "KMS - Martin Dubois <mdubois@kms-quebec.com>" );
MODULE_DESCRIPTION( "ONK_Pro1000" );

MODULE_SUPPORTED_DEVICE( "Intel 82576" );
