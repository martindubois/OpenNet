
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

// ===== ONK_Pro1000 ========================================================
#include "Device_Linux.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define DEVICE_COUNT_MAX ( 32 )

static struct pci_device_id ID_TABLE[] =
{
    { PCI_DEVICE( 0x8086, 0x10c9 ), },

    { 0, }
};

MODULE_DEVICE_TABLE( pci, ID_TABLE );

#define NAME "ONK_Pro1000"

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
    struct class * mClass      ;
    dev_t          mDevice     ;
    unsigned int   mDeviceCount;
    DeviceInfo     mDevices[ DEVICE_COUNT_MAX ];
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

// Static variables
/////////////////////////////////////////////////////////////////////////////

static struct pci_driver sPciDriver =
{
    .name     = NAME    ,
    .id_table = ID_TABLE,
    .probe    = Probe   ,
    .remove   = Remove  ,
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

    return sThis.mDevices[ aMinor - MINOR( sThis.mDevice ) ].mDevice;
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

    // pci_register_driver ==> pci_unregister_driver  See Init
    pci_unregister_driver( & sPciDriver );

    // class_create ==> class_destroy  See Init
    class_destroy( sThis.mClass );

    // alloc_chrdev_region ==> unregister_chrdev_region  See Init
    unregister_chrdev_region( sThis.mDevice, DEVICE_COUNT_MAX );
}

int Init()
{
    int lRet;

    // printk( KERN_DEBUG "%s()\n", __FUNCTION__ );

    sThis.mDeviceCount = 0;

    // class_create ==> class_destroy  See Exit
    sThis.mClass = class_create( THIS_MODULE, NAME );
    if ( NULL == sThis.mClass )
    {
        printk( KERN_ERR "%s - class_create( ,  ) failed", __FUNCTION__ );
        return ( - __LINE__ );
    }

    sThis.mClass->devnode = DevNode;

    // alloc_chrdev_region ==> unregister_chrdev_region  See Exit
    lRet = alloc_chrdev_region( & sThis.mDevice, 0, DEVICE_COUNT_MAX, NAME );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - alloc_chrdev_region( , , ,  ) failed - %d\n", __FUNCTION__, lRet );

        class_destroy( sThis.mClass );

        return ( - __LINE__ );
    }

    // pci_register_driver ==> pci_unregister_driver  See Exit
    lRet = pci_register_driver( & sPciDriver );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - pci_register_driver(  ) failed - %d\n", __FUNCTION__, lRet );

        unregister_chrdev_region( sThis.mDevice, DEVICE_COUNT_MAX );
        
        class_destroy( sThis.mClass );

        return ( - __LINE__ );
    }

    return 0;
}

module_init( Init );
module_exit( Exit );

int Probe( struct pci_dev * aDev, const struct pci_device_id * aId )
{
    // printk( KERN_DEBUG "%s( ,  )\n", __FUNCTION__ );

    sThis.mDevices[ sThis.mDeviceCount ].mDevice = Device_Create( aDev, MAJOR( sThis.mDevice ), MINOR( sThis.mDevice ) + sThis.mDeviceCount, sThis.mDeviceCount, sThis.mClass );
    sThis.mDevices[ sThis.mDeviceCount ].mPciDev =                aDev;

    if ( NULL == sThis.mDevices[ sThis.mDeviceCount ].mDevice )
    {
        printk( KERN_ERR "%s - Device_Create( , , , ,  ) failed\n", __FUNCTION__ );
        return ( - __LINE__ );
    }

    sThis.mDeviceCount ++;

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
