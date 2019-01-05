
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

#define NAME "ONL_Pro1000"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    void           * mDevice;
    struct pci_dev * mPciDev;
}
DeviceInfo;

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static void __exit Exit  ( void );
static int  __init Init  ( void );
static int         Probe ( struct pci_dev * aDev, const struct pci_device_id * aId );
static void        Remove( struct pci_dev * aDev );

// Static variables
/////////////////////////////////////////////////////////////////////////////

static dev_t sDevice;

static unsigned int sDeviceCount = 0;

static DeviceInfo sDevices[ DEVICE_COUNT_MAX ];

static struct pci_driver sPciDriver =
{
    .name     = NAME    ,
    .id_table = ID_TABLE,
    .probe    = Probe   ,
    .remove   = Remove  ,
};

// Functions
/////////////////////////////////////////////////////////////////////////////

// aMinor
//
// Return  The instance address
void * Driver_FindDevice( unsigned char aMinor )
{
    printk( KERN_INFO "%s( %u )\n", __FUNCTION__, aMinor );

    return sDevices[ aMinor - MINOR( sDevice ) ].mDevice;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================

void Exit()
{
    printk( KERN_INFO "%s()\n", __FUNCTION__ );

    // pci_register_driver ==> pci_unregister_driver  See Init
    pci_unregister_driver( & sPciDriver );

    // alloc_chrdev_region ==> unregister_chrdev_region  See Init
    unregister_chrdev_region( sDevice, DEVICE_COUNT_MAX );
}

int Init()
{
    int lRet;

    printk( KERN_INFO "%s()\n", __FUNCTION__ );

    // alloc_chrdev_region ==> unregister_chrdev_region  See Exit
    lRet = alloc_chrdev_region( & sDevice, 0, DEVICE_COUNT_MAX, NAME );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - alloc_chrdev_region( , , ,  ) failed - %d\n", __FUNCTION__, lRet );
        return lRet;
    }

    // pci_register_driver ==> pci_unregister_driver  See Exit
    lRet = pci_register_driver( & sPciDriver );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - pci_register_driver(  ) failed - %d\n", __FUNCTION__, lRet );

        unregister_chrdev_region( sDevice, DEVICE_COUNT_MAX );

        return lRet;
    }

    return 0;
}

module_init( Init );
module_exit( Exit );

int Probe( struct pci_dev * aDev, const struct pci_device_id * aId )
{
    printk( KERN_INFO "%s( ,  )\n", __FUNCTION__ );

    sDevices[ sDeviceCount ].mDevice = Device_Create( aDev, MAJOR( sDevice ), MINOR( sDevice ) + sDeviceCount );
    sDevices[ sDeviceCount ].mPciDev =                aDev;

    if ( NULL == sDevices[ sDeviceCount ].mDevice )
    {
        return ( - __LINE__ );
    }

    sDeviceCount ++;

    return 0;
}

static void Remove( struct pci_dev * aDev )
{
    unsigned int i;

    printk( KERN_INFO "%s(  )\n", __FUNCTION__ );

    for ( i = 0; i < sDeviceCount; i ++ )
    {
        if ( aDev == sDevices[ i ].mPciDev )
        {
            Device_Delete( sDevices[ i ].mDevice );

            sDevices[ i ].mDevice = NULL;
            sDevices[ i ].mPciDev = NULL;

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
