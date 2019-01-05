
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Device_Linux.h

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Linux ==============================================================
#include <linux/cdev.h>
#include <linux/interrupt.h>
#include <linux/pci.h>

// ===== ONK_Pro1000 ========================================================
#include "DeviceCpp.h"
#include "Driver_Linux.h"

#include "Device_Linux.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define VECTOR_MAX (16)
#define VECTOR_MIN ( 1)

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    struct cdev    * mCDev  ;
    struct pci_dev * mPciDev;

    int mIrq        ;
    int mVectorCount;

    unsigned char mReserved0[ 4 ];

    struct tasklet_struct mTasklet;

    unsigned char mDeviceCpp[ 1 ];
}
DeviceContext;

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static int  Dma_Init( DeviceContext * aThis );

static int  Interrupt_Init  ( DeviceContext * aThis );
static void Interrupt_Uninit( DeviceContext * aThis );

static int  IoMem_Init  ( DeviceContext * aThis );
static void IoMem_Uninit( DeviceContext * aThis );

// ===== Entry points =======================================================

static long IoCtl  ( struct file * aFile, unsigned int, unsigned long );
static int  Open   ( struct inode * aINode, struct file * aFile );
static int  Release( struct inode * aINode, struct file * aFile );

static irqreturn_t ProcessIrq( int aIrq, void * aDevId );

static void Tasklet( unsigned long aData );

// Static variables
/////////////////////////////////////////////////////////////////////////////

static struct file_operations sOperations =
{
    .owner          = THIS_MODULE,
    .open           = Open       ,
    .release        = Release    ,
    .unlocked_ioctl = IoCtl      ,
};

// Functions
/////////////////////////////////////////////////////////////////////////////

// aPciDev [---;RW-]
// aMajor
// aMinor
//
// Return
//  NULL   Error
//  Other  The address of the created instance
void * Device_Create( struct pci_dev * aPciDev, unsigned char aMajor, unsigned char aMinor )
{
    DeviceContext * lResult;
    int             lRet   ;

    printk( KERN_DEBUG "%s( , %u, %u )\n", __FUNCTION__, aMajor, aMinor );

    // kmalloc ==> kfree  See Device_Delete
    lResult = kmalloc( sizeof( DeviceContext ) + DeviceCpp_GetContextSize(), GFP_KERNEL );

    lResult->mPciDev = aPciDev;

    tasklet_init( & lResult->mTasklet, Tasklet, (long unsigned int)( lResult ) ); // reinterpret_cast

    // DeviceCpp_Init ==> DeviceCpp_Uninit  See Device_Delete
    DeviceCpp_Init( lResult->mDeviceCpp );

    // cdev_alloc ==> cdev_del  See Device_Delete
    lResult->mCDev = cdev_alloc();

    cdev_init( lResult->mCDev, & sOperations );

    lResult->mCDev->owner = THIS_MODULE;

    lRet = cdev_add( lResult->mCDev, MKDEV( aMajor, aMinor ), 1 );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - cdev_add( , ,  ) failed - %d\n", __FUNCTION__, lRet );
        goto Error0;
    }

    // pci_enable_device ==> pci_disable_device  See Device_Delete
    pci_enable_device( aPciDev );

    // IoMem_Init ==> IoMem_Uninit  See Device_Delete
    if ( 0 != IoMem_Init( lResult ) )  { goto Error1; }

    if ( 0 != Dma_Init( lResult ) ) { goto Error2; }

    // Interrupt_Init ==> Interrupt_Uninit  See Device_Delete
    if ( 0 != Interrupt_Init( lResult ) ) { goto Error2; }

    return lResult;

Error2:
    IoMem_Uninit( lResult );

Error1:
    pci_disable_device( aPciDev );

Error0:
    DeviceCpp_Uninit( lResult->mDeviceCpp );
    cdev_del( lResult->mCDev );
    kfree( lResult );
    return NULL;
}

// aThis [D--;RW-] The instance to delete
void Device_Delete( void * aThis )
{
    DeviceContext * lThis = aThis;

    printk( KERN_INFO "%s(  )\n", __FUNCTION__ );

    // cdev_alloc ==> cdev_del  See Device_Create
    cdev_del( lThis->mCDev );

    // Interrupt_Init ==> Interrupt_Uninit  See Device_Create
    Interrupt_Uninit( lThis );

    // DeviceCpp_Init ==> DeviceCpp_Uninit  See Device_Create
    DeviceCpp_Uninit( lThis->mDeviceCpp );

    // IoMem_Init ==> IoMem_Uninit  See Device_Create
    IoMem_Uninit( lThis );

    // pci_enable_device ==> pci_disable_device  See Device_Delete
    pci_disable_device( lThis->mPciDev );

    // kmalloc ==> kfree  See Device_Create
    kfree( lThis );
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aThis [---;RW-]
//
// Return
//      0  OK
//  Other  Error
int Dma_Init( DeviceContext * aThis )
{
    int lRet;

    printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    lRet = pci_set_dma_mask( aThis->mPciDev, 0xffffffffffffffff );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - pci_set_dma_mask( ,  ) failed - %d\n", __FUNCTION__, lRet );
        return __LINE__;
    }

    pci_set_master( aThis->mPciDev );

    return 0;
}

// aThis [---;RW-]
//
// Return
//      0  OK
//  Other  Error
//
// Interrupt_Init ==> Interrupt_Uninit
int Interrupt_Init( DeviceContext * aThis )
{
    int lRet;

    printk( KERN_INFO "%s(  )\n", __FUNCTION__ );

    // pci_alloc_irq_vectors ==> pci_free_irq_vectors  See Interrupt_Uninit
    aThis->mVectorCount = pci_alloc_irq_vectors( aThis->mPciDev, VECTOR_MIN, VECTOR_MAX, PCI_IRQ_ALL_TYPES );
    if ( ( VECTOR_MIN > aThis->mVectorCount ) || ( VECTOR_MAX < aThis->mVectorCount ) )
    {
        printk( KERN_ERR "%s - pci_alloc_irq_vector( , , ,  ) failed - %d\n", __FUNCTION__, aThis->mVectorCount );
        return ( - __LINE__ );
    }

    aThis->mIrq = pci_irq_vector( aThis->mPciDev, 0 );

    // request_irq ==> free_irq  See Interrupt_Uninit
    lRet = request_irq( aThis->mIrq, ProcessIrq, 0, "ONK_Pro1000", aThis );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - request_irq( %d, , , ,  ) failed - %d\n", __FUNCTION__, aThis->mIrq, lRet );
        return ( - __LINE__ );
    }

    return 0;
}

// aThis [---;RW-]
//
// Interrupt_Init ==> Interrupt_Uninit
void Interrupt_Uninit( DeviceContext * aThis )
{
    printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    // request_irq ==> free_irq  See Interrupt_Init
    free_irq( aThis->mIrq, aThis );

    // pci_alloc_irq_vectors ==> pci_free_irq_vectors  See Interrupt_Init
    pci_free_irq_vectors( aThis->mPciDev );
}

// aThis [---;RW-]
//
// Return
//      0  OK
//  Other  Error
//
// IoMem_Init ==> IoMem_Uninit
int IoMem_Init( DeviceContext * aThis )
{
    unsigned int lMemCount = 0;

    unsigned int i;

    printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    for ( i = 0; i < 6; i ++ )
    {
        // pci_request_region ==> pci_release_regions  See IoMem_Uninit
        if ( 0 == pci_request_region( aThis->mPciDev, i, "ONK_Pro1000" ) )
        {
            if ( IORESOURCE_MEM == ( pci_resource_flags( aThis->mPciDev, i ) & IORESOURCE_MEM ) )
            {
                unsigned int lSize_byte = pci_resource_len( aThis->mPciDev, i );
                void       * lVirtual   = pci_iomap_range ( aThis->mPciDev, i, 0, lSize_byte );

                int lRet = DeviceCpp_SetMemory( aThis->mDeviceCpp, lMemCount, lVirtual, lSize_byte );
                if ( 0 != lRet )
                {
                    printk( KERN_ERR "%s - DeviceCpp_SetMemory( , %u, , %u bytes ) failed - %d", __FUNCTION__, lMemCount, lSize_byte, lRet );
                    return __LINE__;
                }

                lMemCount ++;
            }
        }
    }

    return 0;
}

// aThis [---;RW-]
//
// IoMem_Init ==> IoMem_Uninit
void IoMem_Uninit( DeviceContext * aThis )
{
    printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    // pci_request_region ==> pci_release_regions  See IoMem_Init
    pci_release_regions( aThis->mPciDev );
}

// ===== Entry points =======================================================

long IoCtl( struct file * aFile, unsigned int aCode, unsigned long aArg )
{
    printk( KERN_INFO "%s( , 0x%08x,  )\n", __FUNCTION__, aCode );

    return 0;
}

int Open( struct inode * aINode, struct file * aFile )
{
    printk( KERN_INFO "%s( ,  )\n", __FUNCTION__ );

    aFile->private_data = Driver_FindDevice( iminor( aINode ) );

    return 0;
}

int Release( struct inode * aINode, struct file * aFile )
{
    printk( KERN_INFO "%s( ,  )\n", __FUNCTION__ );

    return 0;
}

irqreturn_t ProcessIrq( int aIrq, void * aDevId )
{
    DeviceContext * lThis = aDevId;

    switch ( DeviceCpp_Interrupt_Process( lThis->mDeviceCpp, aIrq ) )
    {
    case PIR_IGNORED   : return IRQ_NONE;
    case PIR_PROCESSED : break;

    case PIR_TO_PROCESS :
        tasklet_schedule( & lThis->mTasklet );
        break;
    }

    return IRQ_HANDLED;
}

void Tasklet( unsigned long aData )
{
    DeviceContext * lThis = (DeviceContext *)( aData ); // reinterpret_cast

    DeviceCpp_Interrupt_Process2( & lThis->mDeviceCpp );
}
