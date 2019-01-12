
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
#include <linux/uaccess.h>

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
    struct class   * mClass ;
    struct device  * mDevice;
    struct pci_dev * mPciDev;

    void        * mCommon_Virtual  ;
    dma_addr_t    mCommon_Physical ;
    unsigned int  mCommon_Size_byte;

    int mIrq        ;
    int mVectorCount;

    dev_t mDevId;

    unsigned char mReserved0[ 2 ];

    struct tasklet_struct mTasklet;

    unsigned char mDeviceCpp[ 1 ];
}
DeviceContext;

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static int  Copy_FromUser( void * aOut, const void * aIn, unsigned int aMax_byte, unsigned int aMin_byte, unsigned int * aInfo_byte );
static int  Copy_ToUser  ( void * aOut, const void * aIn, unsigned int aSize_byte );

static int  Device_Init  ( DeviceContext * aThis, unsigned int aIndex );
static void Device_Uninit( DeviceContext * aThis );

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
// aIndex
// aClass  [-K-;RW-]
//
// Return
//  NULL   Error
//  Other  The address of the created instance
//
// Device_Create ==> Device_Delete
void * Device_Create( struct pci_dev * aPciDev, unsigned char aMajor, unsigned char aMinor, unsigned int aIndex, struct class * aClass )
{
    DeviceContext * lResult;

    printk( KERN_DEBUG "%s( , %u, %u )\n", __FUNCTION__, aMajor, aMinor );

    // kmalloc ==> kfree  See Device_Delete
    lResult = kmalloc( sizeof( DeviceContext ) + DeviceCpp_GetContextSize(), GFP_KERNEL );
    if ( NULL == lResult )
    {
        printk( KERN_ERR "%s - kmalloc( ,  )\n", __FUNCTION__ );
    }
    else
    {
        lResult->mClass  = aClass ;
        lResult->mDevId  = MKDEV( aMajor, aMinor );
        lResult->mPciDev = aPciDev;

        tasklet_init( & lResult->mTasklet, Tasklet, (long unsigned int)( lResult ) ); // reinterpret_cast

        // DeviceCpp_Init ==> DeviceCpp_Uninit  See Device_Delete
        DeviceCpp_Init( lResult->mDeviceCpp );

        // pci_enable_device ==> pci_disable_device  See Device_Delete
        pci_enable_device( aPciDev );

        // IoMem_Init ==> IoMem_Uninit  See Device_Delete
        if ( 0 != IoMem_Init( lResult ) )  { goto Error0; }

        if ( 0 != Dma_Init( lResult ) ) { goto Error1; }

        // Interrupt_Init ==> Interrupt_Uninit  See Device_Delete
        if ( 0 != Interrupt_Init( lResult ) ) { goto Error1; }

        // Device_Init ==> Device_Uninit  See Device_Delete
        if ( 0 != Device_Init( lResult, aIndex ) ) { goto Error2; }
    }

    return lResult;

Error2:
    Interrupt_Uninit( lResult );

Error1:
    IoMem_Uninit( lResult );

Error0:
    pci_disable_device( aPciDev );
    DeviceCpp_Uninit( lResult->mDeviceCpp );
    kfree( lResult );
    return NULL;
}

// aThis [D--;RW-] The instance to delete
//
// Device_Create ==> Device_Delete
void Device_Delete( void * aThis )
{
    DeviceContext * lThis = aThis;

    printk( KERN_INFO "%s(  )\n", __FUNCTION__ );

    // Device_Init ==> Device_Uninit  See Device_Create
    Device_Uninit( lThis );

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

// aOut       [---;-W-]
// aIn        [---;R--]
// aMax_byte            The maximum size to copy
// aMin_byte            The minimum size to copy
// aInfo_byte [---;RW-] The function puts the copied size there.
//
// Return
//    0  OK
//  < 0  Error
int Copy_FromUser( void * aOut, const void * aIn, unsigned int aMax_byte, unsigned int aMin_byte, unsigned int * aInfo_byte )
{
    if ( 0 < aMax_byte )
    {
        int lSize_byte = copy_from_user( aOut, aIn, aMax_byte );
        if ( ( aMax_byte - aMin_byte ) < lSize_byte )
        {
            printk( KERN_ERR "%s - copy_from_user( , , %u bytes ) failed - %u\n", __FUNCTION__, aMax_byte, lSize_byte );
            return ( - __LINE__ );
        }

        ( * aInfo_byte ) = aMax_byte - lSize_byte;
    }

    return 0;
}

// aOut [---;-W-]
// aIn  [---;R--]
// aSize_byte     The size to copy
//
// Return
//    0  OK
//  < 0  Error
int Copy_ToUser( void * aOut, const void * aIn, unsigned int aSize_byte )
{
    if ( 0 < aSize_byte )
    {
        int lSize_byte = copy_to_user( aOut, aIn, aSize_byte );
        if ( 0 != lSize_byte )
        {
            printk( KERN_ERR "%s - copy_to_user( %p, %p, %u bytes ) failed - %d\n", __FUNCTION__, aOut, aIn, aSize_byte, lSize_byte );
            return ( - __LINE__ );
        }
    }

    return 0;
}

// aThis  [---;RW-]
// aIndex
//
// Return
//    0  OK
//  < 0  Error
//
// Device_Init ==> Device_Uninit
int Device_Init( DeviceContext * aThis, unsigned int aIndex )
{
    int             lRet   ;

    printk( KERN_INFO "%s( , %u )\n", __FUNCTION__, aIndex );

    // cdev_alloc ==> cdev_del  See Device_Uninit
    aThis->mCDev = cdev_alloc();
    if ( NULL == aThis->mCDev )
    {
        printk( KERN_ERR "%s - cdev_alloc() failed\n", __FUNCTION__ );
        return ( - __LINE__ );
    }

    aThis->mCDev->owner = THIS_MODULE;

    cdev_init( aThis->mCDev, & sOperations );

    lRet = cdev_add( aThis->mCDev, aThis->mDevId, 1 );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - cdev_add( , ,  ) failed - %d\n", __FUNCTION__, lRet );
        goto Error0;
    }

    // device_create ==> device_destroy  See Device_Uninit
    aThis->mDevice = device_create( aThis->mClass, NULL, aThis->mDevId, NULL, "ONK_Pro1000_%u", aIndex );
    if ( NULL == aThis->mDevice )
    {
        printk( KERN_ERR "%s - device_create( , , , , , %u ) failed\n", __FUNCTION__, aIndex );
        goto Error0;
    }

    aThis->mCommon_Size_byte = DeviceCpp_CommonBuffer_GetSize( aThis->mDeviceCpp );
    if ( 0 < aThis->mCommon_Size_byte )
    {
        // pci_alloc_consistent ==> pci_free_consistent  See Device_Uninit
        aThis->mCommon_Virtual = pci_alloc_consistent( aThis->mPciDev, aThis->mCommon_Size_byte, & aThis->mCommon_Physical );
        if ( NULL == aThis->mCommon_Virtual )
        {
            printk( KERN_ERR "%s - pci_alloc_consistent( , %u bytes,  ) failed\n", __FUNCTION__, aThis->mCommon_Size_byte );
            device_destroy( aThis->mClass, aThis->mDevId );
            goto Error0;
        }

        DeviceCpp_CommonBuffer_Set( aThis->mDeviceCpp, aThis->mCommon_Physical, aThis->mCommon_Virtual );
    }

    return 0;

Error0:
    cdev_del( aThis->mCDev );
    return ( - __LINE__ );
}

// aThis [---;RW-]
//
// Device_Init ==> Device_Uninit
void Device_Uninit( DeviceContext * aThis )
{
    printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    // pci_alloc_consistent ==> pci_free_consistent  See Device_Init
    pci_free_consistent( aThis->mPciDev, aThis->mCommon_Size_byte, aThis->mCommon_Virtual, aThis->mCommon_Physical );

    // device_create ==> device_destroy  See Device_Uninit
    device_destroy( aThis->mClass, aThis->mDevId );

    // cdev_alloc ==> cdev_del  See Device_Init
    cdev_del( aThis->mCDev );
}

// aThis [---;RW-]
//
// Return
//      0  OK
//  Other  Error
int Dma_Init( DeviceContext * aThis )
{
    int lRet;

    printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    lRet = pci_set_dma_mask( aThis->mPciDev, DMA_BIT_MASK( 64 ) );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - pci_set_dma_mask( ,  ) failed - %d\n", __FUNCTION__, lRet );
        return __LINE__;
    }

    lRet = pci_set_consistent_dma_mask( aThis->mPciDev, DMA_BIT_MASK( 64 ) );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - pci_set_consistent_dma_mask( ,  ) failed - %d\n", __FUNCTION__, lRet );
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
    unsigned int    lInSizeMax_byte;
    unsigned int    lInSizeMin_byte;
    unsigned int    lOutSize_byte  ;
    int             lResult        ;
    DeviceContext * lThis          ;

    printk( KERN_DEBUG "%s( %p, 0x%08x, 0x%lx )\n", __FUNCTION__, aFile, aCode, aArg );

    lThis = (DeviceContext *)( aFile->private_data ); // reinterpret_cast

    lResult = DeviceCpp_IoCtl_GetInfo( aCode, & lInSizeMax_byte, & lInSizeMin_byte, & lOutSize_byte );
    if ( 0 == lResult )
    {
        int lRet;

        unsigned int lSize_byte = ( lInSizeMax_byte > lOutSize_byte ) ? lInSizeMax_byte : lOutSize_byte;
        if ( 0 == lSize_byte )
        {
            lRet = DeviceCpp_IoCtl( & lThis->mDeviceCpp, aCode, NULL, 0 );
            if ( 0 > lRet )
            {
                printk( KERN_ERR "%s - DeviceCpp_IoCtl( , 0x%08x,  ) failed - %d\n", __FUNCTION__, aCode, lRet );
                lResult = ( - __LINE__ );
            }
        }
        else
        {
            void * lArg   = (void *)( aArg ); // reinterpret_cast
            void * lInOut = kmalloc( lSize_byte, GFP_KERNEL );
            if ( NULL == lInOut )
            {
                printk( KERN_ERR "%s - kmalloc( %u bytes ) failed\n", __FUNCTION__, lSize_byte );
                lResult = ( - __LINE__ );
            }
            else
            {
                unsigned int lInSize_byte = 0; // Avoid the warning

                lResult = Copy_FromUser( lInOut, lArg, lInSizeMax_byte, lInSizeMin_byte, & lInSize_byte );
                if ( 0 == lResult )
                {
                    lRet = DeviceCpp_IoCtl( & lThis->mDeviceCpp, aCode, lInOut, lInSize_byte );
                    if ( 0 > lRet )
                    {
                        printk( KERN_ERR "%s - DeviceCpp_IoCtl( , 0x%08x,  ) failed - %d\n", __FUNCTION__, aCode, lRet );
                        lResult = ( - __LINE__ );
                    }
                    else if ( 0 < lRet )
                    {
                        lResult = Copy_ToUser( lArg, lInOut, lOutSize_byte );
                    }
                }

                kfree( lInOut );
            }
        }
    }

    return lResult;
}

int Open( struct inode * aINode, struct file * aFile )
{
    printk( KERN_DEBUG "%s( ,  )\n", __FUNCTION__ );

    aFile->private_data = Driver_FindDevice( iminor( aINode ) );

    return 0;
}

int Release( struct inode * aINode, struct file * aFile )
{
    printk( KERN_DEBUG "%s( ,  )\n", __FUNCTION__ );

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
