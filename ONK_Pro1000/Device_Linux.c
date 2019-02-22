
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Device_Linux.cpp

#define _KMS_LINUX_

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Linux ==============================================================
#include <linux/aer.h>
#include <linux/cdev.h>
#include <linux/dma-mapping.h>
#include <linux/interrupt.h>
#include <linux/pci.h>
#include <linux/timer.h>
#include <linux/uaccess.h>

// ===== NVIDIA =============================================================
#include <nv-p2p.h>

// ===== Includes ===========================================================
#include <OpenNetK/Constants.h>
#include <OpenNetK/IoCtl.h>
#include <OpenNetK/Linux.h>
#include <OpenNetK/OSDep.h>

// ===== ONK_Pro1000 ========================================================
#include "DeviceCpp.h"
#include "Driver_Linux.h"
#include "NvBuffer.h"

#include "Device_Linux.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define BAR_QTY (6)

#define MODULE_NAME "ONK_Pro1000"

#define VECTOR_MAX (16)
#define VECTOR_MIN ( 1)

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    struct
    {
        unsigned mDeleting : 1;

        unsigned mReserved0[ 31 ];
    }
    mFlags;

    // ===== Pointer to linux structures ====================================
    struct cdev    * mCDev  ;
    struct class   * mClass ;
    struct device  * mDevice;
    struct pci_dev * mPciDev;

    // ===== Linux structures ===============================================
    spinlock_t mAdapterLock ;
    spinlock_t mHardwareLock;

    struct tasklet_struct mTasklet;
    struct timer_list     mTimer  ;
    struct work_struct    mWork   ;

    // ===== Common buffer information ======================================
    void        * mCommon_CA;
    dma_addr_t    mCommon_PA;
    unsigned int  mCommon_Size_byte;

    // ===== Interrupt information ==========================================
    int mIrq        ;
    int mVectorCount;

    // ====== Information about buffers =====================================
    NvBuffer mBuffers[ OPEN_NET_BUFFER_QTY ];

    // ===== Shared memory information ======================================
    void          * mShared          ;
    unsigned int    mShared_PageCount;
    struct page * * mShared_Pages    ;

    // ===== Other ==========================================================

    // The major and minor number associated to this device.
    dev_t mDevId;

    // The kernel address used to communication with the device are kept here
    // and used to undo mapping.
    void * mIoMems_MA[ BAR_QTY ];

    // This structure contains pointer to OS dependent functions used by the
    // ONK_Lib.
    OpenNetK_OSDep mOSDep;

    // When allocating the memory for this tructure, the code also allocate
    // the space needed for the DeviceCpp context. The mDeviceCpp member make
    // easy to access this context.
    unsigned char mDeviceCpp[ 1 ];
}
DeviceContext;

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static int  CommonBuffer_Allocate( DeviceContext * aThis );

static int  Copy_FromUser( void * aOut_UA, const void * aIn   , unsigned int aMax_byte, unsigned int aMin_byte, unsigned int * aInfo_byte );
static int  Copy_ToUser  ( void * aOut   , const void * aIn_UA, unsigned int aSize_byte );

static int  Device_Init  ( DeviceContext * aThis, unsigned int aIndex );
static void Device_Uninit( DeviceContext * aThis );

static int  Dma_Init  ( DeviceContext * aThis );
static void Dma_Uninit( DeviceContext * aThis );

static int  Interrupt_Init  ( DeviceContext * aThis );
static void Interrupt_Uninit( DeviceContext * aThis );

static int  IoCtl_ProcessResult( DeviceContext * aThis, int aIoCtlResult );
static int  IoCtl_WithArgument ( DeviceContext * aThis, struct file * aFile, unsigned int aCode, unsigned long aArg_UA, const OpenNetK_IoCtl_Info * aInfo, unsigned int aSize_byte );

static int  IoMem_Init  ( DeviceContext * aThis );
static void IoMem_Uninit( DeviceContext * aThis );

static void OSDep_Init( DeviceContext * aThis );

static void Timer_Start( DeviceContext * aThis );

// ===== Entry points =======================================================

static long IoCtl  ( struct file * aFile, unsigned int aCode, unsigned long aArg_UA );
static int  Open   ( struct inode * aINode, struct file * aFile );
static int  Release( struct inode * aINode, struct file * aFile );

static irqreturn_t ProcessIrq( int aIrq, void * aDevId );

static void Tasklet( unsigned long aData );

static void Timer( struct timer_list  * aTimer );
static void Work ( struct work_struct * aWork  );

// ===== OSDep ==============================================================

static void * AllocateMemory( unsigned int aSize_byte );
static void   FreeMemory    ( void * aMemory );

static uint64_t GetTimeStamp( void );

static void * MapBuffer  ( void * aContext, uint64_t * aBuffer_PA, uint64_t aBuffer_DA, unsigned int aSize_byte, uint64_t aMarker_PA, volatile void * * aMarker_MA );
static void   UnmapBuffer( void * aContext, void * aBuffer, unsigned int aSize_byte, volatile void * aMarker_MA );

static void * MapSharedMemory  ( void * aContext, void * aShared_UA, unsigned int aSize_byte );
static void   UnmapSharedMemory( void * aContext );

static void   LockSpinlock  ( void * aLock );
static void   UnlockSpinlock( void * aLock );

// ===== NVIDIA callback ====================================================

static void FreeCallback( void * aContext_XA );

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

// aPciDev [---;RW-] The pci device structure
// aMajor            The major number associated to this device
// aMinor            The minor number associated to this device
// aIndex            The index of this device
// aClass  [-K-;RW-] The device class structure
//
// Return
//  NULL   Error
//  Other  The address of the created instance
//
// Device_Create ==> Device_Delete
//
// Level   Thread
// Thread  Probe
void * Device_Create( struct pci_dev * aPciDev, unsigned char aMajor, unsigned char aMinor, unsigned int aIndex, struct class * aClass )
{
    DeviceContext * lResult   ;
    unsigned int    lSize_byte;

    // printk( KERN_DEBUG "%s( 0x%px, %u, %u, %u, 0x%px )\n", __FUNCTION__, aPciDev, aMajor, aMinor, aIndex, aClass );

    ASSERT( NULL != aPciDev );
    ASSERT( NULL != aClass  );

    lSize_byte = sizeof( DeviceContext ) + DeviceCpp_GetContextSize();

    // The context is allocated using vmalloc to be page aligned. This way,
    // the 12 lower bits of the address are all 0 and can be used to pass
    // buffer index to the FreeCallback.

    // vmalloc ==> vfree  See Device_Delete
    lResult = vmalloc( lSize_byte );
    if ( NULL == lResult )
    {
        printk( KERN_ERR "%s - vmalloc( %u bytes )\n", __FUNCTION__, lSize_byte );
    }
    else
    {
        int lRet;

        ASSERT( 0 == ( (uint64_t )( lResult ) & 0x00000fff ) ); // reinterpret_cast

        memset( lResult, 0, lSize_byte );

        lResult->mClass  = aClass ;
        lResult->mDevId  = MKDEV( aMajor, aMinor );
        lResult->mPciDev = aPciDev;

        OSDep_Init( lResult );

        // ===== Initialise the linux structures ============================
        spin_lock_init( & lResult->mAdapterLock  );
        spin_lock_init( & lResult->mHardwareLock );
        tasklet_init  ( & lResult->mTasklet, Tasklet, (long unsigned int)( lResult ) ); // reinterpret_cast
        timer_setup   ( & lResult->mTimer  , Timer  , 0 );
        INIT_WORK     ( & lResult->mWork   , Work   );

        // DeviceCpp_Init ==> DeviceCpp_Uninit  See Device_Delete
        DeviceCpp_Init( lResult->mDeviceCpp, & lResult->mOSDep, & lResult->mAdapterLock, & lResult->mHardwareLock );

        lRet = pci_write_config_word( aPciDev, 6, 0x0800 );
        if ( 0 != lRet )
        {
            printk( KERN_ERR "%s - pci_enable_device_mem( 0x%px, ,  ) failed - %d\n", __FUNCTION__, aPciDev, lRet );
        }

        // pci_enable_device_mem ==> pci_disable_device  See Device_Delete
        lRet = pci_enable_device_mem( aPciDev );
        if ( 0 != lRet )
        {
            printk( KERN_ERR "%s - pci_enable_device_mem( 0x%px ) failed - %d\n", __FUNCTION__, aPciDev, lRet );
            goto Error0;
        }

        // IoMem_Init ==> IoMem_Uninit  See Device_Delete
        if ( 0 != IoMem_Init( lResult ) )  { goto Error1; }

        // Dma_Init ==> Dma_Uninit  See Device_Delete
        if ( 0 != Dma_Init( lResult ) ) { goto Error2; }

        // Interrupt_Init ==> Interrupt_Uninit  See Device_Delete
        if ( 0 != Interrupt_Init( lResult ) ) { goto Error3; }

        // Device_Init ==> Device_Uninit  See Device_Delete
        if ( 0 != Device_Init( lResult, aIndex ) ) { goto Error4; }

        Timer_Start( lResult );
    }

    return lResult;

Error4:
    Interrupt_Uninit( lResult );

Error3:
    Dma_Uninit( lResult );

Error2:
    IoMem_Uninit( lResult );

Error1:
    pci_disable_device( aPciDev );

Error0:
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

    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != lThis->mPciDev );

    lThis->mFlags.mDeleting = true;

    // add_timer --> del_timer_sync  See Timer_Start
    del_timer_sync( & lThis->mTimer );

    // schedule_work --> flush_work  See Tasklet
    flush_work( & lThis->mWork );

    // Device_Init ==> Device_Uninit  See Device_Create
    Device_Uninit( lThis );

    // Interrupt_Init ==> Interrupt_Uninit  See Device_Create
    Interrupt_Uninit( lThis );

    // Dma_Init ==> Dma_Uninit  See Device_Create
    Dma_Uninit( lThis );

    // DeviceCpp_Init ==> DeviceCpp_Uninit  See Device_Create
    DeviceCpp_Uninit( lThis->mDeviceCpp );

    // IoMem_Init ==> IoMem_Uninit  See Device_Create
    IoMem_Uninit( lThis );

    // pci_enable_device_mem ==> pci_disable_device  See Device_Delete
    pci_disable_device( lThis->mPciDev );

    // vmalloc ==> vfree  See Device_Create
    vfree( lThis );
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aThis [---;RW-]
//
// Return
//    0  OK
//  < 0  Error
int CommonBuffer_Allocate( DeviceContext * aThis )
{
    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL == aThis->mCommon_CA        );
    ASSERT(    0 == aThis->mCommon_PA        );
    ASSERT(    0 == aThis->mCommon_Size_byte );
    ASSERT( NULL != aThis->mPciDev           );

    aThis->mCommon_Size_byte = DeviceCpp_CommonBuffer_GetSize( aThis->mDeviceCpp );
    if ( 0 < aThis->mCommon_Size_byte )
    {
        printk( KERN_INFO "%s - Common buffer = %u bytes\n", __FUNCTION__, aThis->mCommon_Size_byte );

        // dma_alloc_coherent ==> dma_free_coherent  See Device_Uninit
        aThis->mCommon_CA = dma_alloc_coherent( & aThis->mPciDev->dev, aThis->mCommon_Size_byte, & aThis->mCommon_PA, GFP_KERNEL );
        if ( NULL == aThis->mCommon_CA )
        {
            printk( KERN_ERR "%s - dma_alloc_coherent( 0x%px, %u bytes, 0x%px,  ) failed\n", __FUNCTION__, & aThis->mPciDev->dev, aThis->mCommon_Size_byte, & aThis->mCommon_PA );
            return ( - __LINE__ );
        }

        DeviceCpp_CommonBuffer_Set( aThis->mDeviceCpp, aThis->mCommon_PA, aThis->mCommon_CA );
    }

    return 0;
}

// aOut       [---;-W-] The kernel buffer
// aIn_UA     [---;R--] The user buffer
// aMax_byte            The maximum size to copy
// aMin_byte            The minimum size to copy
// aInfo_byte [---;RW-] The function puts the copied size there.
//
// Return
//    0  OK
//  < 0  Error
int Copy_FromUser( void * aOut, const void * aIn_UA, unsigned int aMax_byte, unsigned int aMin_byte, unsigned int * aInfo_byte )
{
    ASSERT( NULL != aOut       );
    ASSERT( NULL != aInfo_byte );

    if ( 0 < aMax_byte )
    {
        int lSize_byte;

        ASSERT( NULL      != aIn_UA    );
        ASSERT(         0 <  aMin_byte );
        ASSERT( aMax_byte >= aMin_byte );

        lSize_byte = copy_from_user( aOut, aIn_UA, aMax_byte );
        if ( ( aMax_byte - aMin_byte ) < lSize_byte )
        {
            printk( KERN_ERR "%s - copy_from_user( 0x%px, 0x%px, %u bytes ) failed - %u\n", __FUNCTION__, aOut, aIn_UA, aMax_byte, lSize_byte );
            return ( - __LINE__ );
        }

        ( * aInfo_byte ) = aMax_byte - lSize_byte;
    }

    return 0;
}

// aOut_UA [---;-W-] The user buffer
// aIn     [---;R--] The kernel buffer
// aSize_byte        The size to copy
//
// Return
//    0  OK
//  < 0  Error
int Copy_ToUser( void * aOut_UA, const void * aIn, unsigned int aSize_byte )
{
    // printk( KERN_DEBUG "%s( 0x%px, 0x%px, %u byte )\n", __FUNCTION__, aOut_UA, aIn, aSize_byte );

    ASSERT( NULL != aIn );

    if ( 0 < aSize_byte )
    {
        int lSize_byte;

        ASSERT( NULL != aOut_UA );

        lSize_byte = copy_to_user( aOut_UA, aIn, aSize_byte );
        if ( 0 != lSize_byte )
        {
            printk( KERN_ERR "%s - copy_to_user( %p, %p, %u bytes ) failed - %d\n", __FUNCTION__, aOut_UA, aIn, aSize_byte, lSize_byte );
            return ( - __LINE__ );
        }
    }

    // QUESTION  Faudrait-il retourner la taille de donnees copiees?
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
    int lRet   ;
    int lResult;

    // printk( KERN_DEBUG "%s( 0x%px, %u )\n", __FUNCTION__, aThis, aIndex );

    ASSERT( NULL != aThis );

    ASSERT( NULL == aThis->mCDev   );
    ASSERT( NULL != aThis->mClass  );
    ASSERT( NULL == aThis->mDevice );
    ASSERT( NULL != aThis->mPciDev );

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
    if ( 0 == lRet )
    {
        // device_create ==> device_destroy  See Device_Uninit
        aThis->mDevice = device_create( aThis->mClass, NULL, aThis->mDevId, NULL, "ONK_Pro1000_%u", aIndex );
        if ( NULL == aThis->mDevice )
        {
            printk( KERN_ERR "%s - device_create( 0x%px, , , , , %u ) failed\n", __FUNCTION__, aThis->mClass, aIndex );
            lResult = ( - __LINE__ );
        }
        else
        {
            lResult = CommonBuffer_Allocate( aThis );
            if ( 0 == lResult )
            {
                // DeviceCpp_D0_Entry ==> DeviceCpp_D0_Exit  See Device_Uninit
                DeviceCpp_D0_Entry( aThis->mDeviceCpp );;

                // DeviceCpp_Interrupt_Enabled ==> DeviceCpp_D0_Exit  See Device_Uninit
                DeviceCpp_Interrupt_Enable( aThis->mDeviceCpp );
            }
            else
            {
                device_destroy( aThis->mClass, aThis->mDevId );
            }
        }
    }
    else
    {
        printk( KERN_ERR "%s - cdev_add( 0x%px, ,  ) failed - %d\n", __FUNCTION__, aThis->mCDev, lRet );
        lResult = ( - __LINE__ );
    }

    if ( 0 != lResult )
    {
        cdev_del( aThis->mCDev );
    }

    return lResult;
}

// aThis [---;RW-]
//
// Device_Init ==> Device_Uninit
void Device_Uninit( DeviceContext * aThis )
{
    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mCDev             );
    ASSERT(    0 <  aThis->mCommon_Size_byte );
    ASSERT( NULL != aThis->mCommon_CA        );
    ASSERT(    0 != aThis->mCommon_PA        );
    ASSERT( NULL != aThis->mPciDev           );

    // DeviceCpp_D0_Entry ==> DeviceCpp_D0_Exit  See Device_Uninit
    DeviceCpp_D0_Exit( aThis->mDeviceCpp );

    // dma_alloc_coherent ==> dma_free_coherent  See CommonBuffer_Allocate
    dma_free_coherent( & aThis->mPciDev->dev, aThis->mCommon_Size_byte, aThis->mCommon_CA, aThis->mCommon_PA );

    // device_create ==> device_destroy  See Device_Uninit
    device_destroy( aThis->mClass, aThis->mDevId );

    // cdev_alloc ==> cdev_del  See Device_Init
    cdev_del( aThis->mCDev );
}

// aThis [---;RW-]
//
// Return
//    0  OK
//  < 0  Error
//
// Dma_Init ==> Dma_Uninit
int Dma_Init( DeviceContext * aThis )
{
    int lRet;

    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mPciDev );

    // pci_enable_pci_error_reporting ==> pci_disable_pcie_error_reporting  See Dma_Uninit
    lRet = pci_enable_pcie_error_reporting( aThis->mPciDev );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - pci_enable_pcie_error_reporting( 0x%px ) failed - %d\n", __FUNCTION__, aThis->mPciDev, lRet );
        return ( - __LINE__ );
    }

    lRet = dma_set_mask_and_coherent( & aThis->mPciDev->dev, DMA_BIT_MASK( 64 ) );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - dma_set_mask_and_coherent( 0x%px,  ) failed - %d\n", __FUNCTION__, & aThis->mPciDev->dev, lRet );
        lRet = pci_disable_pcie_error_reporting( aThis->mPciDev );
        if ( 0 != lRet )
        {
            printk( KERN_ERR "%s - pci_disable_pcie_error_reporting( 0x%px ) failed - %d\n", __FUNCTION__, aThis->mPciDev, lRet );
            return ( - __LINE__ );
        }

        return ( - __LINE__ );
    }

    // pci_set_master ==> pci_clear_master  See Dma_Uninit
    pci_set_master( aThis->mPciDev );

    return 0;
}

// aThis [---;RW-]
//
// Dma_Init ==> Dma_Uninit
void Dma_Uninit( DeviceContext * aThis )
{
    int lRet;

    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mPciDev );

    // pci_set_master ==> pci_clear_master  See Dma_Init
    pci_clear_master( aThis->mPciDev );

    // pci_enable_pci_error_reporting ==> pci_disable_pcie_error_reporting  See Dma_Init
    lRet = pci_disable_pcie_error_reporting( aThis->mPciDev );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - pci_disable_pcie_error_reporting( 0x%px ) failed - %d\n", __FUNCTION__, aThis->mPciDev, lRet );
    }
}

// aThis [---;RW-]
//
// Return
//    0  OK
//  < 0  Error
//
// Interrupt_Init ==> Interrupt_Uninit
int Interrupt_Init( DeviceContext * aThis )
{
    int lRet;

    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mPciDev      );
    ASSERT(    0 == aThis->mIrq         );
    ASSERT(    0 == aThis->mVectorCount );

    // pci_alloc_irq_vectors ==> pci_free_irq_vectors  See Interrupt_Uninit
    aThis->mVectorCount = pci_alloc_irq_vectors( aThis->mPciDev, VECTOR_MIN, VECTOR_MAX, PCI_IRQ_MSIX );
    if ( ( VECTOR_MIN > aThis->mVectorCount ) || ( VECTOR_MAX < aThis->mVectorCount ) )
    {
        printk( KERN_ERR "%s - pci_alloc_irq_vector( 0x%px, , ,  ) failed - %d\n", __FUNCTION__, aThis->mPciDev, aThis->mVectorCount );
        return ( - __LINE__ );
    }

    aThis->mIrq = pci_irq_vector( aThis->mPciDev, 0 );

    printk( KERN_INFO "%s - IRQ = %d\n", __FUNCTION__, aThis->mIrq );

    // request_irq ==> free_irq  See Interrupt_Uninit
    lRet = request_irq( aThis->mIrq, ProcessIrq, 0, MODULE_NAME, aThis );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - request_irq( %d, , , , 0x%px ) failed - %d\n", __FUNCTION__, aThis->mIrq, aThis, lRet );
        pci_free_irq_vectors( aThis->mPciDev );
        return ( - __LINE__ );
    }

    return 0;
}

// aThis [---;RW-]
//
// Interrupt_Init ==> Interrupt_Uninit
void Interrupt_Uninit( DeviceContext * aThis )
{
    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mPciDev );

    // request_irq ==> free_irq  See Interrupt_Init
    free_irq( aThis->mIrq, aThis );

    // pci_alloc_irq_vectors ==> pci_free_irq_vectors  See Interrupt_Init
    pci_free_irq_vectors( aThis->mPciDev );
}

// aThis [---;RW-]
// aIoCtlResult    See IOCTL_RESULT_...
//
// Return
//    > 0  The number of ouput byte
//  Other  See IOCTL_RESULT_...
int IoCtl_ProcessResult( DeviceContext * aThis, int aIoCtlResult )
{
    int lResult = aIoCtlResult;

    // printk( KERN_DEBUG "%s( 0x%px, %d )\n", __FUNCTION__, aThis, aIoCtlResult );

    ASSERT( NULL != aThis );

    if ( 0 > lResult )
    {
        switch ( aIoCtlResult )
        {
        case IOCTL_RESULT_PROCESSING_NEEDED :
            tasklet_schedule( & aThis->mTasklet );
            lResult = IOCTL_RESULT_OK;
            break;

        default:
            printk( KERN_ERR "%s - The IoCtl failed returning %d\n", __FUNCTION__, lResult );
        }
    }

    return lResult;
}

// aThis   [---;RW-]
// aFile   [---;---]
// aCode
// aArg_UA [---;RW-]
// aInfo   [---;R--]
// aInSizeMin_byte
// aOutSize_byte
// aSize_byte
//
// Return
//    0  OK
//  < 0  See IOCTL_RESULT_... or another error code
int IoCtl_WithArgument( DeviceContext * aThis, struct file * aFile, unsigned int aCode, unsigned long aArg_UA, const OpenNetK_IoCtl_Info * aInfo, unsigned int aSize_byte )
{
    void       * lArg_UA;
    void       * lInOut ;
    unsigned int lInSize_byte = 0; // Avoid the warning
    int          lResult;

    // printk( KERN_DEBUG "%s( 0x%px, , 0x%lx, , , %u bytes )\n", __FUNCTION__, aThis, aArg_UA, aSize_byte );

    ASSERT(    0 <  aSize_byte );
    ASSERT( NULL != aFile      );
    ASSERT( NULL != aInfo      );

    if ( 0 == aArg_UA )
    {
        printk( KERN_ERR "%s - Invalid NULL argument\n", __FUNCTION__ );
        return ( - __LINE__ );
    }

    lInOut = kmalloc( aSize_byte, GFP_KERNEL );
    if ( NULL == lInOut )
    {
        printk( KERN_ERR "%s - kmalloc( %u bytes,  ) failed\n", __FUNCTION__, aSize_byte );
        return ( - __LINE__ );
    }

    memset( lInOut, 0, aSize_byte );

    lArg_UA = (void *)( aArg_UA ); // reinterpret_cast

    lResult = Copy_FromUser( lInOut, lArg_UA, aInfo->mIn_MaxSize_byte, aInfo->mIn_MinSize_byte, & lInSize_byte );
    if ( 0 == lResult )
    {
        int lRet = DeviceCpp_IoCtl( & aThis->mDeviceCpp, aFile, aCode, lInOut, lInSize_byte, lInOut, aInfo->mOut_MinSize_byte );

        lResult = IoCtl_ProcessResult( aThis, lRet );
        if ( 0 < lResult )
        {
            lResult = Copy_ToUser( lArg_UA, lInOut, ( aInfo->mOut_MinSize_byte < lResult ) ? aInfo->mOut_MinSize_byte : lResult );
        }
    }

    kfree( lInOut );

    return lResult;
}

// aThis [---;RW-]
//
// Return
//    0  OK
//  < 0  Error
//
// IoMem_Init ==> IoMem_Uninit
int IoMem_Init( DeviceContext * aThis )
{
    unsigned int lMemCount = 0;
    int          lRet;

    unsigned int i;

    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mPciDev );

    // pci_request_mem_regions ==> pci_release_mem_regions  See IoMem_Uninit
    lRet = pci_request_mem_regions( aThis->mPciDev, MODULE_NAME );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - pci_request_mem_regions( 0x%px,  ) failed - %d\n", __FUNCTION__, aThis, lRet );
        return ( - __LINE__ );
    }

    for ( i = 0; i < BAR_QTY; i ++ )
    {
        ASSERT( NULL == aThis->mIoMems_MA[ lMemCount ] );

        if ( IORESOURCE_MEM == ( pci_resource_flags( aThis->mPciDev, i ) & IORESOURCE_MEM ) )
        {
            int lResult;

            unsigned int lSize_byte = pci_resource_len( aThis->mPciDev, i );
            ASSERT( 0 <  lSize_byte );

            // ioremap_nocache ==> iounmap  See IoMem_Uninit
            aThis->mIoMems_MA[ lMemCount ] = ioremap_nocache( pci_resource_start( aThis->mPciDev, i ), lSize_byte );
            ASSERT( NULL != aThis->mIoMems_MA[ lMemCount ] );

            // DeviceCpp_SetMemory ==> DeviceCpp_ResetMemory  See IoMem_Uninit
            lResult = DeviceCpp_SetMemory( aThis->mDeviceCpp, lMemCount, aThis->mIoMems_MA[ lMemCount ], lSize_byte );
            if ( 0 != lResult )
            {
                return lResult;
            }

            lMemCount ++;
        }
    }

    return 0;
}

// aThis [---;RW-]
//
// IoMem_Init ==> IoMem_Uninit
void IoMem_Uninit( DeviceContext * aThis )
{
    unsigned int i;

    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mPciDev );

    // DeviceCpp_SetMemory ==> DeviceCpp_ResetMemory  See IoMem_Init
    DeviceCpp_ResetMemory( aThis->mDeviceCpp );

    for ( i = 0; i < BAR_QTY; i ++ )
    {
        if ( NULL != aThis->mIoMems_MA[ i ] )
        {
            // ioremap_nocache ==> iounmap  See IoMem_Init
            iounmap( aThis->mIoMems_MA[ i ] );
        }
    }

    // pci_request_mem_regions ==> pci_release_mem_regions  See IoMem_Init
    pci_release_mem_regions( aThis->mPciDev );
}

// aThis [---;-W-]
void OSDep_Init( DeviceContext * aThis )
{
    ASSERT( NULL != aThis );

    aThis->mOSDep.mContext = aThis;

    aThis->mOSDep.AllocateMemory    = AllocateMemory   ;
    aThis->mOSDep.FreeMemory        = FreeMemory       ;
    aThis->mOSDep.GetTimeStamp      = GetTimeStamp     ;
    aThis->mOSDep.MapBuffer         = MapBuffer        ;
    aThis->mOSDep.UnmapBuffer       = UnmapBuffer      ;
    aThis->mOSDep.MapSharedMemory   = MapSharedMemory  ;
    aThis->mOSDep.UnmapSharedMemory = UnmapSharedMemory;
    aThis->mOSDep.LockSpinlock      = LockSpinlock     ;
    aThis->mOSDep.UnlockSpinlock    = UnlockSpinlock   ;
}

// aThis [---;RW-]
void Timer_Start( DeviceContext * aThis )
{
    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    aThis->mTimer.expires = jiffies + ( HZ / 10 );

    // add_timer ==> del_timer_sync  See Device_Delete
    add_timer( & aThis->mTimer );
}

// ===== Entry points =======================================================

long IoCtl( struct file * aFile, unsigned int aCode, unsigned long aArg_UA )
{
    OpenNetK_IoCtl_Info lInfo  ;
    int                 lResult;
    DeviceContext     * lThis  ;

    // printk( KERN_DEBUG "%s( %px, 0x%08x, 0x%lx )\n", __FUNCTION__, aFile, aCode, aArg_UA );

    ASSERT( NULL != aFile );

    ASSERT( NULL != aFile->private_data );

    lThis = (DeviceContext *)( aFile->private_data ); // reinterpret_cast

    lResult = DeviceCpp_IoCtl_GetInfo( aCode, & lInfo );
    if ( 0 == lResult )
    {
        int lRet;

        unsigned int lSize_byte = ( lInfo.mIn_MaxSize_byte > lInfo.mOut_MinSize_byte ) ? lInfo.mIn_MaxSize_byte : lInfo.mOut_MinSize_byte;
        if ( 0 == lSize_byte )
        {
            lRet    = DeviceCpp_IoCtl( & lThis->mDeviceCpp, aFile, aCode, NULL, 0, NULL, 0 );
            lResult = IoCtl_ProcessResult( lThis, lRet );
        }
        else
        {
            lResult = IoCtl_WithArgument( lThis, aFile, aCode, aArg_UA, & lInfo, lSize_byte );
        }
    }

    return lResult;
}

int Open( struct inode * aINode, struct file * aFile )
{
    // printk( KERN_DEBUG "%s( 0x%px, 0x%px )\n", __FUNCTION__, aINode, aFile );

    ASSERT( NULL != aINode );
    ASSERT( NULL != aFile  );

    ASSERT( NULL == aFile->private_data );

    aFile->private_data = Driver_FindDevice( iminor( aINode ) );

    return 0;
}

int Release( struct inode * aINode, struct file * aFile )
{
    DeviceContext * lThis;

    // printk( KERN_DEBUG "%s( 0x%px, 0x%px )\n", __FUNCTION__, aINode, aFile );

    ASSERT( NULL != aFile );

    ASSERT( NULL != aFile->private_data );

    lThis = (DeviceContext *)( aFile->private_data ); // reinterpret_cast

    DeviceCpp_Release( & lThis->mDeviceCpp, aFile );

    // printk( KERN_DEBUG "%s - OK\n", __FUNCTION__ );

    return 0;
}

irqreturn_t ProcessIrq( int aIrq, void * aDevId )
{
    DeviceContext * lThis = aDevId;

    // printk( KERN_DEBUG "%s( %d, 0x%px )\n", __FUNCTION__, aIrq, aDevId );

    ASSERT( NULL != aDevId );

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

    bool lNeedMoreProcessing = false;

    // printk( KERN_DEBUG "%s( 0x%lx )\n", __FUNCTION__, aData );

    ASSERT( 0 != aData );

    DeviceCpp_Interrupt_Process2( & lThis->mDeviceCpp, & lNeedMoreProcessing );

    if ( lNeedMoreProcessing && ( ! lThis->mFlags.mDeleting ) )
    {
        // schedule_work --> flush_work  See Device_Delete
        schedule_work( & lThis->mWork );
    }
}

void Timer( struct timer_list * aTimer )
{
    DeviceContext * lThis = container_of( aTimer, DeviceContext, mTimer );

    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aTimer );

    ASSERT( NULL != aTimer );

    DeviceCpp_Tick( lThis->mDeviceCpp );

    tasklet_schedule( & lThis->mTasklet );

    if ( ! lThis->mFlags.mDeleting )
    {
        Timer_Start( lThis );
    }
}

void Work( struct work_struct * aWork )
{
    DeviceContext * lThis = container_of( aWork, DeviceContext, mWork );

    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aWork );

    ASSERT( NULL != aWork );

    DeviceCpp_Interrupt_Process3( lThis->mDeviceCpp );
}

// ===== OSDep ==============================================================

// AllocateMemory ==> FreeMemory
void * AllocateMemory( unsigned int aSize_byte )
{
    // printk( KERN_DEBUG "%s( %u bytes )", __FUNCTION__, aSize_byte );

    ASSERT( 0 < aSize_byte );

    // kmalloc ==> kfree  See FreeMemory
    return kmalloc( aSize_byte, GFP_KERNEL );
}

// AllocateMemory ==> FreeMemory
void FreeMemory( void * aMemory )
{
    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aMemory );

    ASSERT( NULL != aMemory );

    // kmalloc ==> kfree  See AllocateMemory
    kfree( aMemory );
}

uint64_t GetTimeStamp()
{
    uint64_t lResult = jiffies;

    lResult *= 1000000;
    lResult /= HZ     ;

    return lResult;
}

// MapBuffer ==> UnmapBuffer or FreeCallback
void * MapBuffer( void * aContext, uint64_t * aBuffer_PA, uint64_t aBuffer_DA, unsigned int aSize_byte, uint64_t aMarker_PA, volatile void * * aMarker_MA )
{
    unsigned int i;

    DeviceContext * lThis = aContext;

    // printk( KERN_DEBUG "%s( 0x%px, 0x%px, 0x%llx, %u bytes, ,  )\n", __FUNCTION__, aContext, aBuffer_PA, aBuffer_DA, aSize_byte );

    ASSERT( NULL !=   aContext              );
    ASSERT(    0 !=   aBuffer_DA            );
    ASSERT(    0 == ( aBuffer_DA & 0xffff ) );
    ASSERT(    0 <    aSize_byte            );
    ASSERT( NULL !=   aMarker_MA            );

    ( * aMarker_MA ) = NULL;

    for ( i = 0; i < OPEN_NET_BUFFER_QTY; i ++ )
    {
        if ( NvBuffer_IsAvailable( lThis->mBuffers + i ) )
        {
            uint64_t lData_XA = (uint64_t)( lThis ); //reinterpret_cast
            int      lRet;

            ASSERT( 0 == ( lData_XA & 0x00000fff ) );

            lData_XA |= i;

            lRet = NvBuffer_Map( lThis->mBuffers + i, lThis->mPciDev, aBuffer_DA, aSize_byte, FreeCallback, (void *)( lData_XA ) ); // reinterpret_cast
            if ( 0 != lRet )
            {
                return NULL;
            }

            ( * aBuffer_PA ) = NvBuffer_GetPhysicalAddress( lThis->mBuffers + i );

            return NvBuffer_GetMappedAddress( lThis->mBuffers + i );
        }
    }

    printk( KERN_ERR "%s - Too many buffer\n", __FUNCTION__ );
    return NULL;
}

// MapBuffer ==> UnmapBuffer
void UnmapBuffer( void * aContext, void * aBuffer_MA, unsigned int aSize_byte, volatile void * aMarker_MA )
{
    unsigned int i;

    DeviceContext * lThis = aContext;

    // printk( KERN_DEBUG "%s( 0x%px, 0x%px, %u bytes,  )\n", __FUNCTION__, aContext, aBuffer_MA, aSize_byte );

    ASSERT( NULL != aContext   );
    ASSERT( NULL != aBuffer_MA );
    ASSERT(    0 <  aSize_byte );
    ASSERT( NULL == aMarker_MA );

    ASSERT( NULL != lThis->mPciDev );

    for (  i = 0; i < OPEN_NET_BUFFER_QTY; i ++ )
    {
        if ( NvBuffer_Is( lThis->mBuffers + i, aBuffer_MA ) )
        {
            // NvBuffer_Map ==> NvBuffer_Unmap  See MapBuffer
            NvBuffer_Unmap( lThis->mBuffers + i, lThis->mPciDev, true );
            return;
        }
    }

    printk( KERN_ERR "%s - Invalid buffer\n", __FUNCTION__ );
}

// MapSharedMemory ==> UnmapSharedMemory
void * MapSharedMemory( void * aContext, void * aShared_VA, unsigned int aSize_byte )
{
    int lRet;

    DeviceContext * lThis = aContext;

    // printk( KERN_DEBUG "%s( 0x%px, 0x%px, %u bytes )\n", __FUNCTION__, aContext, aShared_VA, aSize_byte );

    ASSERT( NULL != aContext   );
    ASSERT( NULL != aShared_VA );
    ASSERT(    0 <  aSize_byte );

    lThis->mShared_PageCount = aSize_byte / PAGE_SIZE;

    if ( 0 != ( aSize_byte % PAGE_SIZE ) )
    {
        lThis->mShared_PageCount ++;
    }

    // kmalloc ==> kfree  See UnmapSharedMemory
    lThis->mShared_Pages = kmalloc( sizeof( struct page * ) * lThis->mShared_PageCount, GFP_KERNEL );
    if ( NULL == lThis )
    {
        printk( KERN_ERR "%s - kmallog( %lu bytes,  ) failed\n", __FUNCTION__, sizeof( struct page * ) * lThis->mShared_PageCount );
        return NULL;
    }

    down_write( & current->mm->mmap_sem );

        // get_user_pages ==> put_page  See UnmapSharedMemory
        lRet = get_user_pages( (long unsigned int)( aShared_VA ), lThis->mShared_PageCount, 1, lThis->mShared_Pages, NULL ); // reinterpret_cast
        if ( lThis->mShared_PageCount == lRet )
        {
            // vmap ==> vunmap  See UnmapSharedMemory
            lThis->mShared = vmap( lThis->mShared_Pages, lThis->mShared_PageCount, VM_MAP, PAGE_KERNEL );
        }
        else
        {
            printk( KERN_ERR "%s - get_user_pages( 0x%px, %u, , 0x%px,  ) failed - %d\n", __FUNCTION__, aShared_VA, lThis->mShared_PageCount, lThis->mShared_Pages, lRet );
        }

    up_write( & current->mm->mmap_sem );

    if ( NULL == lThis->mShared )
    {
        printk( KERN_ERR "%s - vmap( 0x%px, %u, ,  ) failed\n", __FUNCTION__, lThis->mShared_Pages, lThis->mShared_PageCount );
        kfree( lThis->mShared_Pages );
        lThis->mShared_PageCount =    0;
        lThis->mShared_Pages     = NULL;
    }

    return lThis->mShared;
}

// MapSharedMemory ==> UnmapSharedMemory
static void UnmapSharedMemory( void * aContext )
{
    unsigned int i;

    DeviceContext * lThis = aContext;

    // printk( KERN_DEBUG "%s( 0x%px )\n", __FUNCTION__, aContext );

    ASSERT( NULL != aContext );

    ASSERT( NULL != lThis->mShared           );
    ASSERT(    0 <  lThis->mShared_PageCount );
    ASSERT( NULL != lThis->mShared_Pages     );

    // vmap ==> vunmap  See MapSharedMemory
    vunmap( lThis->mShared );
    lThis->mShared = NULL;

    for ( i = 0; i < lThis->mShared_PageCount; i ++ )
    {
        ASSERT( NULL != lThis->mShared_Pages[ i ] );

        // get_user_pages ==> put_page  See MapSharedMemory
        put_page( lThis->mShared_Pages[ i ] );
    }

    lThis->mShared_PageCount = 0;

    // kmalloc ==> kfree  See MapSharedMemory
    kfree( lThis->mShared_Pages );
    lThis->mShared_Pages = NULL;
}

void LockSpinlock( void * aLock )
{
    ASSERT( NULL != aLock );

    spin_lock( aLock );
}

void UnlockSpinlock( void * aLock )
{
    ASSERT( NULL != aLock );

    spin_unlock( aLock );
}

// ===== NVIDIA callback ====================================================

// MapBuffer ==> FreeCallback
void FreeCallback( void * aData_XA )
{
    uint64_t        lData_XA = (uint64_t)( aData_XA ); // reinterpret_cast
    unsigned int    lIndex   = lData_XA % PAGE_SIZE;
    DeviceContext * lThis    = (void *)( lData_XA & PAGE_MASK ); // reinterpret_cast

    // printk( KERN_DEBUG "%s( 0x%llx )\n", __FUNCTION__, lData_XA );

    ASSERT( NULL != aData_XA );

    ASSERT( NULL != lThis->mPciDev );

    ASSERT( OPEN_NET_BUFFER_QTY > lIndex );

    NvBuffer_Unmap( lThis->mBuffers + lIndex, lThis->mPciDev, false );
}
