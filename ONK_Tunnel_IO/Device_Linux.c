
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/Device_Linux.cpp

#define _KMS_LINUX_

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Linux ==============================================================
#include <linux/aer.h>
#include <linux/cdev.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/timer.h>
#include <linux/uaccess.h>

// ===== Includes ===========================================================
#include <OpenNetK/Constants.h>
#include <OpenNetK/IoCtl.h>
#include <OpenNetK/Linux.h>
#include <OpenNetK/OSDep.h>
#include <OpenNetK/Tunnel.h>

// ===== ONK_Pro1000 ========================================================
#include "DeviceCpp.h"
#include "Driver_Linux.h"

#include "Device_Linux.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define MODULE_NAME "ONK_Tunnel_IO"

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

    // ===== Linux structures ===============================================
    spinlock_t mAdapterLock ;
    spinlock_t mHardwareLock;

    struct timer_list     mTimer  ;

    // ===== Shared memory information ======================================
    void          * mShared          ;
    unsigned int    mShared_PageCount;
    struct page * * mShared_Pages    ;

    // ===== Other ==========================================================

    // The major and minor number associated to this device.
    dev_t mDevId;

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

static int  Copy_FromUser( void * aOut_UA, const void * aIn   , unsigned int aMax_byte, unsigned int aMin_byte, unsigned int * aInfo_byte );
static int  Copy_ToUser  ( void * aOut   , const void * aIn_UA, unsigned int aSize_byte );

static int  Device_Init  ( DeviceContext * aThis, unsigned int aIndex );
static void Device_Uninit( DeviceContext * aThis );

static int  IoCtl_ProcessResult( DeviceContext * aThis, int aIoCtlResult );
static int  IoCtl_WithArgument ( DeviceContext * aThis, struct file * aFile, unsigned int aCode, unsigned long aArg_UA, const OpenNetK_IoCtl_Info * aInfo, unsigned int aSize_byte );

static void OSDep_Init( DeviceContext * aThis );

static unsigned int Pages_Get( void * aStart_XA, unsigned int aSize_byte, struct page * * * aPages );
static void         Pages_Put( struct page * * aPages, unsigned int aPageCount );

static void Timer_Start( DeviceContext * aThis );

// ===== Entry points =======================================================

static long    IoCtl  ( struct file * aFile, unsigned int aCode, unsigned long aArg_UA );
static int     Open   ( struct inode * aINode, struct file * aFile );
static ssize_t Read   ( struct file * aFile, char * aBuffer_UA, size_t aSize_byte, loff_t * aOffset );
static int     Release( struct inode * aINode, struct file * aFile );

static void Timer( struct timer_list  * aTimer );

// ===== OSDep ==============================================================

static void * AllocateMemory( unsigned int aSize_byte );
static void   FreeMemory    ( void * aMemory );

static uint64_t GetTimeStamp( void );

static void * MapSharedMemory  ( void * aContext, void * aShared_UA, unsigned int aSize_byte );
static void   UnmapSharedMemory( void * aContext );

static void     LockSpinlock            ( void * aLock );
static uint32_t LockSpinlockFromThread  ( void * aLock );
static void     UnlockSpinlock          ( void * aLock );
static void     UnlockSpinlockFromThread( void * aLock, uint32_t aFlags );

// Static variables
/////////////////////////////////////////////////////////////////////////////

static struct file_operations sOperations =
{
    .owner          = THIS_MODULE,
    .open           = Open       ,
    .read           = Read       ,
    .release        = Release    ,
    .unlocked_ioctl = IoCtl      ,
};

// Functions
/////////////////////////////////////////////////////////////////////////////

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
void * Device_Create( unsigned char aMajor, unsigned char aMinor, unsigned int aIndex, struct class * aClass )
{
    DeviceContext * lResult   ;
    unsigned int    lSize_byte;

    // printk( KERN_DEBUG "%s( 0x%p, %u, %u, %u, 0x%p )\n", __FUNCTION__, aPciDev, aMajor, aMinor, aIndex, aClass );

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
        ASSERT( 0 == ( (uint64_t )( lResult ) & 0x00000fff ) ); // reinterpret_cast

        memset( lResult, 0, lSize_byte );

        lResult->mClass  = aClass ;
        lResult->mDevId  = MKDEV( aMajor, aMinor );

        OSDep_Init( lResult );

        // ===== Initialise the linux structures ============================
        spin_lock_init( & lResult->mAdapterLock  );
        spin_lock_init( & lResult->mHardwareLock );
        timer_setup   ( & lResult->mTimer  , Timer  , 0 );

        DeviceCpp_Init( lResult->mDeviceCpp, & lResult->mOSDep, & lResult->mAdapterLock, & lResult->mHardwareLock );

        // Device_Init ==> Device_Uninit  See Device_Delete
        if ( 0 != Device_Init( lResult, aIndex ) ) { goto Error0; }

        Timer_Start( lResult );
    }

    return lResult;

Error0:
    vfree( lResult );
    return NULL;
}

// aThis [D--;RW-] The instance to delete
//
// Device_Create ==> Device_Delete
void Device_Delete( void * aThis )
{
    DeviceContext * lThis = aThis;

    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != lThis->mPciDev );

    lThis->mFlags.mDeleting = true;

    // add_timer --> del_timer_sync  See Timer_Start
    del_timer_sync( & lThis->mTimer );

    // Device_Init ==> Device_Uninit  See Device_Create
    Device_Uninit( lThis );

    // vmalloc ==> vfree  See Device_Create
    vfree( lThis );
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

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
            printk( KERN_ERR "%s - copy_from_user( 0x%p, 0x%px, %u bytes ) failed - %u\n", __FUNCTION__, aOut, aIn_UA, aMax_byte, lSize_byte );
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
    // printk( KERN_DEBUG "%s( 0x%px, 0x%p, %u byte )\n", __FUNCTION__, aOut_UA, aIn, aSize_byte );

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

    // printk( KERN_DEBUG "%s( 0x%p, %u )\n", __FUNCTION__, aThis, aIndex );

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
        aThis->mDevice = device_create( aThis->mClass, NULL, aThis->mDevId, NULL, "OpenNet%u", aIndex );
        if ( NULL == aThis->mDevice )
        {
            printk( KERN_ERR "%s - device_create( 0x%p, , , , , %u ) failed\n", __FUNCTION__, aThis->mClass, aIndex );
            lResult = ( - __LINE__ );
        }
        else
        {
            // DeviceCpp_D0_Entry ==> DeviceCpp_D0_Exit  See Device_Uninit
            DeviceCpp_D0_Entry( aThis->mDeviceCpp );

            lResult = 0;
        }
    }
    else
    {
        printk( KERN_ERR "%s - cdev_add( 0x%p, ,  ) failed - %d\n", __FUNCTION__, aThis->mCDev, lRet );
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
    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mCDev             );
    ASSERT(    0 <  aThis->mCommon_Size_byte );
    ASSERT( NULL != aThis->mCommon_CA        );
    ASSERT(    0 != aThis->mCommon_PA        );
    ASSERT( NULL != aThis->mPciDev           );

    // DeviceCpp_D0_Entry ==> DeviceCpp_D0_Exit  See Device_Uninit
    DeviceCpp_D0_Exit( aThis->mDeviceCpp );

    // device_create ==> device_destroy  See Device_Uninit
    device_destroy( aThis->mClass, aThis->mDevId );

    // cdev_alloc ==> cdev_del  See Device_Init
    cdev_del( aThis->mCDev );
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

    // printk( KERN_DEBUG "%s( 0x%p, %d )\n", __FUNCTION__, aThis, aIoCtlResult );

    ASSERT( NULL != aThis );

    if ( 0 > lResult )
    {
        switch ( aIoCtlResult )
        {
        case IOCTL_RESULT_PROCESSING_NEEDED :
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

    // printk( KERN_DEBUG "%s( 0x%p, , 0x%lx, , , %u bytes )\n", __FUNCTION__, aThis, aArg_UA, aSize_byte );

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

// aThis [---;-W-]
void OSDep_Init( DeviceContext * aThis )
{
    ASSERT( NULL != aThis );

    aThis->mOSDep.mContext = aThis;

    aThis->mOSDep.AllocateMemory           = AllocateMemory          ;
    aThis->mOSDep.FreeMemory               = FreeMemory              ;
    aThis->mOSDep.GetTimeStamp             = GetTimeStamp            ;
    aThis->mOSDep.LockSpinlock             = LockSpinlock            ;
    aThis->mOSDep.LockSpinlockFromThread   = LockSpinlockFromThread  ;
    aThis->mOSDep.MapBuffer                = NULL                    ;
    aThis->mOSDep.UnmapBuffer              = NULL                    ;
    aThis->mOSDep.MapSharedMemory          = MapSharedMemory         ;
    aThis->mOSDep.UnmapSharedMemory        = UnmapSharedMemory       ;
    aThis->mOSDep.UnlockSpinlock           = UnlockSpinlock          ;
    aThis->mOSDep.UnlockSpinlockFromThread = UnlockSpinlockFromThread;
}

// aStart_UA [---;RW-]
// aSize_byte
// aPages    [---;-W-]
//
// Return  This function returns the number of pages
//
// Pages_Get ==> Pages_Put
unsigned int Pages_Get( void * aStart_UA, unsigned int aSize_byte, struct page * * * aPages )
{
    struct page * * lPages;
    unsigned int    lResult_page;
    int             lRet;

    printk( KERN_DEBUG "%s( 0x%px, %u bytes,  )\n", __FUNCTION__, aStart_UA, aSize_byte );

    ASSERT( NULL != aStart_UA  );
    ASSERT(    0 <  aSize_byte );
    ASSERT( NULL != aPages     );

    lResult_page = aSize_byte / PAGE_SIZE;

    if ( 0 != ( aSize_byte % PAGE_SIZE ) )
    {
        lResult_page ++;
    }

    if ( 0 != ( ( uint64_t )( aStart_UA ) & 0xfff ) )
    {
        lResult_page ++;
    }

    lPages = kmalloc( sizeof( struct page * ) * lResult_page, GFP_KERNEL );
    if ( NULL == lPages )
    {
        printk( KERN_DEBUG "%s - kmalloc( ,  ) failed\n", __FUNCTION__ );
        return 0;
    }

    down_write( & current->mm->mmap_sem );

        lRet = get_user_pages( ( long unsigned int )( aStart_UA ), lResult_page, 1, lPages, NULL );

    up_write( & current->mm->mmap_sem );

    if ( 0 >= lRet )
    {
        printk( KERN_DEBUG "%s - get_user_pages( 0x%px, %u pages, , , ,  ) failed\n", __FUNCTION__, aStart_UA, lResult_page );
        kfree( lPages );
        return 0;
    }

    lResult_page = lRet;

    ( * aPages ) = lPages;

    return lResult_page;
}

// aPages [D--;R--]
// aPageCount
//
// Pages_Get ==> Pages_Put
void Pages_Put( struct page * * aPages, unsigned int aPageCount )
{
    unsigned int i;

    printk( KERN_DEBUG "%s( , %u pages )\n", __FUNCTION__, aPageCount );

    ASSERT( NULL != aPages     );
    ASSERT(    0 <  aPageCount );

    for ( i = 0; i < aPageCount; i ++ )
    {
        ASSERT( NULL != aPages[ i ] );

        set_page_dirty( aPages[ i ] );

        // get_user_pages ==> put_page  See Pages_Get
        put_page( aPages[ i ] );
    }

    // kmalloc ==> kfree  See Pages_Get
    kfree( aPages );
}

// aThis [---;RW-]
void Timer_Start( DeviceContext * aThis )
{
    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aThis );

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

    // printk( KERN_DEBUG "%s( %p, 0x%08x, 0x%lx )\n", __FUNCTION__, aFile, aCode, aArg_UA );

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
    // printk( KERN_DEBUG "%s( 0x%p, 0x%p )\n", __FUNCTION__, aINode, aFile );

    ASSERT( NULL != aINode );
    ASSERT( NULL != aFile  );

    ASSERT( NULL == aFile->private_data );

    aFile->private_data = Driver_FindDevice( iminor( aINode ) );

    return 0;
}

ssize_t Read( struct file * aFile, char * aBuffer_UA, size_t aSize_byte, loff_t * aOffset )
{
    unsigned int    lPageCount;
    struct page * * lPages    ;
    ssize_t         lResult   ;
    DeviceContext * lThis     ;

    printk( KERN_DEBUG "%s( 0x%p, 0x%px, %lu bytes,  )\n", __FUNCTION__, aFile, aBuffer_UA, aSize_byte );

    ASSERT( NULL != aFile );

    ASSERT( NULL != aFile->private_data );

    lThis = ( DeviceContext * )( aFile->private_data ); // reinterpret_cast

    if ( ( NULL == aBuffer_UA ) || ( sizeof( OpenNet_Tunnel_PacketHeader ) >= aSize_byte ) )
    {
        printk( KERN_DEBUG "%s - Invalid user buffer\n", __FUNCTION__ );
        return ( - __LINE__ );
    }

    lPageCount = Pages_Get( aBuffer_UA, aSize_byte, & lPages );
    if ( 0 >= lPageCount )
    {
        return ( - __LINE__ );
    }

    lResult = DeviceCpp_Read( & lThis->mDeviceCpp, aBuffer_UA, ( unsigned int )( aSize_byte ) ); // static_cast

    Pages_Put( lPages, lPageCount );

    return lResult;
}

int Release( struct inode * aINode, struct file * aFile )
{
    DeviceContext * lThis;

    // printk( KERN_DEBUG "%s( 0x%p, 0x%p )\n", __FUNCTION__, aINode, aFile );

    ASSERT( NULL != aFile );

    ASSERT( NULL != aFile->private_data );

    lThis = (DeviceContext *)( aFile->private_data ); // reinterpret_cast

    DeviceCpp_Release( & lThis->mDeviceCpp, aFile );

    // printk( KERN_DEBUG "%s - OK\n", __FUNCTION__ );

    return 0;
}

// CRITICAL PATH  Interrupt
//                1 / tick
void Timer( struct timer_list * aTimer )
{
    DeviceContext * lThis = container_of( aTimer, DeviceContext, mTimer );

    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aTimer );

    ASSERT( NULL != aTimer );

    DeviceCpp_Tick( lThis->mDeviceCpp );

    if ( ! lThis->mFlags.mDeleting )
    {
        Timer_Start( lThis );
    }
}

// ===== OSDep ==============================================================

void * AllocateMemory( unsigned int aSize_byte )
{
    // printk( KERN_DEBUG "%s( %u bytes )", __FUNCTION__, aSize_byte );

    ASSERT( 0 < aSize_byte );

    // kmalloc ==> kfree  See FreeMemory
    return kmalloc( aSize_byte, GFP_KERNEL );
}

void FreeMemory( void * aMemory )
{
    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aMemory );

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

void * MapSharedMemory( void * aContext, void * aShared_VA, unsigned int aSize_byte )
{
    int lRet;

    DeviceContext * lThis = aContext;

    // printk( KERN_DEBUG "%s( 0x%p, 0x%px, %u bytes )\n", __FUNCTION__, aContext, aShared_VA, aSize_byte );

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
            ASSERT( NULL != lThis->mShared );
        }
        else
        {
            printk( KERN_ERR "%s - get_user_pages( 0x%px, %u, , 0x%p,  ) failed - %d\n", __FUNCTION__, aShared_VA, lThis->mShared_PageCount, lThis->mShared_Pages, lRet );
        }

    up_write( & current->mm->mmap_sem );

    if ( NULL == lThis->mShared )
    {
        printk( KERN_ERR "%s - vmap( 0x%p, %u, ,  ) failed\n", __FUNCTION__, lThis->mShared_Pages, lThis->mShared_PageCount );
        kfree( lThis->mShared_Pages );
        lThis->mShared_PageCount =    0;
        lThis->mShared_Pages     = NULL;
    }

    return lThis->mShared;
}

static void UnmapSharedMemory( void * aContext )
{
    unsigned int i;

    DeviceContext * lThis = aContext;

    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aContext );

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

uint32_t LockSpinlockFromThread( void * aLock )
{
    unsigned long lResult;

    ASSERT( NULL != aLock );

    spin_lock_irqsave( aLock, lResult );

    return lResult;
}

void UnlockSpinlock( void * aLock )
{
    ASSERT( NULL != aLock );

    spin_unlock( aLock );
}

void UnlockSpinlockFromThread( void * aLock, uint32_t aFlags )
{
    ASSERT( NULL != aLock );

    spin_unlock_irqrestore( aLock, aFlags );
}
