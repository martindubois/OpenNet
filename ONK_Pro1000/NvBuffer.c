
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/NvBuffer.c

#define _KMS_LINUX_

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Linux ==============================================================
#include <linux/pci.h>

// ===== NVIDIA =============================================================
#include <nv-p2p.h>

// ===== Includes ===========================================================
#include <OpenNetK/Linux.h>

// ===== ONK_Pro1000 ========================================================
#include "NvBuffer.h"

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static void Validate( NvBuffer * aThis );

// Functions
/////////////////////////////////////////////////////////////////////////////

// aThis   [---;R--]
//
// Return  This method return the physical address
uint64_t NvBuffer_GetPhysicalAddress( NvBuffer * aThis )
{
    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mPageTable             );
    ASSERT( NULL != aThis->mPageTable->pages      );
    ASSERT( NULL != aThis->mPageTable->pages[ 0 ] );

    return aThis->mPageTable->pages[ 0 ]->physical_address;
}

// aThis   [---;R--]
//
// Return  This method return the mapped address
void * NvBuffer_GetMappedAddress( NvBuffer * aThis )
{
    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mAddress_MA );

    return aThis->mAddress_MA;
}

// aThis       [---;R--]
// aAddress_MA [---;---]
//
// Return
//  0  No
//  1  Yes it is
int NvBuffer_Is( NvBuffer * aThis, void * aAddress_MA )
{
    // printk( KERN_DEBUG "%s( 0x%p, 0x%p )\n", __FUNCTION__, aThis, aAddress_MA );

    ASSERT( NULL != aThis );

    return ( aThis->mAddress_MA == aAddress_MA );
}

// aThis   [---;RW-]
int NvBuffer_IsAvailable( NvBuffer * aThis )
{
    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    return ( NULL == aThis->mPageTable );
}

// aThis         [---;RW-]
// aPciDev       [---;RW-]
// aAddress_DA   [-K-;-W-]
// aSize_byte
// aFreeCallback [-K-;--X]
// aData         [-KO;---]
//
// Return
//    0  OK
//  < 0  Error
//
// NbBuffer_Map ==> NvBuffer_Unmap
int NvBuffer_Map( NvBuffer * aThis, struct pci_dev * aPciDev, uint64_t aAddress_DA, unsigned int aSize_byte, void ( * aFreeCallback )( void * ), void * aData )
{
    int lRet;

    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis         );
    ASSERT(    0 != aAddress_DA   );
    ASSERT(    0 <  aSize_byte    );
    ASSERT( NULL != aFreeCallback );

    ASSERT(    0 == aThis->mAddress_DA );
    ASSERT( NULL == aThis->mAddress_MA );
    ASSERT( NULL == aThis->mPageTable  );

    // nvidia_p2p_get_pages ==> Pages_Releases  See NvBuffer_Unmap
    lRet = nvidia_p2p_get_pages( 0, 0, aAddress_DA, aSize_byte, & aThis->mPageTable, aFreeCallback, aData );
    if ( 0 != lRet )
    {
        printk( KERN_ERR "%s - nvidia_p2p_get_pages( , , %llx, %u bytes, , , 0x%p ) failed - %d\n", __FUNCTION__, aAddress_DA, aSize_byte, aData, lRet );
        return ( - __LINE__ );
    }

    ASSERT( NULL != aThis->mPageTable );

    aThis->mAddress_DA = aAddress_DA;

    Validate( aThis );

    // iorempa_nocache ==> iounmap  See NvBuffer_Unmap
    aThis->mAddress_MA = ioremap_nocache( aThis->mPageTable->pages[ 0 ]->physical_address, aSize_byte );
    if ( NULL == aThis->mAddress_MA )
    {
        printk( KERN_ERR "%s - ioremap_nocache( 0x%llx, %u bytes ) failed\n", __FUNCTION__, aThis->mPageTable->pages[ 0 ]->physical_address, aSize_byte );
        NvBuffer_Unmap( aThis, aPciDev, true );
        return ( - __LINE__ );
    }

    return 0;
}

// aThis   [---;RW-]
// aPciDev [---;RW-]
//
// NbBuffer_Map ==> NvBuffer_Unmap
void NvBuffer_Unmap( NvBuffer * aThis, struct pci_dev * aPciDev, bool aPutPage )
{
    // printk( KERN_DEBUG "%s( 0x%p, 0x%p )\n", __FUNCTION__, aThis, aPciDev );

    ASSERT( NULL != aThis   );
    ASSERT( NULL != aPciDev );

    if ( NULL != aThis->mPageTable )
    {
        int lRet;

        ASSERT( 0 != aThis->mAddress_DA );

        if ( NULL != aThis->mAddress_MA )
        {
            // ioremap_nocache ==> iounmap  See NvBuffer_Map
            iounmap( aThis->mAddress_MA );

            aThis->mAddress_MA = NULL;
        }

        if ( aPutPage )
        {
            // printk( KERN_DEBUG "%s - Calling nvidia_p2p_put_pages( , , 0x%llx, 0x%p )\n", __FUNCTION__, aThis->mAddress_DA, aThis->mPageTable );

            // nvidia_p2p_get_pages ==> nvidia_p2p_put_pages  See NvBuffer_Map
            lRet = nvidia_p2p_put_pages( 0, 0, aThis->mAddress_DA, aThis->mPageTable );
            if ( 0 != lRet )
            {
                printk( KERN_ERR "%s - nvidia_p2p_put_pages( , , 0x%llx, 0x%p ) failed - %d\n", __FUNCTION__, aThis->mAddress_DA, aThis->mPageTable, lRet );
            }
        }
        else
        {
            // printk( KERN_DEBUG "%s - Calling nvidia_p2p_free_page_table( 0x%p )\n", __FUNCTION__, aThis->mPageTable );

            // nvidia_p2p_get_pages ==> nvidia_p2p_free_page_table  See NvBuffer_Map
            lRet = nvidia_p2p_free_page_table( aThis->mPageTable );
            if ( 0 != lRet )
            {
                printk( KERN_ERR "%s - nvidia_p2p_free_page_table( 0x%p ) failed - %d\n", __FUNCTION__, aThis->mPageTable, lRet );
            }
        }

        aThis->mAddress_DA = 0;
        aThis->mPageTable = NULL;
    }
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

void Validate( NvBuffer * aThis )
{
    unsigned int i;

    // printk( KERN_DEBUG "%s( 0x%p )\n", __FUNCTION__, aThis );

    ASSERT( NULL != aThis );

    ASSERT( NULL != aThis->mPageTable );

    ASSERT( NVIDIA_P2P_PAGE_SIZE_64KB == aThis->mPageTable->page_size  );
    ASSERT( NULL                      != aThis->mPageTable->pages      );
    ASSERT( NULL                      != aThis->mPageTable->pages[ 0 ] );

    for ( i = 1; i < aThis->mPageTable->entries; i ++ )
    {
        ASSERT( NULL != aThis->mPageTable->pages[ i ] );

        ASSERT( aThis->mPageTable->pages[ i - 1 ]->physical_address + 0x10000 == aThis->mPageTable->pages[ i ]->physical_address );
    }
}
