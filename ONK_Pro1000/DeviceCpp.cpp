
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/DeviceCpp.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Adapter.h>
#include <OpenNetK/Adapter_Linux.h>
#include <OpenNetK/Hardware_Linux.h>

// ===== ONK_Pro1000 ========================================================
#include "Pro1000.h"

extern "C"
{
    #include "DeviceCpp.h"
}

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Adapter        mAdapter       ;
    Pro1000                  mHardware      ;
    OpenNetK::Adapter_Linux  mAdapter_Linux ;
    OpenNetK::Hardware_Linux mHardware_Linux;
}
DeviceCppContext;

// Functions
/////////////////////////////////////////////////////////////////////////////

// Return  This function return the context size in bytes.
unsigned int DeviceCpp_GetContextSize()
{
    printk( KERN_DEBUG "%s()\n", __FUNCTION__ );

    return sizeof( DeviceCppContext );
}

// aThis [---;-W-]
void DeviceCpp_Init( void * aThis )
{
    printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    memset( lThis, 0, sizeof( DeviceCppContext ) );

    new ( & lThis->mHardware ) Pro1000();

    lThis->mHardware_Linux.Init( & lThis->mHardware );
    lThis->mAdapter_Linux .Init( & lThis->mAdapter, & lThis->mHardware_Linux );

    lThis->mAdapter .SetHardware( & lThis->mHardware );
    lThis->mHardware.SetAdapter ( & lThis->mAdapter  );
}

// aThis [---;RW-]
void DeviceCpp_Uninit( void * aThis )
{
    printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );
}

// aThis [---;RW-]
//
// Return  This function return the size of the needed common buffer in
//         bytes.
unsigned int DeviceCpp_CommonBuffer_GetSize( void * aThis )
{
    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    return lThis->mHardware.GetCommonBufferSize();
}

// aThis [---;RW-]
// aPhysical       The physical address of the common buffer
// aVirtual        The virtual address of the common buffer
void DeviceCpp_CommonBuffer_Set( void * aThis, unsigned long aPhysical, void * aVirtual )
{
    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    lThis->mHardware.SetCommonBuffer( aPhysical, aVirtual );
}

// aThis [---;RW-]
// aMessageId
//
// Return  See PIR_...
ProcessIrqResult DeviceCpp_Interrupt_Process( void * aThis, unsigned int aMessageId )
{
    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    bool lNeedMoreProcessing;

    if ( lThis->mHardware.Interrupt_Process( aMessageId, & lNeedMoreProcessing ) )
    {
        return ( lNeedMoreProcessing ? PIR_TO_PROCESS : PIR_PROCESSED );
    }

    return PIR_IGNORED;
}

// aThis [---;RW-]
void DeviceCpp_Interrupt_Process2( void * aThis )
{
    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    lThis->mHardware.Interrupt_Process2();
}

// aThis  [---;RW-]
// aCode            The command code
// aInOut [--O;RW-] The input and output buffer
// aInSize_byte     The maximum size of the input buffer
//
// Return
//    0  OK
//  < 0  Error
int DeviceCpp_IoCtl( void * aThis, unsigned int aCode, void * aInOut, unsigned int aInSize_byte )
{
    printk( KERN_DEBUG "%s( , 0x%08x, , %u bytes )\n", __FUNCTION__, aCode, aInSize_byte );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    return lThis->mAdapter_Linux.IoDeviceControl( aInOut, aCode, aInSize_byte );
}

// aCode
// aInSizeMax_byte [---;-W-] The function puts the maximum size of the input
//                           buffer there.
// aInSizeMin_byte [---;-W-] The function puts the minimum size of the input
//                           buffer there
// aOutSize_byte   [---;-W-] The function puts the size of the output buffer
//                           there.
//
// Return
//    0  OK
//  < 0  Error
int DeviceCpp_IoCtl_GetInfo( unsigned int aCode, unsigned int * aInSizeMax_byte, unsigned int * aInSizeMin_byte, unsigned int * aOutSize_byte )
{
    printk( KERN_DEBUG "%s( 0x%08x, , , )\n", __FUNCTION__, aCode );

    ASSERT( NULL != aInSizeMax_byte );
    ASSERT( NULL != aInSizeMin_byte );
    ASSERT( NULL != aOutSize_byte   );

    OpenNetK::Adapter::IoCtl_Info lIoCtlInfo;

    if ( ! OpenNetK::Adapter::IoCtl_GetInfo( aCode, & lIoCtlInfo ) )
    {
        printk( KERN_ERR "%s - OpenNetK::Adapter::IoCtl_GetInfo( 0x%08x,  ) failed\n", __FUNCTION__, aCode );
        return ( - __LINE__ );
    }

    ( * aInSizeMax_byte ) = lIoCtlInfo.mIn_MaxSize_byte ;
    ( * aInSizeMin_byte ) = lIoCtlInfo.mIn_MinSize_byte ;
    ( * aOutSize_byte   ) = lIoCtlInfo.mOut_MinSize_byte;

    return 0;
}

// aThis    [---;RW-]
// aIndex
// aVirtual [-K-;RW-]
// aSize_byte
//
// Return
//    0  OK
//  < 0  Error
int DeviceCpp_SetMemory( void * aThis, unsigned int aIndex, void * aVirtual, unsigned int aSize_byte )
{
    printk( KERN_DEBUG "%s( , %u, , %u bytes )\n", __FUNCTION__, aIndex, aSize_byte );

    ASSERT( NULL != aThis      );
    ASSERT( NULL != aVirtual   );
    ASSERT(    0 <  aSize_byte );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    return ( lThis->mHardware.SetMemory( aIndex, aVirtual, aSize_byte ) ? 0 : ( - __LINE__ ) );
}
