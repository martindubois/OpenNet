
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/DeviceCpp.cpp
//
// This file if the glue between the linux C code compiled using kbuild and
// the cpp code of the ONK_Lib.
//
// The code in this file do not depend on any linux header. It can be
// compiled without using kbuild and can run with any kernel version.

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Adapter.h>
#include <OpenNetK/Adapter_Linux.h>
#include <OpenNetK/Hardware_Linux.h>

// ===== ONK_Pro1000 ========================================================
#include "Intel_82576.h"
#include "Intel_82599.h"

extern "C"
{
    #include "DeviceCpp.h"
}

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Adapter        mAdapter       ;
    Intel                  * mHardware      ;
    OpenNetK::Adapter_Linux  mAdapter_Linux ;
    OpenNetK::Hardware_Linux mHardware_Linux;

    union
    {
        Intel_82576              mIntel_82576;
        Intel_82599::Intel_82599 mIntel_82599;
    };
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

// aThis         [---;-W-]
// aOSDep        [-K-;R--] The OS dependent functions
// aAdapterLock  [-K-;RW-] The spinlock for the Adapter instance
// aHardwareLock [-K-]RW-] The spinlock for the Hardware instance
// aDeviceId               The PCI device id
void DeviceCpp_Init( void * aThis, OpenNetK_OSDep * aOSDep, void * aAdapterLock, void * aHardwareLock, unsigned short aDeviceId )
{
    // printk( KERN_DEBUG "%s( , , ,  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis         );
    ASSERT( NULL != aOSDep        );
    ASSERT( NULL != aAdapterLock  );
    ASSERT( NULL != aHardwareLock );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    memset( lThis, 0, sizeof( DeviceCppContext ) );

    switch ( aDeviceId )
    {
        case 0x10c9 : lThis->mHardware = new ( & lThis->mIntel_82576 )              Intel_82576(); break;
        case 0x10fb : lThis->mHardware = new ( & lThis->mIntel_82599 ) Intel_82599::Intel_82599(); break;

        default : ASSERT( false );
    }

    ASSERT( NULL != lThis->mHardware );

    lThis->mHardware_Linux.Init(   lThis->mHardware, aOSDep, aHardwareLock );
    lThis->mAdapter_Linux .Init( & lThis->mAdapter, aOSDep, aAdapterLock );

    lThis->mAdapter  .SetHardware( lThis->mHardware  );
    lThis->mAdapter  .SetOSDep   ( aOSDep            );
    lThis->mHardware->SetAdapter ( & lThis->mAdapter );
}

// aThis [---;RW-]
void DeviceCpp_Uninit( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );
}

// aThis [---;RW-]
//
// Return  This function return the size of the needed common buffer in
//         bytes.
unsigned int DeviceCpp_CommonBuffer_GetSize( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    return lThis->mHardware->GetCommonBufferSize();
}

// aThis [---;RW-]
// aPhysical       The physical address of the common buffer
// aVirtual        The virtual address of the common buffer
void DeviceCpp_CommonBuffer_Set( void * aThis, uint64_t aPhysical, void * aVirtual )
{
    // printk( KERN_DEBUG "%s( , 0x%llx,  )\n", __FUNCTION__, aPhysical );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    ASSERT( NULL != lThis->mHardware );

    lThis->mHardware->SetCommonBuffer( aPhysical, aVirtual );
}

// aThis [---;RW-]
//
// DeviceCpp_D0_Entry ==> DeviceCpp_D0_Exit
void DeviceCpp_D0_Entry( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    ASSERT( NULL != lThis->mHardware );

    // Hardware::D0_Entry ==> Hardware::D0_Exit  See DeviceCpp_D0_Exit
    lThis->mHardware->D0_Entry();
}

// aThis [---;RW-]
//
// DeviceCpp_D0_Entry ==> DeviceCpp_D0_Exit
void DeviceCpp_D0_Exit( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    ASSERT( NULL != lThis->mHardware );

    // Hardware::D0_Entry ==> Hardware::D0_Exit  See DeviceCpp_D0_Entry
    lThis->mHardware->D0_Exit();
}

// aThis [---;RW-]
//
// DeviceCpp_Interrupt_Enable ==> DeviceCpp_D0_Exit
void DeviceCpp_Interrupt_Enable( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    ASSERT( NULL != lThis->mHardware );

    // Hardware::Interrupt_Enabled ==> Hardware::D0_Exit  See DeviceCpp_D0_Exit
    lThis->mHardware->Interrupt_Enable();
}

// aThis [---;RW-]
// aMessageId
//
// Return  See PIR_...

// CRITICAL PATH  Interrupt
//                1 / hardware interrupt
ProcessIrqResult DeviceCpp_Interrupt_Process( void * aThis, unsigned int aMessageId )
{
    // printk( KERN_DEBUG "%s( , %u )\n", __FUNCTION__, aMessageId );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    ASSERT( NULL != lThis->mHardware );

    bool lNeedMoreProcessing;

    if ( lThis->mHardware->Interrupt_Process( aMessageId, & lNeedMoreProcessing ) )
    {
        return ( lNeedMoreProcessing ? PIR_TO_PROCESS : PIR_PROCESSED );
    }

    return PIR_IGNORED;
}

// aThis               [---;RW-]
// aNeedMoreProcessing [---;-W-]

// CRITICAL PATH  Interrupt
//                1 / hardware interrupt + 1 / tick
void DeviceCpp_Interrupt_Process2( void * aThis, bool * aNeedMoreProcessing )
{
    // printk( KERN_DEBUG "%s( ,  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );
    ASSERT( NULL != aNeedMoreProcessing );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    ASSERT( NULL != lThis->mHardware );

    lThis->mHardware->Interrupt_Process2( aNeedMoreProcessing );
}

// aThis [---;RW-]
void DeviceCpp_Interrupt_Process3( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    ASSERT( NULL != lThis->mHardware );

    lThis->mHardware->Interrupt_Process3();
}

// aThis       [---;RW-]
// aFileObject [-K-;---] The file object used to send the IoCtl request
// aCode                 The command code
// aIn         [--O;R--] The input buffer
// aInSize_byte          The maximum size of the input buffer
// aOut        [--O;-W-] The output buffer
// aOutSize_byte         The size of the output buffer
//
// Return
//    0  OK
//  < 0  Error
int DeviceCpp_IoCtl( void * aThis, void * aFileObject, unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte )
{
    // printk( KERN_DEBUG "%s( , 0x%08x, , %u bytes, , %u bytes )\n", __FUNCTION__, aCode, aInSize_byte, aOutSize_byte );

    ASSERT( NULL != aThis       );
    ASSERT( NULL != aFileObject );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    return lThis->mAdapter.IoCtl( aFileObject, aCode, aIn, aInSize_byte, aOut, aOutSize_byte );
}

// aCode
// aInfo [---;-W-] The function puts the info here
//
// Return
//    0  OK
//  < 0  Error
int DeviceCpp_IoCtl_GetInfo( unsigned int aCode, OpenNetK_IoCtl_Info * aInfo )
{
    // printk( KERN_DEBUG "%s( 0x%08x,  )\n", __FUNCTION__, aCode );

    ASSERT( NULL != aInfo );

    if ( ! OpenNetK::Adapter::IoCtl_GetInfo( aCode, aInfo ) )
    {
        printk( KERN_ERR "%s - OpenNetK::Adapter::IoCtl_GetInfo( 0x%08x,  ) failed\n", __FUNCTION__, aCode );
        return ( - __LINE__ );
    }

    return 0;
}

// aThis       [---;RW-]
// aFileObject [---;---]
void DeviceCpp_Release( void * aThis, void * aFileObject )
{
    // printk( KERN_DEBUG "%s( ,  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis       );
    ASSERT( NULL != aFileObject );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    lThis->mAdapter.FileCleanup( aFileObject );
}

// aThis    [---;RW-]
void DeviceCpp_ResetMemory( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    ASSERT( NULL != lThis->mHardware );

    lThis->mHardware->ResetMemory();
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
    // printk( KERN_DEBUG "%s( , %u, , %u bytes )\n", __FUNCTION__, aIndex, aSize_byte );

    ASSERT( NULL != aThis      );
    ASSERT( NULL != aVirtual   );
    ASSERT(    0 <  aSize_byte );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    ASSERT( NULL != lThis->mHardware );

    return ( lThis->mHardware->SetMemory( aIndex, aVirtual, aSize_byte ) ? 0 : ( - __LINE__ ) );
}

// aThis [---;RW-]
void DeviceCpp_Tick( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    ASSERT( NULL != lThis->mHardware );

    lThis->mHardware->Tick();
}
