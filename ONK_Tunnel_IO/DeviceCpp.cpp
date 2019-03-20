
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/DeviceCpp.cpp
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
#include "VirtualHardware.h"

extern "C"
{
    #include "DeviceCpp.h"
}

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Adapter        mAdapter       ;
    VirtualHardware          mHardware      ;
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

// aThis         [---;-W-]
// aOSDep        [-K-;R--] The OS dependent functions
// aAdapterLock  [-K-;RW-] The spinlock for the Adapter instance
// aHardwareLock [-K-]RW-] The spinlock for the Hardware instance
void DeviceCpp_Init( void * aThis, OpenNetK_OSDep * aOSDep, void * aAdapterLock, void * aHardwareLock )
{
    // printk( KERN_DEBUG "%s( , , ,  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis         );
    ASSERT( NULL != aOSDep        );
    ASSERT( NULL != aAdapterLock  );
    ASSERT( NULL != aHardwareLock );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    memset( lThis, 0, sizeof( DeviceCppContext ) );

    new ( & lThis->mHardware ) VirtualHardware();

    lThis->mHardware_Linux.Init( & lThis->mHardware, aOSDep, aHardwareLock );
    lThis->mAdapter_Linux .Init( & lThis->mAdapter , aOSDep, aAdapterLock  );

    lThis->mAdapter .SetHardware( & lThis->mHardware  );
    lThis->mAdapter .SetOSDep   ( aOSDep              );
    lThis->mHardware.SetAdapter ( & lThis->mAdapter   );
}

// aThis [---;RW-]
//
// DeviceCpp_D0_Entry ==> DeviceCpp_D0_Exit
void DeviceCpp_D0_Entry( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    // Hardware::D0_Entry ==> Hardware::D0_Exit  See DeviceCpp_D0_Exit
    lThis->mHardware.D0_Entry();
}

// aThis [---;RW-]
//
// DeviceCpp_D0_Entry ==> DeviceCpp_D0_Exit
void DeviceCpp_D0_Exit( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    // Hardware::D0_Entry ==> Hardware::D0_Exit  See DeviceCpp_D0_Entry
    lThis->mHardware.D0_Exit();
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

// aThis      [---;RW-]
// aBuffer_UA [---;-W-]
// aSize_byte
//
// Return  This function returns the number of byte read
unsigned int DeviceCpp_Read( void * aThis, void * aBuffer_UA, unsigned int aSize_byte )
{
    printk( KERN_DEBUG "%s( 0x%p, 0x%px, %u bytes )\n", __FUNCTION__, aThis, aBuffer_UA, aSize_byte );

    ASSERT( NULL != aThis      );
    ASSERT( NULL != aBuffer_UA );
    ASSERT(    0 <  aSize_byte );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    return lThis->mHardware.Read( aBuffer_UA, aSize_byte );
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

// aThis [---;RW-]
void DeviceCpp_Tick( void * aThis )
{
    // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

    ASSERT( NULL != aThis );

    DeviceCppContext * lThis = reinterpret_cast< DeviceCppContext * >( aThis );

    lThis->mHardware.Tick();
}
