
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Kernel.h

#pragma once

// Standard int types
/////////////////////////////////////////////////////////////////////////////

#define uint32_t unsigned int
#define uint8_t  unsigned char

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/Types.h>

// Macros
/////////////////////////////////////////////////////////////////////////////

#define OPEN_NET_FUNCTION_DECLARE(F)                        \
    void F( __global OpenNet_BufferHeader * aBufferHeader )

#define OPEN_NET_FUNCTION_BEGIN                                                                                                       \
    if ( OPEN_NET_BUFFER_STATE_PX_RUNNING == aBufferHeader->mBufferState )                                                            \
    {                                                                                                                                 \
        __global unsigned char      * lBase       = (__global unsigned char      *)( aBufferHeader );                                 \
        __global OpenNet_PacketInfo * lPacketInfo = (__global OpenNet_PacketInfo *)( lBase + aBufferHeader->mPacketInfoOffset_byte ); \
        lPacketInfo += get_local_id( 0 );

#define OPEN_NET_FUNCTION_END                                                 \
        if ( 0 == get_local_id( 0 ) )                                         \
        {                                                                     \
            aBufferHeader->mBufferState = OPEN_NET_BUFFER_STATE_PX_COMPLETED; \
        }                                                                     \
    }

#define OPEN_NET_KERNEL_DECLARE                                           \
    __kernel void Filter( __global OpenNet_BufferHeader * aBufferHeader )

#define OPEN_NET_KERNEL_BEGIN                                                                                                     \
    __global unsigned char      * lBase       = (__global unsigned char      *)( aBufferHeader );                                 \
    __global OpenNet_PacketInfo * lPacketInfo = (__global OpenNet_PacketInfo *)( lBase + aBufferHeader->mPacketInfoOffset_byte ); \
    lPacketInfo += get_global_id( 0 );

#define OPEN_NET_KERNEL_END                                               \
    if ( 0 == get_global_id( 0 ) )                                        \
    {                                                                     \
        aBufferHeader->mBufferState = OPEN_NET_BUFFER_STATE_PX_COMPLETED; \
    }
