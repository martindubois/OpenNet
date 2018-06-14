
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Kernel.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/Types.h>

// Macros
/////////////////////////////////////////////////////////////////////////////

#define OPEN_NET_KERNEL_DECLARE                                           \
    __kernel void Filter( __global OpenNet_BufferHeader * aBufferHeader )

#define OPEN_NET_KERNEL_BEGIN          \
    cl_uint8           * lBase       = reinterpret_cast< cl_uint8           * >( aBufferHeader ); \
    OpenNet_PacketInfo * lPacketInfo = reinterpret_cast< OpenNet_PacketInfo * >( lBase + aBufferHeader->mPacketInfoOffset_byte ); \
                                       \
    lPacketInfo += get_global_id( 0 );

#define OPEN_NET_KERNEL_END                                          \
    if ( 0 == get_global_id( 0 ) )                                   \
    {                                                                \
        aBufferHeader->mBufferState = OPEN_NET_BUFFER_STATE_TO_SEND; \
    }
