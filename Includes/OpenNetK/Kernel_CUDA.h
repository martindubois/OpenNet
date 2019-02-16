
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Kernel_CUDA.h
/// \todo       Document macros

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

#define OPEN_NET_FUNCTION_DECLARE(F)                          \
    __device__ void F( OpenNet_BufferHeader * aBufferHeader )

#define OPEN_NET_FUNCTION_BEGIN                                                                                       \
    if ( OPEN_NET_BUFFER_STATE_PX_RUNNING == aBufferHeader->mBufferState )                                            \
    {                                                                                                                 \
        unsigned char      * lBase       = ( unsigned char      * )( aBufferHeader );                                 \
        OpenNet_PacketInfo * lPacketInfo = ( OpenNet_PacketInfo * )( lBase + aBufferHeader->mPacketInfoOffset_byte ); \
        lPacketInfo += threadIdx.x;                                                                                   \
        if (0 == lPacketInfo->mSendTo)                                                                                \
        {

#define OPEN_NET_FUNCTION_END                                                 \
        }                                                                     \
        if ( 0 == threadIdx.x )                                               \
        {                                                                     \
            aBufferHeader->mBufferState = OPEN_NET_BUFFER_STATE_PX_COMPLETED; \
        }                                                                     \
    }

#define OPEN_NET_KERNEL_DECLARE                                               \
    extern "C" __global__ void Filter( OpenNet_BufferHeader * aBufferHeader )

#define OPEN_NET_KERNEL_BEGIN                                                                                         \
    if ( OPEN_NET_BUFFER_STATE_PX_RUNNING == aBufferHeader->mBufferState )                                            \
    {                                                                                                                 \
        unsigned char      * lBase       = ( unsigned char      * )( aBufferHeader );                                 \
        OpenNet_PacketInfo * lPacketInfo = ( OpenNet_PacketInfo * )( lBase + aBufferHeader->mPacketInfoOffset_byte ); \
        lPacketInfo += threadIdx.x;                                                                                   \
        if (0 == lPacketInfo->mSendTo)                                                                                \
        {

#define OPEN_NET_KERNEL_END                                                   \
        }                                                                     \
        if ( 0 == threadIdx.x )                                               \
        {                                                                     \
            aBufferHeader->mBufferState = OPEN_NET_BUFFER_STATE_PX_COMPLETED; \
        }                                                                     \
    }
