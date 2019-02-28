
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Kernel.h
/// \brief      Defines the macro used to declare main functions and kernels.

#pragma once

// Standard int types
/////////////////////////////////////////////////////////////////////////////

#define uint32_t unsigned int
#define uint8_t  unsigned char

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================

#ifdef _OPEN_NET_CUDA_
    #include <OpenNetK/Kernel_CUDA.h>
#endif

#ifdef _OPEN_NET_OPEN_CL_
    #include <OpenNetK/Kernel_OpenCL.h>
#endif

#include <OpenNetK/Types.h>

// Macros
/////////////////////////////////////////////////////////////////////////////

// ===== Function ===========================================================

#define OPEN_NET_FUNCTION_DECLARE(F)                                                        \
    OPEN_NET_DEVICE void F( OPEN_NET_GLOBAL OpenNet_BufferHeader * aBufferHeader )

#define OPEN_NET_FUNCTION_BEGIN                                                                                                                     \
    OPEN_NET_GLOBAL_MEMORY_FENCE;                                                                                                                   \
    if ( OPEN_NET_BUFFER_STATE_PX_RUNNING == aBufferHeader->mBufferState )                                                                          \
    {                                                                                                                                               \
        OPEN_NET_GLOBAL unsigned char      * lBase       = (OPEN_NET_GLOBAL unsigned char      *)( aBufferHeader );                                 \
        OPEN_NET_GLOBAL OpenNet_PacketInfo * lPacketInfo = (OPEN_NET_GLOBAL OpenNet_PacketInfo *)( lBase + aBufferHeader->mPacketInfoOffset_byte ); \
        lPacketInfo += OPEN_NET_PACKET_INDEX;                                                                                                       \
        if ( 0 == lPacketInfo->mSendTo )                                                                                                            \
        {

#define OPEN_NET_FUNCTION_END                                                 \
        }                                                                     \
        OPEN_NET_GLOBAL_MEMORY_FENCE;                                         \
        if ( 0 == OPEN_NET_PACKET_INDEX )                                     \
        {                                                                     \
            aBufferHeader->mBufferState = OPEN_NET_BUFFER_STATE_PX_COMPLETED; \
        }                                                                     \
        OPEN_NET_GLOBAL_MEMORY_FENCE;                                         \
    }

// ===== Kernel =============================================================

#define OPEN_NET_KERNEL_DECLARE                                                                  \
    OPEN_NET_KERNEL void Filter( OPEN_NET_GLOBAL OpenNet_BufferHeader * aBufferHeader )

#define OPEN_NET_KERNEL_BEGIN                                                                                                                       \
    OPEN_NET_GLOBAL_MEMORY_FENCE;                                                                                                                   \
    if ( OPEN_NET_BUFFER_STATE_PX_RUNNING == aBufferHeader->mBufferState )                                                                          \
    {                                                                                                                                               \
        OPEN_NET_GLOBAL unsigned char      * lBase       = (OPEN_NET_GLOBAL unsigned char      *)( aBufferHeader );                                 \
        OPEN_NET_GLOBAL OpenNet_PacketInfo * lPacketInfo = (OPEN_NET_GLOBAL OpenNet_PacketInfo *)( lBase + aBufferHeader->mPacketInfoOffset_byte ); \
        lPacketInfo += OPEN_NET_PACKET_INDEX;                                                                                                       \
        if ( 0 == lPacketInfo->mSendTo )                                                                                                            \
        {

#define OPEN_NET_KERNEL_END                                                   \
        }                                                                     \
        OPEN_NET_GLOBAL_MEMORY_FENCE;                                         \
        if ( 0 == OPEN_NET_PACKET_INDEX )                                     \
        {                                                                     \
            aBufferHeader->mBufferState = OPEN_NET_BUFFER_STATE_PX_COMPLETED; \
        }                                                                     \
        OPEN_NET_GLOBAL_MEMORY_FENCE;                                         \
    }
