
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Kernel.h (RT)

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

/// \cond en
/// \brief  Declare a packet processing function
/// \param  F  The function name
/// \endcond
/// \cond fr
/// \brief  D&eacute;clare une fonction de traitement de paquet
/// \param  F  Le nom de la fonction
/// \endcond
#define OPEN_NET_FUNCTION_DECLARE(F)                                                        \
    OPEN_NET_DEVICE void F( OPEN_NET_GLOBAL OpenNet_BufferHeader * aBufferHeader )

/// \cond en
/// \brief  Begining of a packet processing function
/// \endcond
/// \cond fr
/// \brief  D&eacute;but d'une fonction de traitement de paquet
/// \endcond
#define OPEN_NET_FUNCTION_BEGIN                                                                                                                     \
    OPEN_NET_GLOBAL_MEMORY_FENCE;                                                                                                                   \
    if ( 0 == ( OPEN_NET_BUFFER_PROCESSED & aBufferHeader->mEvents ) )                                                                              \
    {                                                                                                                                               \
        OPEN_NET_GLOBAL unsigned char      * lBase       = (OPEN_NET_GLOBAL unsigned char      *)( aBufferHeader );                                 \
        OPEN_NET_GLOBAL OpenNet_PacketInfo * lPacketInfo = (OPEN_NET_GLOBAL OpenNet_PacketInfo *)( lBase + aBufferHeader->mPacketInfoOffset_byte ); \
        lPacketInfo += OPEN_NET_PACKET_INDEX;                                                                                                       \
        if ( 0 == lPacketInfo->mSendTo )                                                                                                            \
        {

/// \cond en
/// \brief  End of a packet processing function
/// \endcond
/// \cond fr
/// \brief  Fin d'une fonction de traitement de paquet
/// \endcond
#define OPEN_NET_FUNCTION_END(E)                                      \
        }                                                             \
        OPEN_NET_GLOBAL_MEMORY_FENCE;                                 \
        if ( 0 == OPEN_NET_PACKET_INDEX )                             \
        {                                                             \
            aBufferHeader->mEvents = (E) | OPEN_NET_BUFFER_PROCESSED; \
        }                                                             \
        OPEN_NET_GLOBAL_MEMORY_FENCE;                                 \
    }

// ===== Kernel =============================================================

/// \cond en
/// \brief  Declare a packet processing kernel
/// \endcond
/// \cond fr
/// \brief  D&eacute;clare un kernel de traitement de paquet
/// \endcond
#define OPEN_NET_KERNEL_DECLARE                                                                  \
    OPEN_NET_KERNEL void Filter( OPEN_NET_GLOBAL OpenNet_BufferHeader * aBufferHeader )

/// \cond en
/// \brief  Begining of a packet processing kernel
/// \endcond
/// \cond fr
/// \brief  D&eacute;but d'un kernel de traitement de paquet
/// \endcond
#define OPEN_NET_KERNEL_BEGIN                                                                                                                       \
    OPEN_NET_GLOBAL_MEMORY_FENCE;                                                                                                                   \
    if ( 0 == ( OPEN_NET_BUFFER_PROCESSED & aBufferHeader->mEvents ) )                                                                              \
    {                                                                                                                                               \
        OPEN_NET_GLOBAL unsigned char      * lBase       = (OPEN_NET_GLOBAL unsigned char      *)( aBufferHeader );                                 \
        OPEN_NET_GLOBAL OpenNet_PacketInfo * lPacketInfo = (OPEN_NET_GLOBAL OpenNet_PacketInfo *)( lBase + aBufferHeader->mPacketInfoOffset_byte ); \
        lPacketInfo += OPEN_NET_PACKET_INDEX;                                                                                                       \
        if ( 0 == lPacketInfo->mSendTo )                                                                                                            \
        {

/// \cond en
/// \brief  End of a packet processing kernel
/// \endcond
/// \cond fr
/// \brief  Fin d'un kernel de traitement de paquet
/// \endcond
#define OPEN_NET_KERNEL_END(E)                                        \
        }                                                             \
        OPEN_NET_GLOBAL_MEMORY_FENCE;                                 \
        if ( 0 == OPEN_NET_PACKET_INDEX )                             \
        {                                                             \
            aBufferHeader->mEvents = (E) | OPEN_NET_BUFFER_PROCESSED; \
        }                                                             \
        OPEN_NET_GLOBAL_MEMORY_FENCE;                                 \
    }
