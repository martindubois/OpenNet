
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Types.h
/// \brief      OpenNet_BufferHeader, OpenNet_PacketInfo

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

#define OPEN_NET_BUFFER_EVENT_PROCESSED (0x00000001)
#define OPEN_NET_BUFFER_EVENT_RESERVED  (0xfffffffe)

#define OPEN_NET_PACKET_PROCESSED (0x80000000)

// Data type
/////////////////////////////////////////////////////////////////////////////

/// \cond en
/// \brief  This structure is used to pass the information about a buffer.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour passer les informations au sujet
///         d'un espace memoire.
/// \endcond
/// \todo   Document members
typedef struct
{
    uint32_t mPacketQty            ;
    uint32_t mPacketInfoOffset_byte;
    uint32_t mPacketSize_byte      ;

    volatile uint32_t mEvents;

    uint8_t mReserved0[48];
}
OpenNet_BufferHeader;

/// \cond en
/// \brief  This structure is used to pass the information about a packet.
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilise pour passer les informations au sujet
///         d'un paquet.
/// \endcond
/// \todo   Document members
typedef struct
{
    volatile uint32_t mSendTo;

    uint8_t mReserved0[4];

    uint32_t mOffset_byte;

    uint8_t mReserved1[4];

    volatile uint32_t mSize_byte;

    uint8_t mReserved2[12];
}
OpenNet_PacketInfo;
