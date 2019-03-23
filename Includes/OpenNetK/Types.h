
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Types.h
/// \brief      OpenNet_BufferHeader, OpenNet_PacketInfo (RT)

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

#define OPEN_NET_BUFFER_PROCESSED (0x00000001)
#define OPEN_NET_BUFFER_EVENT     (0x00000002)
#define OPEN_NET_BUFFER_RESERVED  (0xfffffffc)

#define OPEN_NET_PACKET_PROCESSED (0x80000000)
#define OPEN_NET_PACKET_EVENT     (0x40000000)

// Data type
/////////////////////////////////////////////////////////////////////////////

/// \cond en
/// \brief  This structure is used to pass the information about a buffer.
/// \todo   Document member of OpenNet_BufferHeader
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilis&eacute;e pour passer les informations
///         au sujet d'un espace m&eacute;moire.
/// \todo   Documenter les membres de OpenNet_BufferHeader
/// \endcond
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
/// \todo   Document member of OpenNet_PacketInfo
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilis&eacute;e pour passer les informations
///         au sujet d'un paquet.
/// \todo   Documenter les membres de OpenNet_PacketInfo
/// \endcond
typedef struct
{
    volatile uint32_t mSendTo;

    uint32_t mOffset_byte;

    volatile uint32_t mSize_byte;

    uint8_t mReserved2[4];
}
OpenNet_PacketInfo;
