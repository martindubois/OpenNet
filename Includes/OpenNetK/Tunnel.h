
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Tunnel.h
/// \brief      OpenNet_Tunnel_PacketHeader

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

#define OPEN_NET_SYNC_CHECK_VALUE (0xbef9)

// Data type
/////////////////////////////////////////////////////////////////////////////

/// \cond en
/// \brief  This structure is used as packet header into a tunnel.
/// \todo   Document member of OpenNet_Tunnel_PacketHeader
/// \endcond
/// \cond fr
/// \brief  Cette structure est utilis&eacute;e pour passer les informations
///         au sujet d'un paquet dans un tunnel.
/// \todo   Documenter les membres de OpenNet_Tunne_PacketHeader
/// \endcond
typedef struct
{
    uint16_t mSyncCheck;

    uint16_t mPacketSize_byte;

    uint32_t mReserved0;
}
OpenNet_Tunnel_PacketHeader;
