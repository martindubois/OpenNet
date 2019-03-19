
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Tunnel.h
/// \brief      OpenNetK::Tunnel_PacketHeader

#pragma once

namespace OpenNetK
{

    // Constants
    /////////////////////////////////////////////////////////////////////////

    static const uint16_t SYNC_CHECK_VALUE = 0xbef9;

    // Data type
    /////////////////////////////////////////////////////////////////////////

    typedef struct
    {
        uint16_t mSyncCheck;

        uint16_t mPacketSize_byte;

        uint32_t mReserved0;
    }
    Tunnel_PacketHeader;

}
