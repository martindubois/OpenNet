
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2019 KMS. All rights reserved
/// \file       Includes/OpenNetK/PacketGenetator_Types.h
/// \brief      OpenNetK::PacketGenerator_Config

#pragma once

namespace OpenNetK
{

    /// \cond en
    /// \brief  This structure is used to pass the configuration.
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilise pour passer la configuration.
    /// \endcond
    /// \toto   Document the members
    typedef struct
    {
        uint32_t mAllowedIndexRepeat;
        uint32_t mIndexOffset_byte  ;
        uint32_t mPacketPer100ms    ;
        uint32_t mPacketSize_byte   ;

        unsigned int mReserved2[48];

        uint8_t  mPacket[16 * 1024];
    }
    PacketGenerator_Config;

}
