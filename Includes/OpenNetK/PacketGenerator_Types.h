
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2019 KMS. All rights reserved
/// \file       Includes/OpenNetK/PacketGenerator_Types.h
/// \brief      OpenNetK::PacketGenerator_Config

#pragma once

namespace OpenNetK
{

    /// \cond en
    /// \brief  This structure is used to pass the configuration.
    /// \note   This data type is part of the Driver Development Kit (DDK).
    /// \todo   Document members of PacketGenerator_Config
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilis&eacute;e pour passer la
    ///         configuration.
    /// \note   Ce type de donn&eacute;e fait partie de l'ensemble de
    ///         developpement de pilotes (DDK).
    /// \todo   Documenter les membres de PacketGenerator_Config
    /// \endcond
    typedef struct
    {
        uint32_t mAllowedIndexRepeat;
        uint32_t mIndexOffset_byte  ;
        uint32_t mPacketPer100ms    ;
        uint32_t mPacketSize_byte   ;

        uint8_t  mReserved2[48];

        uint8_t  mPacket[16 * 1024];
    }
    PacketGenerator_Config;

}
