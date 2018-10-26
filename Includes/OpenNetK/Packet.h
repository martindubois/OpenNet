
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Packet.h
/// \brief   OpenNetK::Packet

#pragma once

namespace OpenNetK
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class cache packet information
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         method
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe cache les information d'un paquet
    /// \note   Classe noyau - Pas de constructeur, pas de destructor, pas de
    ///         method virtuel
    /// \endcond
    class Packet
    {

    public:

        uint32_t mOffset_byte;
        uint32_t mSendTo     ;
        uint32_t mState      ;

    };

}
