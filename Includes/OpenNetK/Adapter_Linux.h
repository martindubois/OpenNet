
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Adapter_Linux.h
/// \brief      OpenNetK::Adapter_Linux

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/SpinLock_Linux.h>

namespace OpenNetK
{
    class Adapter;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class maintains information about an adapter on the
    ///         OpenNet internal network.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe maintien les information concernant un
    ///         adaptateur sur le reseau interne OpenNet.
    /// \endcond
    class Adapter_Linux
    {

    public:

        /// \cond en
        /// \brief  Initialize the instance.
        /// \param  aAdapter  The Adapter
        /// \param  aOSDep    The operating system dependent function table
        /// \param  aLock     The spinlock to use
        /// \endcond
        /// \cond fr
        /// \brief  Initialise l'instance
        /// \param  aAdapter  L'Adapter
        /// \param  aOSDep    La table de fonctions specifique au systeme
        ///                   d'exploitation
        /// \param  aLock     Le spinlock a utiliser
        /// \endcond
        void Init( Adapter * aAdapter, OpenNetK_OSDep * aOSDep, void * aZone0 );

    private:

        // ===== Zone 0 =====================================================
        SpinLock_Linux mZone0;

    };

}
