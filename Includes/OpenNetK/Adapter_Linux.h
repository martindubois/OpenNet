
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Adapter_Linux.h
/// \brief      OpenNetK::Adapter_Linux (DDK, Linux)

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/SpinLock.h>

namespace OpenNetK
{
    class Adapter;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  The Adapter_Linux class (DDK, Linux only)
    /// \note   This class is part of the Driver Development Kit (DDK).
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         method
    /// \endcond
    /// \cond fr
    /// \brief  La classe Adapter_Linux (DDK, Linux seulement)
    /// \note   Cette classe fait partie de l'ensemble de developpement de
    ///         pilotes (DDK).
    /// \note   Classe noyau - Pas de constructeur, pas de destructor, pas
    ///         de m&eacute;thodes virtuelle
    /// \endcond
    class Adapter_Linux
    {

    public:

        /// \cond en
        /// \brief  Initialize the instance.
        /// \param  aAdapter  The Adapter
        /// \param  aOSDep    The operating system dependent function table
        /// \param  aZone0    The spinlock to use
        /// \endcond
        /// \cond fr
        /// \brief  Initialise l'instance
        /// \param  aAdapter  L'Adapter
        /// \param  aOSDep    La table de fonctions sp&eacute;cifique au
        ///                   syst&egrave;me d'exploitation
        /// \param  aZone0    Le spinlock a utiliser
        /// \endcond
        void Init( Adapter * aAdapter, OpenNetK_OSDep * aOSDep, void * aZone0 );

    private:

        // ===== Zone 0 =====================================================
        SpinLock mZone0;

    };

}
