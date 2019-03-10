
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All right reserved.
/// \file       Includes/OpenNetK/Hardware_Linux.h
/// \brief      OpenNetK::Hardware_Linux (DDK, Linux)

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/SpinLock.h>

namespace OpenNetK
{
    class Hardware;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  The Hardware_Linux class (DDK, Linux only)
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         methode
    /// \endcond
    /// \cond fr
    /// \brief  La classe Hardware_Linux (DDK, Linux seulement)
    /// \note   Classe noyau - Pas de constructeur, pas de destructeur, pas
    ///         de methode virtuelle
    /// \endcond
    class Hardware_Linux
    {

    public:

        /// \cond en
        /// \brief  Initialize the instance.
        /// \param  aHardware  The Hardware
        /// \param  aOSDep     The operating system dependent function table
        /// \param  aZone0     The spinlock to use
        /// \endcond
        /// \cond fr
        /// \brief  Initialise l'instance
        /// \param  aHardware  L'Hardware
        /// \param  aOSDep     La table de fonctions dependente du systeme
        ///                    d'exploitation
        /// \param  aZone0     Le spinlock a utiliser
        /// \endcond
        void Init( Hardware * aHardware, OpenNetK_OSDep * aOSDep, void * aZone0 );

    private:

        // ===== Zone 0 =====================================================
        SpinLock mZone0;

    };

}
