
// Product / Produit  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All right reserved.
/// \file       Includes/OpenNetK/SpinLock_Linux.h
/// \brief      OpenNetK::SpinLock_Linux

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/SpinLock.h>

namespace OpenNetK
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  SpinLock interface
    /// \endcond
    /// \cond fr
    /// \brief  Interface d'un spinlock
    /// \endcond
    class SpinLock_Linux : public SpinLock
    {

    public:

        /// \cond en
        /// \brief  Constructor
        /// \param  aDevice [---;RW-] The WDFDEVICE instance
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur
        /// \param  aDevice [---;RW-] L'intance de WDFDEVICE
        /// \endcond
        SpinLock_Linux();

        // ===== SpinLock ===================================================
        virtual void Lock  ();
        virtual void Unlock();

    private:


    };

}
