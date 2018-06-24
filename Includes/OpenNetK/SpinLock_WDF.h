
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/SpinLock_WDF.h

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
    class SpinLock_WDF : public SpinLock
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
        SpinLock_WDF(WDFDEVICE aDevice);

        // ===== SpinLock ===================================================
        virtual void Lock  ();
        virtual void Unlock();

    private:

        WDFSPINLOCK mSpinLock;

    };

}
