
// Product / Produit  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All right reserved.
/// \file       Includes/OpenNetK/SpinLock_Linux.h
/// \brief      OpenNetK::SpinLock_Linux

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
extern "C"
{
    #include <OpenNetK/OSDep.h>
}

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
        /// \param  aOSDep
        /// \param  aLock
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur
        /// \param  aOSDep
        /// \param  aLock
        /// \endcond
        SpinLock_Linux( OpenNetK_OSDep * aOSDep, void * aLock );

        // ===== SpinLock ===================================================
        virtual void Lock  ();
        virtual void Unlock();

    private:

        void           * mLock ;
        OpenNetK_OSDep * mOSDep;

    };

}
