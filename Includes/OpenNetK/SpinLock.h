
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/SpinLock.h
/// \brief      OpenNetK::SpinLock

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/OSDep.h>

namespace OpenNetK
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  SpinLock interface
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         method
    /// \endcond
    /// \cond fr
    /// \brief  Interface d'un spinlock
    /// \note   Classe noyau - Pas de constructeur, pas de destructeur, pas
    ///         de methode virtuelle
    /// \endcond
    class SpinLock
    {

    public:

        /// \cond en
        /// \brief  Set the OS dependant lock instance
        /// \param  aLock  The instance
        /// \endcond
        /// \cond fr
        /// \brief  Initialiser l'instance de verou d&eacute;pendant du
        ///         systeme d'exploitation
        /// \param  aLock  L'instance
        /// \endcond
        void SetLock(void * aLock);

        /// \cond en
        /// \brief  Set the OS dependant function table
        /// \param  aOSDep  The function table
        /// \endcond
        /// \cond fr
        /// \brief  Initialiser la table de fonctions d&eacute;pendantes du
        ///         systeme d'exploitation
        /// \param  aOSDep  La table de fonctions
        /// \endcond
        void SetOSDep(OpenNetK_OSDep * aOSDep);

        /// \cond en
        /// \brief  Lock
        /// \endcond
        /// \cond fr
        /// \brief  Verouiller
        /// \endcond
        void Lock();

        /// \cond en
        /// \brief  Lock
        /// \return Value to pass to UnlockFromThread
        /// \endcond
        /// \cond fr
        /// \brief  Verouiller
        /// \return La valeur &agrave; passer &agrave; UnlockFromThread
        /// \endcond
        uint32_t LockFromThread();

        /// \cond en
        /// \brief  Unlock
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;verouiller
        /// \endcond
        void Unlock();

        /// \cond en
        /// \brief  Unlock
        /// \param  aFlags  The value returned by LockFromThread
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;verouiller
        /// \param  aFlags  La valeur retourn&eacute; par LockFromThread
        /// \endcond
        void UnlockFromThread( uint32_t aFlags );

    private:

        void           * mLock ;
        OpenNetK_OSDep * mOSDep;

    };

    // Public
    /////////////////////////////////////////////////////////////////////////

    inline void SpinLock::Lock()
    {
        mOSDep->LockSpinlock(mLock);
    }

    inline void SpinLock::Unlock()
    {
        mOSDep->UnlockSpinlock(mLock);
    }

}
