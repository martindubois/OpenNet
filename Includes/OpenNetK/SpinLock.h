
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
    /// \note   This class is part of the Driver Development Kit (DDK).
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         method
    /// \endcond
    /// \cond fr
    /// \brief  Interface d'un spinlock
    /// \note   Cette classe fait partie de l'ensemble de developpement de
    ///         pilotes (DDK).
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
        /// \brief  Initialiser l'instance de verrou d&eacute;pendant du
        ///         syst&egrave;me d'exploitation
        /// \param  aLock  L'instance
        /// \endcond
        void SetLock(void * aLock);

        /// \cond en
        /// \brief  Set the OS dependant function table
        /// \param  aOSDep  The function table
        /// \endcond
        /// \cond fr
        /// \brief  Initialiser la table de fonctions d&eacute;pendantes du
        ///         syst&egrave;me d'exploitation
        /// \param  aOSDep  La table de fonctions
        /// \endcond
        void SetOSDep(OpenNetK_OSDep * aOSDep);

        /// \cond en
        /// \brief  Lock
        /// \endcond
        /// \cond fr
        /// \brief  Verouiller
        /// \endcond
        /// \sa     LockFromThread, Unlock
        void Lock();

        /// \cond en
        /// \brief  Lock
        /// \return Value to pass to UnlockFromThread
        /// \endcond
        /// \cond fr
        /// \brief  Verouiller
        /// \return La valeur &agrave; passer &agrave; UnlockFromThread
        /// \endcond
        /// \sa     Lock, UnlockFromThread
        uint32_t LockFromThread();

        /// \cond en
        /// \brief  Unlock
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;verouiller
        /// \endcond
        /// \sa     Lock, UnlockFromThread
        void Unlock();

        /// \cond en
        /// \brief  Unlock
        /// \param  aFlags  The value returned by LockFromThread
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;verouiller
        /// \param  aFlags  La valeur retourn&eacute; par LockFromThread
        /// \endcond
        /// \sa     LockFromThread, Unlock
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
