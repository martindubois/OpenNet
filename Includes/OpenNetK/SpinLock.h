
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/SpinLock.h

#pragma once

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
    class SpinLock
    {

    public:

        /// \cond en
        /// \brief  new operator without allocation
        /// \param  aSize_byte         The size
        /// \param  aAddress [---;RW-] The address
        /// \endcond
        /// \cond fr
        /// \brief  Operateur new sans allocation
        /// \param  aSize_byte         La taille
        /// \param  aAddress [---;RW-] L'adresse
        /// \endcond
        /// \note   Level = Thread, Thread = Initialisation
        void * operator new(size_t aSize_byte, void * aAddress);

        /// \cond en
        /// \brief  Lock
        /// \endcond
        /// \cond fr
        /// \brief  Verouiller
        /// \endcond
        virtual void Lock() = 0;

        /// \cond en
        /// \brief  Unlock
        /// \endcond
        /// \cond fr
        /// \brief  Deverouiller
        /// \endcond
        virtual void Unlock() = 0;

    };
}
