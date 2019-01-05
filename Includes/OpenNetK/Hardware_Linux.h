
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All right reserved.
/// \file       Includes/OpenNetK/Hardware_Linux.h
/// \brief      OpenNetK::Hardware_Linux

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/SpinLock_Linux.h>

namespace OpenNetK
{
    class Hardware;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class connect the Linux device with the Hardware class
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         methode
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe connecte un device Linux avec la class Hardware
    /// \note   Classe noyau - Pas de constructeur, pas de destructeur, pas
    ///         de methode virtuelle
    /// \endcond
    class Hardware_Linux
    {

    public:

        /// \cond en
        /// \brief  Initialize the instance.
        /// \param  aHardware  The Hardware
        /// \retval     0  OK
        /// \retval Other  Error
        /// \endcond
        /// \cond fr
        /// \brief  Initialise l'instance
        /// \param  aHardware  L'Hardware
        /// \retval     0  OK
        /// \retval Other  Erreur
        /// \endcond
        int Init( Hardware * aHardware );

        /// \cond en
        /// \brief  Enter the D0 state
        /// \retval     0  OK
        /// \retval Other  Error
        /// \endcond
        /// \cond fr
        /// \brief  Entrer dans l'etat D0
        /// \retval     0  OK
        /// \retval Other  Erreur
        /// \endcond
        int D0Entry();

        /// \cond en
        /// \brief  Exit the D0 state
        /// \retval     0  OK
        /// \retval Other  Error
        /// \endcond
        /// \cond fr
        /// \brief  Sortir de l'etat D0
        /// \retval     0  OK
        /// \retval Other  Erreur
        /// \endcond
        int D0Exit();

        /// \cond en
        /// \brief  Prepare the hardware
        /// \retval     0  OK
        /// \retval Other  Error
        /// \endcond
        /// \cond fr
        /// \brief  Prepare le meteriel
        /// \retval     0  OK
        /// \retval Other  Erreur
        /// \endcond
        int PrepareHardware();

        /// \cond en
        /// \brief  Release the hardware
        /// \retval     0  OK
        /// \retval Other  Error
        /// \endcond
        /// \cond fr
        /// \brief  Relacher le materiel
        /// \retval     0  OK
        /// \retval Other  Erreur
        /// \endcond
        int ReleaseHardware();

    //internal:

        int  Interrupt_Disable ();
        int  Interrupt_Enable  ();
        bool Interrupt_Isr     (unsigned int aMessageId);
        void Interrupt_Dpc     ();

        void Tick();

        void TrigProcess2();

    private:

        void InitTimer();

        int PrepareInterrupt();
        int PrepareMemory   ();

        Hardware * mHardware;

        unsigned int mIntCount ;

        unsigned int mMemCount;
        unsigned int mMemSize_byte[6];
        void       * mMemVirtual  [6];

        // ===== Zone 0 =====================================================
        SpinLock_Linux mZone0;

    };

}
