
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Hardware_WDF.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/SpinLock_WDF.h>

namespace OpenNetK
{
    class Hardware;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class connect the WDF device with the Hardware class
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         methode
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe connecte un device WDF avec la class Hardware
    /// \note   Classe noyau - Pas de constructeur, pas de destructeur, pas
    ///         de methode virtuelle
    /// \endcond
    class Hardware_WDF
    {

    public:

        /// \cond en
        /// \brief  Initialize the instance.
        /// \param  aDevice   [-K-;RW-] The WDFDEVICE
        /// \param  aHardware [-K-;RW-] The Hardware
        /// \endcond
        /// \cond fr
        /// \brief  Initialise l'instance
        /// \param  aDevice   [-K-;RW-] Le WDFDEVICE
        /// \param  aHardware [-K-;RW-] L'Hardware
        /// \endcond
        /// \retval STATUS_SUCCESS
        NTSTATUS Init(WDFDEVICE aDevice, Hardware * aHardware);

        /// \cond en
        /// \brief  Enter the D0 state
        /// \param  aPreviousState  Previous state
        /// \endcond
        /// \cond fr
        /// \brief  Entrer dans l'etat D0
        /// \param  aPreviousState  Etat precedent
        /// \endcond
        /// \retval STATUS_SUCCESS
        NTSTATUS D0Entry(WDF_POWER_DEVICE_STATE aPreviousState);

        /// \cond en
        /// \brief  Exit the D0 state
        /// \param  aTargetState  Target state
        /// \endcond
        /// \cond fr
        /// \brief  Sortir de l'etat D0
        /// \param  aTargetState  Etat cible
        /// \endcond
        /// \retval STATUS_SUCCESS
        NTSTATUS D0Exit(WDF_POWER_DEVICE_STATE aTargetState);

        /// \cond en
        /// \brief  Prepare the hardware
        /// \param  aRaw        [---;R--] The raw ressources
        /// \param  aTranslated [---;RW-] The translated ressources
        /// \endcond
        /// \cond fr
        /// \brief  Prepare le meteriel
        /// \param  aRaw        [---;R--] Les ressources "raw"
        /// \param  aTranslated [---;RW-] Les ressources "translated"
        /// \endcond
        /// \retval STATUS_SUCCESS
        NTSTATUS PrepareHardware(WDFCMRESLIST aRaw, WDFCMRESLIST aTranslated);

        /// \cond en
        /// \brief  Release the hardware
        /// \param  aTranslated [---;RW-] The translated ressources
        /// \endcond
        /// \cond fr
        /// \brief  Relacher le materiel
        /// \param  aTranslated [---;RW-] Les ressources "translated"
        /// \endcond
        /// \retval STATUS_SUCCESS
        NTSTATUS ReleaseHardware(WDFCMRESLIST aTranslated);

    //internal:

        NTSTATUS Interrupt_Disable ();
        NTSTATUS Interrupt_Enable  ();
        BOOLEAN  Interrupt_Isr     (ULONG aMessageId);
        void     Interrupt_Dpc     ();

        void Tick();

        void TrigProcess2();

    private:

        void InitTimer();

        NTSTATUS PrepareInterrupt(CM_PARTIAL_RESOURCE_DESCRIPTOR * aTranslated, CM_PARTIAL_RESOURCE_DESCRIPTOR * aRaw);
        NTSTATUS PrepareMemory   (CM_PARTIAL_RESOURCE_DESCRIPTOR * aTranslated);

        WDFCOMMONBUFFER mCommonBuffer;
        WDFDEVICE       mDevice      ;
        WDFDMAENABLER   mDmaEnabler  ;
        Hardware      * mHardware    ;

        unsigned int mIntCount ;
        WDFINTERRUPT mInterrupt;

        unsigned int    mMemCount;
        unsigned int    mMemSize_byte[6];
        volatile void * mMemVirtual  [6];

        WDFTIMER mTimer;

        // ===== Zone 0 =====================================================
        SpinLock_WDF mZone0;

    };

}
