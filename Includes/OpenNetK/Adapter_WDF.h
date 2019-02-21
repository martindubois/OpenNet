
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Adapter_WDF.h
/// \brief      OpenNetK::Adapter_WDF

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNetK ==================================================
#include <OpenNetK/OSDep.h>
#include <OpenNetK/SpinLock.h>

namespace OpenNetK
{
    class Adapter     ;
    class Hardware_WDF;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class maintains information about an adapter on the
    ///         OpenNet internal network.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe maintien les information concernant un
    ///         adaptateur sur le reseau interne OpenNet.
    /// \endcond
    class Adapter_WDF
    {

    public:

        /// \cond en
        /// \brief  Initialize the instance.
        /// \param  aAdapter      [-K-;RW-] The Adapter
        /// \param  aDevice       [-K-;RW-] The WDFDEVICE
        /// \param  aHardware_WDF [-K-;RW-] The Hardware_WDF
        /// \endcond
        /// \cond fr
        /// \brief  Initialise l'instance
        /// \param  aAdapter      [-K-;RW-] L'Adapter
        /// \param  aDevice       [-K-;RW-] Le WDFDEVICE
        /// \param  aHardware_WDF [-K-;RW-] Le Hardware_WDF
        /// \endcond
        void Init(Adapter * aAdapter, WDFDEVICE aDevice, Hardware_WDF * aHardware_WDF);

        /// \cond en
        /// \brief  Cleanup file
        /// \param  aFileObject [---;RW-] The WDFFILEOBJECT instance
        /// \endcond
        /// \cond fr
        /// \brief  Nettoyer un fichier
        /// \param  aFileObject [---;RW-] L'instance de WDFFILEOBJECT
        /// \endcond
        void FileCleanup(WDFFILEOBJECT aFileObject);

        /// \cond en
        /// \brief  Process an IoCtl request
        /// \param  aRequest [---;RW-] The request
        /// \param  aOutSize_byte      The maximum output data size
        /// \param  aInSize_byte       The input data size
        /// \param  aCode              The IoCtl request code
        /// \endcond
        /// \cond fr
        /// \brief  Traite une commande IoCtl
        /// \param  aRequest [---;RW-] La requete
        /// \param  aOutSize_byte      La taille maximal des donnes de sortie
        /// \param  aInSize_byte       La taille des donnees d'entree
        /// \param  aCode              Le code de la commande IoCtl
        /// \endcond
        void IoDeviceControl(WDFREQUEST aRequest, size_t aOutSize_byte, size_t aInSize_byte, ULONG aCode);

        /// \cond en
        /// \brief  Process the request in the caller context.
        /// \param  aRequest [---;RW-] The request
        /// \endcond
        /// \cond fr
        /// \brief  Traite la requete dans le context de l'appelant
        /// \param  aRequest [---;RW-] La requete
        /// \endcond
        void IoInCallerContext(WDFREQUEST aRequest);

    // internal:

        void SharedMemory_Release();

    private:

        NTSTATUS Connect(void * aIn);

        void ProcessIoCtlResult(int aIoCtlResult);

        NTSTATUS SharedMemory_ProbeAndLock();
        NTSTATUS SharedMemory_Translate   (void ** aSharedMemory);

        NTSTATUS ResultToStatus(WDFREQUEST aRequest, int aIoCtlResult);

        Adapter      * mAdapter         ;
        WDFDEVICE      mDevice          ;
        Hardware_WDF * mHardware_WDF    ;
        OpenNetK_OSDep mOSDep           ;
        MDL          * mSharedMemory_MDL;

        // ===== Zone 0 =====================================================
        SpinLock mZone0;

    };

}
