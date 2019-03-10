
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Adapter_WDF.h
/// \brief      OpenNetK::Adapter_WDF (DDK, Windows)

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
    /// \brief  The Adapter_WDF class (DDK, Windows only)
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         method
    /// \endcond
    /// \cond fr
    /// \brief  La classe Adapter_WDF (DDK, Windows seulement)
    /// \note   Classe noyau - Pas de constructeur, pas de destructor, pas
    ///         de m&eacute;thodes virtuelle
    /// \endcond
    class Adapter_WDF
    {

    public:

        /// \cond en
        /// \brief  Initialize the instance.
        /// \param  aAdapter       The Adapter
        /// \param  aDevice        The WDFDEVICE
        /// \param  aHardware_WDF  The Hardware_WDF
        /// \endcond
        /// \cond fr
        /// \brief  Initialiser l'instance
        /// \param  aAdapter       L'Adapter
        /// \param  aDevice        Le WDFDEVICE
        /// \param  aHardware_WDF  Le Hardware_WDF
        /// \endcond
        void Init(Adapter * aAdapter, WDFDEVICE aDevice, Hardware_WDF * aHardware_WDF);

        /// \cond en
        /// \brief  Cleanup file
        /// \param  aFileObject  The WDFFILEOBJECT instance
        /// \endcond
        /// \cond fr
        /// \brief  Nettoyer un fichier
        /// \param  aFileObject  L'instance de WDFFILEOBJECT
        /// \endcond
        void FileCleanup(WDFFILEOBJECT aFileObject);

        /// \cond en
        /// \brief  Process an IoCtl request
        /// \param  aRequest       The request
        /// \param  aOutSize_byte  The maximum output data size
        /// \param  aInSize_byte   The input data size
        /// \param  aCode          The IoCtl request code
        /// \endcond
        /// \cond fr
        /// \brief  Traite une commande IoCtl
        /// \param  aRequest       La requete
        /// \param  aOutSize_byte  La taille maximale des donn&eacute;es de
        ///                        sortie
        /// \param  aInSize_byte   La taille des donn&eacute;es
        ///                        d'entr&eacute;e
        /// \param  aCode          Le code de la commande IoCtl
        /// \endcond
        void IoDeviceControl(WDFREQUEST aRequest, size_t aOutSize_byte, size_t aInSize_byte, ULONG aCode);

        /// \cond en
        /// \brief  Process the request in the caller context.
        /// \param  aRequest  The request
        /// \endcond
        /// \cond fr
        /// \brief  Traiter la requ&ecirc;te dans le context de l'appelant
        /// \param  aRequest  La requ&ecirc;te
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
