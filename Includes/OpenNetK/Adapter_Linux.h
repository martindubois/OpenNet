
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Adapter_Linux.h
/// \brief      OpenNetK::Adapter_Linux

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNetK ==================================================
#include <OpenNetK/SpinLock_Linux.h>

namespace OpenNetK
{
    class Adapter       ;
    class Hardware_Linux;

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
    class Adapter_Linux
    {

    public:

        /// \cond en
        /// \brief  Initialize the instance.
        /// \param  aAdapter         The Adapter
        /// \param  aHardware_Linux  The Hardware_Linux
        /// \endcond
        /// \cond fr
        /// \brief  Initialise l'instance
        /// \param  aAdapter         L'Adapter
        /// \param  aHardware_Linux  Le Hardware_Linux
        /// \endcond
        void Init( Adapter * aAdapter, Hardware_Linux * aHardware_Linux );

        /// \cond en
        /// \brief  Cleanup file
        /// \endcond
        /// \cond fr
        /// \brief  Nettoyer un fichier
        /// \endcond
        void FileCleanup();

        /// \cond en
        /// \brief  Process an IoCtl request
        /// \param  aInOut      The data
        /// \param  aSize_byte  The maximum data size
        /// \param  aCode       The IoCtl request code
        /// \retval     0  OK
        /// \retval Other  Error
        /// \endcond
        /// \cond fr
        /// \brief  Traite une commande IoCtl
        /// \param  aInOut      La requete
        /// \param  aSize_byte  La taille maximal des donnes
        /// \param  aCode       Le code de la commande IoCtl
        /// \retval     0  OK
        /// \retval Other  Erreur
        /// \endcond
        int IoDeviceControl( void * aInOut, size_t aSize_byte, unsigned int aCode );

        /// \cond en
        /// \brief  Process the request in the caller context.
        /// \endcond
        /// \cond fr
        /// \brief  Traite la requete dans le context de l'appelant
        /// \endcond
        void IoInCallerContext();

    private:

        int  Connect   (void * aIn);
        void Disconnect();

        void Event_Release  ();
        int  Event_Translate(uint64_t * aEvent);

        void ProcessIoCtlResult(int aIoCtlResult);

        int  SharedMemory_ProbeAndLock();
        void SharedMemory_Release     ();
        int  SharedMemory_Translate   (void ** aSharedMemory);

        int ResultToStatus(int aIoCtlResult);

        Adapter        * mAdapter       ;
        Hardware_Linux * mHardware_Linux;

        // ===== Zone 0 =====================================================
        SpinLock_Linux mZone0;

    };

}
