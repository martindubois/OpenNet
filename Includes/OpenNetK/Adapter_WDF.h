
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Adapter_WDF.h

#pragma once

namespace OpenNetK
{
    class Adapter;

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
        /// \param  aAdapter [-K-;RW-] The Adapter
        /// \param  aDevice  [-K-;RW-] The WDFDEVICE
        /// \endcond
        /// \cond fr
        /// \brief  Initialise l'instance
        /// \param  aAdapter [-K-;RW-] L'Adapter
        /// \param  aDevice  [-K-;RW-] Le WDFDEVICE
        /// \endcond
        void Init(Adapter * aAdapter, WDFDEVICE aDevice);

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
        /// \param  aCode  Le code de la commande IoCtl
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

    private:

        Adapter * mAdapter;
        WDFDEVICE mDevice;

    };

}