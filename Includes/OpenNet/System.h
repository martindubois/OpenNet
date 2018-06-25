
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/System.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNet ===================================================
#include <OpenNet/OpenNet.h>
#include <OpenNet/Status.h>

namespace OpenNet
{

    class Adapter  ;
    class Processor;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class define the system level interface.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe definit l'interface au niveau systeme.
    /// \endcond
    class System
    {

    public:

        /// \cond en
        /// \brief  Use this flag when adapters are physically connected to
        ///         each other. If force the system to send packet to unblock
        ///         reception operations.
        /// \endcond
        /// \cond fr
        /// \brief  Utiliser ce drapeau quand des adaptateur sont
        ///         physiquement connectes entre eux. Il force le system a
        ///         envoyer des paquets pour debloquer d'eventuelles
        ///         operations de reception.
        /// \endcond
        static OPEN_NET_PUBLIC const unsigned int STOP_FLAG_LOOPBACK;

        /// \cond en
        /// \brief   This static methode create an instance of the System
        ///          class.
        /// \return  This static methode return the new instance address.
        /// \endcond
        /// \cond fr
        /// \brief   Cette methode statique cree une instance de la classe
        ///          System.
        /// \return  Cette methode statique retourne l'adresse de la nouvelle
        ///          instance.
        /// \endcond
        static OPEN_NET_PUBLIC System * Create();

        /// \cond en
        /// \brief   Retrieve the system identifier
        /// \return  This method return the system identifier.
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir l'identificateur de system.
        /// \return  Cette methode retourne l'identificateur de system.
        /// \endcond
        virtual unsigned int GetSystemId() const = 0;

        /// \cond en
        /// \brief   Set the maximum packet size
        /// \param   aSize_byte  The maximum packet size
        /// \endcond
        /// \cond fr
        /// \brief   Changer la taille maximal des paquets
        /// \param   aSize_byte  La taille maximal des paquets
        /// \endcond
        /// \retval  STATUS_PACKET_TOO_LARGE
        /// \retval  STATUS_PACKET_TOO_SMALL
        virtual Status SetPacketSize(unsigned int aSize_byte) = 0;

        /// \cond en
        /// \brief   This methode delete the instance.
        /// \endcond
        /// \cond fr
        /// \brief   Cette methode detruit l'instance.
        /// \endcond
        virtual void Delete();

        /// \cond en
        /// \brief   Connect an Adapter to the System
        /// \param   aAdapter  The Adapter
        /// \endcond
        /// \cond fr
        /// \brief   Connecte un Adapter au System
        /// \param   aAdapter  L'Adapter
        /// \endcond
        /// \retval  STATUS_OK
        virtual Status Adapter_Connect(Adapter * aAdapter) = 0;

        /// \cond en
        /// \return  This methode return the number of adapters.
        /// \endcond
        /// \cond fr
        /// \return  Cette methode retourne le nombre d'adaptateurs.
        /// \endcond
        virtual unsigned int Adapter_GetCount() const = 0;

        /// \cond en
        /// \param   aIndex  The index of the adapter to get
        /// \return  This methode return the adapter.
        /// \endcond
        /// \cond fr
        /// \param   aIndex  L'index de l'adaptateur a retourner
        /// \return  Cette methode retourne l'adaptateurs.
        /// \endcond
        virtual Adapter * Adapter_Get( unsigned int aIndex ) = 0;

        /// \cond en
        /// \brief  Display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche
        /// \retval aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        virtual Status Display(FILE * aOut) = 0;

        /// \cond en
        /// \return  This methode return the number of processors.
        /// \endcond
        /// \cond fr
        /// \return  Cette methode retourne le nombre de processeurs.
        /// \endcond
        virtual unsigned int Processor_GetCount() const = 0;

        /// \cond en
        /// \param   aIndex  The index of the processor to get
        /// \return  This methode return the processor.
        /// \endcond
        /// \cond fr
        /// \param   aIndex  L'index du processeur a retourner
        /// \return  Cette methode retourne le processeur.
        /// \endcond
        virtual Processor * Processor_Get( unsigned int aIndex ) = 0;

        /// \cond en
        /// \brief  Start
        /// \endcond
        /// \cond fr
        /// \brief  Demarrer
        /// \endcond
        /// \retval STATUS_OK
        virtual Status Start() = 0;

        /// \cond en
        /// \brief  Stop
        /// \param  aFlags  STOP_FLAG_LOOPBACK
        /// \endcond
        /// \cond fr
        /// \brief  Arreter
        /// \param  aFlags  STOP_FLAG_LOOPBACK
        /// \endcond
        /// \retval STATUS_OK
        virtual Status Stop(unsigned int aFlags) = 0;

    protected:

        System();

        virtual ~System();

    private:

        System(const System &);

        const System & operator = (const System &);

    };

}
