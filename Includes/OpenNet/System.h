
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/System.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/OpenNet.h>
#include <OpenNet/Status.h>

namespace OpenNet
{

    class Adapter  ;
    class Kernel   ;
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
        /// \brief  System configuration
        /// \endcond
        /// \cond fr
        /// \brief  Configuration du system
        /// \endcond
        typedef struct
        {
            unsigned int mPacketSize_byte;

            unsigned char mReserved0[60];
        }
        Config;

        /// \cond en
        /// \brief  System information
        /// \endcond
        /// \cond fr
        /// \brief  Information au sujet du system
        /// \endcond
        typedef struct
        {
            unsigned int mSystemId;

            unsigned char mReserved0[60];
        }
        Info;

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
        static OPEN_NET_PUBLIC const unsigned int START_FLAG_LOOPBACK;

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
        /// \brief   This static methode display the system configuration.
        /// \param   aConfig [---;R--] The configuration
        /// \param   aOut    [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief   Cette methode statique affiche la configuration d'un
        ///          system.
        /// \param   aConfig [---;R--] La configuration
        /// \param   aOut    [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_INVALID_REFERENCE
        /// \retval  STATUS_NOT_ALLOWER_NULL_ARGUMENT
        static OPEN_NET_PUBLIC Status Display(const Config & aConfig, FILE * aOut);

        /// \cond en
        /// \brief   This static methode display the system information.
        /// \param   aInfo [---;R--] The information
        /// \param   aOut  [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief   Cette methode statique affiche l'information au sujet
        ///          d'un system.
        /// \param   aInfo [---;R--] L'information
        /// \param   aOut  [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_INVALID_REFERENCE
        /// \retval  STATUS_NOT_ALLOWER_NULL_ARGUMENT
        static OPEN_NET_PUBLIC Status Display(const Info & aInfo, FILE * aOut);

        /// \cond en
        /// \brief   Retrieve the configuration of the system
        /// \param   aOut [---;-W-] The output space
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir la configuration du system
        /// \param   aOut [---;-W-] L'espace memoire de sortie
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status GetConfig(Config * aOut) const = 0;

        /// \cond en
        /// \brief   Retrieve the information about the system
        /// \param   aOut [---;-W-] The output space
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir l'information au sujet du system
        /// \param   aOut [---;-W-] L'espace memoire de sortie
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status GetInfo(Info * aOut) const = 0;

        /// \cond en
        /// \brief   Modify the configuration of the system
        /// \param   aConfig [---;-W-] The configuration
        /// \endcond
        /// \cond fr
        /// \brief   Changer la configuration du system
        /// \param   aConfig [---;-W-] La configuration
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_INVALID_REFERENCE
        virtual Status SetConfig(const Config & aConfig) = 0;

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
        /// \brief   Retrieve a Kernel
        /// \param   aIndex  The kernel index
        /// \retval  NULL  Invalid index
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir un Kernel
        /// \param   aIndex  L'index du Kernel
        /// \retval  NULL  Index invalide
        /// \endcond
        virtual OpenNet::Kernel * Kernel_Get(unsigned int aIndex) = 0;

        /// \cond en
        /// \brief   Retrieve the number of Kernel
        /// \return  The number of Kernel
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir le nombre de Kernel
        /// \return  Le nombre de Kernel
        /// \endcond
        virtual unsigned int Kernel_GetCount() const = 0;

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
        /// \param  aFlags  START_FLAG_LOOPBACK
        /// \endcond
        /// \cond fr
        /// \brief  Demarrer
        /// \param  aFlags  START_FLAG_LOOPBACK
        /// \endcond
        /// \retval STATUS_OK
        virtual Status Start(unsigned int aFlags) = 0;

        /// \cond en
        /// \brief  Stop
        /// \endcond
        /// \cond fr
        /// \brief  Arreter
        /// \endcond
        /// \retval STATUS_OK
        virtual Status Stop() = 0;

    protected:

        System();

        virtual ~System();

    private:

        System(const System &);

        const System & operator = (const System &);

    };

}
