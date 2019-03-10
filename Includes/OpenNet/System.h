
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/System.h
/// \brief   OpenNet::System

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
        /// \todo   Document members of OpenNet::System::Config
        /// \endcond
        /// \cond fr
        /// \brief  Configuration du system
        /// \todo   Documenter les membres de OpenNet::System::Config
        /// \endcond
        typedef struct
        {
            unsigned int mPacketSize_byte;

            unsigned char mReserved0[60];
        }
        Config;

        /// \cond en
        /// \brief  System information
        /// \brief  Document members of OpenNet::System::Info
        /// \endcond
        /// \cond fr
        /// \brief  Information au sujet du system
        /// \todo   Documenter les membres de OpenNet::System::Info
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
        /// \brief  Utiliser ce drapeau quand des adaptateurs sont
        ///         physiquement connect&eacute;s entre eux. Il force le
        ///         syst&egrave;me &agrave; envoyer des paquets pour
        ///         d&eacute;bloquer d'eventuelles op&eacute;rations de
        ///         r&eacute;ception.
        /// \endcond
        /// \sa     Start
        static OPEN_NET_PUBLIC const unsigned int START_FLAG_LOOPBACK;

        /// \cond en
        /// \brief   This static methode create an instance of the System
        ///          class.
        /// \return  This static methode return the new instance address.
        /// \endcond
        /// \cond fr
        /// \brief   Cette m&eacute;thode statique cr&eacute;e une instance
        ///          de la classe System.
        /// \return  Cette m&eacute;thode statique retourne l'adresse de la
        ///          nouvelle instance.
        /// \endcond
        /// \sa      Delete
        static OPEN_NET_PUBLIC System * Create();

        /// \cond en
        /// \brief   This static methode display the system configuration.
        /// \param   aConfig  The configuration
        /// \param   aOut     The output stream
        /// \endcond
        /// \cond fr
        /// \brief   Cette m&eacute;thode statique affiche la configuration
        ///          d'un syst&egrave;me.
        /// \param   aConfig  La configuration
        /// \param   aOut     Le fichier de sortie
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_INVALID_REFERENCE
        /// \retval  STATUS_NOT_ALLOWER_NULL_ARGUMENT
        static OPEN_NET_PUBLIC Status Display(const Config & aConfig, FILE * aOut);

        /// \cond en
        /// \brief   This static methode display the system information.
        /// \param   aInfo  The information
        /// \param   aOut   The output stream
        /// \endcond
        /// \cond fr
        /// \brief   Cette m&eacute;thode statique affiche l'information au
        ///          sujet d'un syst&egrave;me.
        /// \param   aInfo  L'information
        /// \param   aOut   Le fichier de sortie
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_INVALID_REFERENCE
        /// \retval  STATUS_NOT_ALLOWER_NULL_ARGUMENT
        static OPEN_NET_PUBLIC Status Display(const Info & aInfo, FILE * aOut);

        /// \cond en
        /// \brief   Retrieve the configuration of the system
        /// \param   aOut  The output space
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir la configuration du syst&egrave;me
        /// \param   aOut  L'espace m&eacute;moire de sortie
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_NOT_ALLOWER_NULL_ARGUMENT
        /// \sa      SetConfig
        virtual Status GetConfig(Config * aOut) const = 0;

        /// \cond en
        /// \brief   Retrieve the information about the system
        /// \param   aOut  The output space
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir l'information au sujet du syst&egrave;me
        /// \param   aOut  L'espace m&eacute;moire de sortie
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status GetInfo(Info * aOut) const = 0;

        /// \cond en
        /// \brief   Modify the configuration of the system
        /// \param   aConfig  The configuration
        /// \endcond
        /// \cond fr
        /// \brief   Changer la configuration du syst&egrave;me
        /// \param   aConfig  La configuration
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_INVALID_REFERENCE
        /// \sa      GetConfig
        virtual Status SetConfig(const Config & aConfig) = 0;

        /// \cond en
        /// \brief   This methode delete the instance.
        /// \endcond
        /// \cond fr
        /// \brief   Cette m&eacute;thode detruit l'instance.
        /// \endcond
        /// \sa      Create
        virtual void Delete();

        /// \cond en
        /// \brief   Connect an Adapter to the System
        /// \param   aAdapter  The Adapter
        /// \endcond
        /// \cond fr
        /// \brief   Connecter un Adapter au System
        /// \param   aAdapter  L'Adapter
        /// \endcond
        /// \retval  STATUS_OK
        virtual Status Adapter_Connect(Adapter * aAdapter) = 0;

        /// \cond en
        /// \return  This methode return the number of adapters.
        /// \endcond
        /// \cond fr
        /// \return  Cette m&eacute;thode retourne le nombre d'adaptateurs.
        /// \endcond
        virtual unsigned int Adapter_GetCount() const = 0;

        /// \cond en
        /// \param   aIndex  The index of the adapter to get
        /// \retval  NULL    Not found
        /// \retval  Other   The address of the Adapter instance
        /// \endcond
        /// \cond fr
        /// \param   aIndex  L'index de l'adaptateur &agrave; retourner
        /// \retval  NULL    Introuvable
        /// \retval  Other   L'adresse de l'instance d'Adapter
        /// \endcond
        virtual Adapter * Adapter_Get( unsigned int aIndex ) = 0;

        /// \cond en
        /// \param   aAddress   The Ethernet address to look for
        /// \param   aMask      The bit set to 1 indicate the bit that must
        ///                     match the address.
        /// \param   aMaskDiff  The bit set to 1 indicate the part of address
        ///                     that must be different.
        /// \retval  NULL       Not found
        /// \retval  Other      The address of the Adapter instance
        /// \endcond
        /// \cond fr
        /// \param   aAddress   L'adresse Ethernet &agrave; rechercher
        /// \param   aMask      Les bit &agrave; 1 correspondent au bits qui
        ///                     doivent correspondres &agrave; l'adresse.
        /// \param   aMaskDiff  Les bits &agrave; 1 indiquent la partie de
        ///                     l'adresse qui doit etre differente
        /// \retval  NULL       Introuvable
        /// \retval  Other      L'adresse de l'instance d'Adapter
        /// \endcond
        virtual Adapter * Adapter_Get(const unsigned char * aAddress, const unsigned char * aMask, const unsigned char * aMaskDiff) = 0;

        /// \cond en
        /// \brief  Display
        /// \param  aOut  The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Afficher
        /// \retval aOut  Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        virtual Status Display(FILE * aOut) = 0;

        /// \cond en
        /// \brief   Retrieve a Kernel
        /// \param   aIndex  The kernel index
        /// \retval  NULL   Invalid index
        /// \retval  Other  The Kernel instance
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir un Kernel
        /// \param   aIndex  L'index du Kernel
        /// \retval  NULL  Index invalide
        /// \retval  Others  L'adresse de l'instance de Kernel
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
        /// \param   aIndex  The index of the Processor to get
        /// \return  This method returns the processor.
        /// \endcond
        /// \cond fr
        /// \param   aIndex  L'index du processeur &agrave; retourner
        /// \return  Cette m&eacute;thode retourne le processeur.
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
        /// \sa     Stop, START_FLAG_LOOPBACK
        virtual Status Start(unsigned int aFlags) = 0;

        /// \cond en
        /// \brief  Stop
        /// \endcond
        /// \cond fr
        /// \brief  Arreter
        /// \endcond
        /// \retval STATUS_OK
        /// \sa     Start
        virtual Status Stop() = 0;

    protected:

        System();

        virtual ~System();

    private:

        System(const System &);

        const System & operator = (const System &);

    };

}
