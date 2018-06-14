
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Adapter.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Processor.h>
#include <OpenNet/Status.h>
#include <OpenNetK/Interface.h>

namespace OpenNet
{

    class Filter;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class define the adapter level interface.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe definit l'interface au niveau de l'adaptateur.
    /// \endcond
    class Adapter
    {

    public:

        typedef OpenNet_Config Config;
        typedef OpenNet_Info   Info  ;
        typedef OpenNet_State  State ;
        typedef OpenNet_Stats  Stats ;

        /// \cond en
        /// \brief  Display
        /// \param  aIn  [---;R--] The Config
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche  Le Status
        /// \param  aIn  [---;R--] Le Config
        /// \param  aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_INVALID_REFERENCE
        static OPEN_NET_PUBLIC Status Display(const Config & aIn, FILE * aOut);

        /// \cond en
        /// \brief  Display
        /// \param  aIn  [---;R--] The Info
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche  Le Status
        /// \param  aIn  [---;R--] Le Info
        /// \param  aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_INVALID_REFERENCE
        static OPEN_NET_PUBLIC Status Display(const Info & aIn  , FILE * aOut);

        /// \cond en
        /// \brief  Display
        /// \param  aIn  [---;R--] The State
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche  Le Status
        /// \param  aIn  [---;R--] Le State
        /// \param  aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_INVALID_REFERENCE
        static OPEN_NET_PUBLIC Status Display(const State  & aIn, FILE * aOut);

        /// \cond en
        /// \brief  Display
        /// \param  aIn  [---;R--] The Stats
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Afficher
        /// \param  aIn  [---;R--] Le Stats
        /// \param  aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        static OPEN_NET_PUBLIC Status Display(const Stats & aIn , FILE * aOut);

        /// \cond en
        /// \brief  This methode return the adapter numero.
        /// \param  aOut [---;-W-] The methode return the information here.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode retourne le numero de l'adaptateur.
        /// \param aOut [---;-W-] La medhode retourne les informations ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_NOT_CONNECTED
        virtual Status GetAdapterNo(unsigned int * aOut) = 0;

        /// \cond en
        /// \brief  This methode return the configuration of the adapter.
        /// \param  aOut [---;-W-] The methode return the information here.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode retourne la configuration de l'adaptateur.
        /// \param aOut [---;-W-] La medhode retourne les informations ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status GetConfig(Config * aOut) const = 0;

        /// \cond en
        /// \brief  This methode returns the information about the adapter.
        /// \param  aOut [---;-W-] The methode return the information here.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode retourne les informations au sujet de
        ///        l'adaptateur.
        /// \param aOut [---;-W-] La medhode retourne les informations ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status GetInfo(Info * aOut) const = 0;

        /// \cond en
        /// \brief  This methode returns the state of the adapter.
        /// \param  aOut [---;-W-] The methode return the information here.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode retourne l'etat de l'adaptateur.
        /// \param aOut [---;-W-] La medhode retourne les informations ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_EXCEPTION
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status GetState(State * aOut) = 0;

        /// \cond en
        /// \brief  This methode return the statistics of the adapter.
        /// \param  aOut [---;-W-] The methode return the statistics here.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode retourne les statistiques de l'adaptateur.
        /// \param aOut [---;-W-] La medhode retourne les statistiques ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_EXCEPTION
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status GetStats(Stats * aOut) = 0;

        /// \cond en
        /// \brief  This methode reset the input filter.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode reset le filtre d'entre.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_FILTER_NOT_SET
        virtual Status ResetInputFilter() = 0;

        /// \cond en
        /// \brief  This methode reset the processor.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode reset le processeur.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_PROCESSOR_NOT_SET
        virtual Status ResetProcessor() = 0;

        /// \cond en
        /// \brief  This methode resets the statistics of the adapter.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode reset les statistiques de l'adaptateur.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_EXCEPTION
        /// \retval STATUS_IOCTL_ERROR
        virtual Status ResetStats() = 0;

        /// \cond en
        /// \brief  This methode set the configuration.
        /// \param  aConfig [---;RW-] The configuration
        /// \endcond
        /// \cond fr
        /// \brief Cette methode change la configuration.
        /// \param aConfig [---;RW-] La configuration
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status SetConfig(const Config & aConfig) = 0;

        /// \cond en
        /// \brief  This methode set the input filter.
        /// \param  aFilter [---;RW-] The Open CL program.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode affecte le filtre d'entre.
        /// \param aFilter [---;RW-] Le programme Open CL
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_FILTER_ALREADY_SET
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual OpenNet::Status SetInputFilter(Filter * aFilter) = 0;

        /// \cond en
        /// \brief  This methode associate a processor to the adapter.
        /// \param  aProcessor The processor to associate
        /// \endcond
        /// \cond fr
        /// \brief Cette methode associe un processeur a l'adaptateur.
        /// \param aProcessor Le processeur a associer
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_INVALID_PROCESSOR
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_PROCESSOR_ALREADY_SET
        virtual Status SetProcessor(Processor * aProcessor) = 0;

        /// \cond en
        /// \brief  This methode allocate buffers.
        /// \param  aCount The number of buffer to allocate
        /// \endcond
        /// \cond fr
        /// \brief Cette methode alloue des espace memoire.
        /// \param aCount Le nombre d'espace memoire a allouer
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_INVALID_BUFFER_COUNT
        virtual Status Buffer_Allocate(unsigned int aCount) = 0;

        /// \cond en
        /// \brief  This methode release buffers.
        /// \param  aCount The number of buffer to release
        /// \endcond
        /// \cond fr
        /// \brief Cette methode relache des espace memoire.
        /// \param aCount Le nombre d'espace memoire a relacher
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_INVALID_BUFFER_COUNT
        virtual Status Buffer_Release(unsigned int aCount) = 0;

        /// \cond en
        /// \brief  Display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche
        /// \retval aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status Display(FILE * aOut) const = 0;

        /// \cond en
        /// \brief  This methode send a packet.
        /// \param  aData [---;R--] The data
        /// \param  aSize_byte      The size
        /// \endcond
        /// \cond fr
        /// \brief  Cette methode transmet un paquet.
        /// \param  aData [---;R--] Les donnees
        /// \param  aSize_byte      La taille
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_EXCEPTION
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        /// \retval STATUS_PACKET_TO_LARGE
        /// \retval STATUS_PACKET_TO_SMALL
        virtual Status Packet_Send(void * aData, unsigned int aSize_byte) = 0;

    protected:

        /// \cond en
        /// \brief  Default constructor
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur par defaut
        /// \endcond
        Adapter();

    private:

        Adapter(const Adapter &);

        const Adapter & operator = (const Adapter &);

    };

}
