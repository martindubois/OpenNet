
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
    class System;

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

        /// \cond en
        /// \brief  This structure contains statistics about the Dll internal
        ///         operation. The fields may change in future versions.
        /// \endcond
        /// \cond fr
        /// \brief  Cette structure contient des statistiques au sujet du
        ///         fonctionnement inerne de la DLL.
        ///         Les champs peuvent changer dans les versions futures.
        /// \endcond
        typedef struct
        {
            unsigned int mBuffer_Allocated             ; //  0
            unsigned int mBuffer_Released              ;
            unsigned int mLoopBackPacket               ;
            unsigned int mPacket_Send                  ;
            unsigned int mRun_Entry                    ;
            unsigned int mRun_Exception                ; //  5
            unsigned int mRun_Exit                     ;
            unsigned int mRun_Iteration_Queue          ;
            unsigned int mRun_Iteration_Wait           ;
            unsigned int mRun_Loop_Exception           ;
            unsigned int mRun_Loop_UnexpectedException ; // 10
            unsigned int mRun_Queue                    ;
            unsigned int mRun_UnexpectedException      ;
            unsigned int mStart                        ;
            unsigned int mStop_Request                 ;
            unsigned int mStop_Wait                    ; // 15

            unsigned int mReserved0[112];
        }
        Stats_Dll;

        /// \cond en
        /// \brief  This structure contains statistics.
        /// \endcond
        /// \cond fr
        /// \brief  Cette structure contient des statistiques.
        /// \endcond
        typedef struct
        {
            Stats_Dll      mDll   ;
            OpenNet_Stats  mDriver;
        }
        Stats;

        /// \cond en
        /// \brief  Display
        /// \param  aIn  [---;R--] The Config instance to display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche  Le Status
        /// \param  aIn  [---;R--] L'instance de Config a afficher
        /// \param  aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_INVALID_REFERENCE
        static OPEN_NET_PUBLIC Status Display(const Config & aIn, FILE * aOut);

        /// \cond en
        /// \brief  Display
        /// \param  aIn  [---;R--] The Info instance to display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche  Le Status
        /// \param  aIn  [---;R--] L'instance de Info a afficher
        /// \param  aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_INVALID_REFERENCE
        static OPEN_NET_PUBLIC Status Display(const Info & aIn  , FILE * aOut);

        /// \cond en
        /// \brief  Display
        /// \param  aIn  [---;R--] The State instance to display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche  Le Status
        /// \param  aIn  [---;R--] L'instance de State a afficher
        /// \param  aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_INVALID_REFERENCE
        static OPEN_NET_PUBLIC Status Display(const State  & aIn, FILE * aOut);

        /// \cond en
        /// \brief  Display
        /// \param  aIn  [---;R--] The Stats instance to display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Afficher
        /// \param  aIn  [---;R--] L'instance de Stats a afficher
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
        /// \param aOut [---;-W-] La medhode retourne l'information ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_ADAPTER_NOT_CONNECTED
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
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
        /// \brief  This method returns the adapter's name.
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Cette methode retourne le nom de l'adaptateur
        /// \retval Cette methode retourne l'adresse d'un espace memoire
        ///         interne.
        /// \endcond
        virtual const char * GetName() const = 0;

        /// \cond en
        /// \brief  This method returns the packet size.
        /// \return This method returns the packet size in bytes.
        /// \endcond
        /// \cond fr
        /// \brief  Cette methode retourne la taille des paquets.
        /// \retval Cette methode retourne la taille des paquets en octets.
        /// \endcond
        virtual unsigned int GetPacketSize() const = 0;

        /// \cond en
        /// \brief  This methode returns the state of the adapter.
        /// \param  aOut [---;-W-] The methode return the information here.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode retourne l'etat de l'adaptateur.
        /// \param aOut [---;-W-] La medhode retourne les informations ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status GetState(State * aOut) = 0;

        /// \cond en
        /// \brief  This methode return the statistics of the adapter.
        /// \param  aOut [---;-W-] The methode return the statistics here.
        /// \param  aReset         Reset statistics to 0
        /// \endcond
        /// \cond fr
        /// \brief Cette methode retourne les statistiques de l'adaptateur.
        /// \param aOut [---;-W-] La medhode retourne les statistiques ici.
        /// \param aReset         Remettre les statistiques a zero
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status GetStats(Stats * aOut, bool aReset) = 0;

        /// \cond en
        /// \brief  This methode indicate if the adapter is connected to a
        ///         system.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode indique si l'adaptateur est connecte a un
        ///        system.
        /// \endcond
        /// \retval false
        /// \retval true
        virtual bool IsConnected() = 0;

        /// \cond en
        /// \brief  This methode indicate if the adapter is connected to the
        ///         system.
        /// \param  aSystem [---;R--] The System instance
        /// \endcond
        /// \cond fr
        /// \brief Cette methode indique si l'adaptateur est connecte au
        ///        system.
        /// \param aSystem [---;R--] L'instance de System
        /// \endcond
        /// \retval false
        /// \retval true
        virtual bool IsConnected(const System & aSystem) = 0;

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
        /// \retval STATUS_IOCTL_ERROR
        virtual Status ResetStats() = 0;

        /// \cond en
        /// \brief  This methode set the configuration.
        /// \param  aConfig [---;RW-] The Config instance
        /// \endcond
        /// \cond fr
        /// \brief Cette methode change la configuration.
        /// \param aConfig [---;RW-] L'instance de Config
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_INVALID_REFERENCE
        /// \retval STATUS_IOCTL_ERROR
        virtual Status SetConfig(const Config & aConfig) = 0;

        /// \cond en
        /// \brief  This methode set the input filter.
        /// \param  aFilter [---;RW-] The Filter instance
        /// \endcond
        /// \cond fr
        /// \brief Cette methode affecte le filtre d'entre.
        /// \param aFilter [---;RW-] L'instance de Filter
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_FILTER_ALREADY_SET
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_PROCESSOR_NOT_SET
        virtual OpenNet::Status SetInputFilter(Filter * aFilter) = 0;

        /// \cond en
        /// \brief  This methode set the packet size.
        /// \param  aSize_byte  The packet size
        /// \endcond
        /// \cond fr
        /// \brief Cette methode change la taille des paquets.
        /// \param aSize_byte  La taille des paquets
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_PACKET_TOO_LARGE
        /// \retval STATUS_PACKET_TOO_SMALL
        virtual OpenNet::Status SetPacketSize(unsigned int aSize_byte) = 0;

        /// \cond en
        /// \brief  This methode associate a processor to the adapter.
        /// \param  aProcessor [-K-;RW-] The Processor instance
        /// \endcond
        /// \cond fr
        /// \brief  Cette methode associe un processeur a l'adaptateur.
        /// \param  aProcessor [-K-;RW-] L'intance de Processeur
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
        /// \retval STATUS_TOO_MANY_BUFFER
        virtual Status Buffer_Allocate(unsigned int aCount) = 0;

        /// \cond en
        /// \brief  This methode release buffers.
        /// \param  aCount  The number of buffer to release
        /// \endcond
        /// \cond fr
        /// \brief  Cette methode relache des espace memoire.
        /// \param  aCount  Le nombre d'espace memoire a relacher
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
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        /// \retval STATUS_PACKET_TO_LARGE
        /// \retval STATUS_PACKET_TO_SMALL
        virtual Status Packet_Send(const void * aData, unsigned int aSize_byte) = 0;

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
