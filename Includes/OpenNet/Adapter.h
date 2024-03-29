
// Product  OpenNet

/// \author     KMS - Martin Dubois, P.Eng.
/// \copyright  Copyright &copy; 2018-2020 KMS. All rights reserved.
/// \file       Includes/OpenNet/Adapter.h
/// \brief      OpenNet::Adapter (SDK)

// CODE REVIEW  2020-04-14  KMS - Martin Dubois, P.Eng.

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/StatisticsProvider.h>
#include <OpenNetK/Adapter_Types.h>

namespace OpenNet
{
    class Processor ;
    class SourceCode;
    class System    ;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class define the adapter level interface.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe d&eacute;finit l'interface au niveau de
    ///         l'adaptateur.
    /// \endcond
    class Adapter : public StatisticsProvider
    {

    public:

        /// \cond en
        /// \brief  The Adapter's configuration
        /// \endcond
        /// \cond fr
        /// \brief  La configuration de l'Adapter
        /// \endcond
        /// \sa     GetConfig, ResetConfig, SetConfig
        typedef struct
        {
            unsigned int mBufferQty      ;
            unsigned int mPacketSize_byte;

            struct
            {
                unsigned mMulticastPromiscuousDisable : 1;
                unsigned mUnicastPromiscuousDisable   : 1;

                unsigned mReserved0 : 30;
            }
            mFlags;

            // TODO  OpenNet.Adapter
            //       Normal (Feature) - Ajouter un "Buffer Factor" pour
            //       allouer un multiple du multiple prefere du Kernel
            //       (division ou multiplication).

            unsigned char mReserved0[612];

            OpenNetK::EthernetAddress  mEthernetAddress[50];
        }
        Config;

        /// \cond en
        /// \brief  The Adapter's information
        /// \endcond
        /// \cond fr
        /// \brief  L'information au sujet de l'Adapter
        /// \endcond
        /// \sa     GetInfo
        typedef OpenNetK::Adapter_Info   Info  ;

        /// \cond en
        /// \brief  The Adapter's state
        /// \endcond
        /// \cond fr
        /// \brief  L'&eacute de l'Adapter
        /// \endcond
        /// \sa     GetState
        typedef OpenNetK::Adapter_State  State ;

        /// \cond en
        /// \brief  The event callback declaration
        /// \param  aContext       The context passed to
        ///                        Event_RegisterCallback
        /// \param  aType          The event type
        /// \param  aTimestamp_us  The event's timestamp
        /// \endcond
        /// \cond fr
        /// \brief  La d&eacute;claration de la fonction appel&eacute; pour
        ///         signaler un evennement
        /// \param  aContext       Le context pass&eacute; &agrave;
        ///                        Event_RegisterCallback
        /// \param  aType          Le type d'evennement
        /// \param  aTimestamp_us  La marque de temps de l'evennement
        /// \endcond
        /// \param  aData0
        /// \param  aData1
        /// \sa     Event_RegisterCallback
        typedef void(*Event_Callback)(void * aContext, const OpenNetK::Event_Type aType, uint64_t aTimestamp_us, uint32_t aData0, void * aData1);

        /// \cond en
        /// \brief  Display
        /// \param  aIn   The Config instance to display
        /// \param  aOut  The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Afficher
        /// \param  aIn   L'instance de Config &agrave; afficher
        /// \param  aOut  Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_INVALID_REFERENCE
        static OPEN_NET_PUBLIC Status Display(const Config & aIn, FILE * aOut);

        /// \cond en
        /// \brief  Display
        /// \param  aIn   The Info instance to display
        /// \param  aOut  The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche
        /// \param  aIn   L'instance de Info &agrave; afficher
        /// \param  aOut  Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_INVALID_REFERENCE
        static OPEN_NET_PUBLIC Status Display(const Info & aIn, FILE * aOut);

        /// \cond en
        /// \brief  Display
        /// \param  aIn   The State instance to display
        /// \param  aOut  The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche
        /// \param  aIn   L'instance de State &agrave; afficher
        /// \param  aOut  Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_INVALID_REFERENCE
        static OPEN_NET_PUBLIC Status Display(const State & aIn, FILE * aOut);

        /// \cond en
        /// \brief  This methode return the adapter numero.
        /// \param  aOut  The methode return the information here.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode retourne le numero de l'adaptateur.
        /// \param  aOut  La medhode retourne l'information ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_ADAPTER_NOT_CONNECTED
        /// \retval STATUS_CORRUPTED_DRIVER_DATA
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \sa     IsConnected
        virtual Status GetAdapterNo(unsigned int * aOut) = 0;

        /// \cond en
        /// \brief  This methode return the configuration of the adapter.
        /// \param  aOut  The methode return the information here.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode retourne la configuration de
        ///         l'adaptateur.
        /// \param  aOut  La m&eacute;dhode retourne les informations ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \sa     SetConfig
        virtual Status GetConfig(Config * aOut) const = 0;

        /// \cond en
        /// \brief  This methode returns the information about the adapter.
        /// \param  aOut  The methode return the information here.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode retourne les informations au sujet
        ///         de l'adaptateur.
        /// \param  aOut  La m&eacute;thode retourne les informations ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status GetInfo(Info * aOut) const = 0;

        /// \cond en
        /// \brief  This method returns the adapter's name.
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode retourne le nom de l'adaptateur.
        /// \retval Cette m&eacute;thode retourne l'adresse d'un espace
        ///         m&eacute;moire interne.
        /// \endcond
        virtual const char * GetName() const = 0;

        /// \cond en
        /// \brief  This methode returns the state of the adapter.
        /// \param  aOut  The methode return the information here.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode retourne l'&eacute;tat de
        ///         l'adaptateur.
        /// \param  aOut  La m&eacute;thode retourne les informations ici.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status GetState(State * aOut) = 0;

        /// \cond en
        /// \brief  This methode indicate if the adapter is connected to a
        ///         system.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode indique si l'adaptateur est
        ///         connect&eacute; &agrave; un syst&egrave;me.
        /// \endcond
        /// \retval false
        /// \retval true
        /// \sa     GetAdapterNo
        virtual bool IsConnected() = 0;

        /// \cond en
        /// \brief  This methode indicate if the adapter is connected to the
        ///         system.
        /// \param  aSystem  The System instance
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode indique si l'adaptateur est
        ///         connect&eacute; au syst&egrave;me.
        /// \param  aSystem  L'instance de System
        /// \endcond
        /// \retval false
        /// \retval true
        /// \sa     GetAdapterNo
        virtual bool IsConnected(const System & aSystem) = 0;

        /// \cond en
        /// \brief  This methode reset the input filter.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode retire le filtre d'entr&eacute;.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_ADAPTER_RUNNING
        /// \retval STATUS_CUDA_ERROR
        /// \retval STATUS_FILTER_NOT_SET
        /// \retval STATUS_OPEN_CL_ERROR
        /// \sa     SetInputFilter
        virtual Status ResetInputFilter() = 0;

        /// \cond en
        /// \brief  This methode reset the processor.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode retire le processeur.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_FILTER_SET
        /// \retval STATUS_PROCESSOR_NOT_SET
        /// \sa     ResetInputFilter, SetProcessor
        virtual Status ResetProcessor() = 0;

        /// \cond en
        /// \brief  This methode set the configuration.
        /// \param  aConfig  The Config instance
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode change la configuration.
        /// \param  aConfig  L'instance de Config
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_INVALID_REFERENCE
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_PACKET_TOO_LARGE
        /// \retval STATUS_PACKET_TOO_SMALL
        /// \retval STATUS_TOO_MANY_BUFFER
        /// \sa     GetConfig, ResetConfig
        virtual Status SetConfig(const Config & aConfig) = 0;

        /// \cond en
        /// \brief  This methode set the input filter.
        /// \param  aSourceCode  The SourceCode instance
        /// \endcond
        /// \cond fr
        /// \brief Cette m&eacute;thode affecte le filtre d'entr&eacute.
        /// \param aSourceCode  L'instance de SourceCode
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_ADAPTER_NOT_CONNECTED
        /// \retval STAUS_CUDA_ERROR
        /// \retval STATUS_FILTER_ALREADY_SET
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_OPEN_CL_ERROR
        /// \retval STATUS_PROCESSOR_NOT_SET
        /// \sa     ResetInputFilter
        virtual Status SetInputFilter(SourceCode * aSourceCode) = 0;

        /// \cond en
        /// \brief  This methode associate a processor to the adapter.
        /// \param  aProcessor  The Processor instance
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode associe un processeur &agrave;
        ///         l'adaptateur.
        /// \param  aProcessor  L'intance de Processor
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_INVALID_PROCESSOR
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \retval STATUS_PROCESSOR_ALREADY_SET
        /// \sa     ResetProcessor
        virtual Status SetProcessor(Processor * aProcessor) = 0;

        /// \cond en
        /// \brief  Display
        /// \param  aOut  The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Afficher
        /// \param  aOut  Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status Display(FILE * aOut) const = 0;

        /// \cond en
        /// \brief  Wait for event
        /// \param  aCallback  The event processing function. Use NULL to
        ///                    unregister.
        /// \param  aContext   The context to pass to the function
        /// \endcond
        /// \cond fr
        /// \brief  Attendre pour des evennements
        /// \param  aCallback  La fonction de traitement des
        ///                    &eacute;v&eacute;nements. Passer NULL pour
        ///                    annuler l'enregistrement.
        /// \param  aContext   La contexte pass&eacute; &agrave; la fonction
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        virtual Status Event_RegisterCallback(Event_Callback aCallback, void * aContext) = 0;

        /// \cond en
        /// \brief  This methode send a packet.
        /// \param  aData       The data
        /// \param  aSize_byte  The size
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode transmet un paquet.
        /// \param  aData       Les donnees
        /// \param  aSize_byte  La taille
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        /// \retval STATUS_PACKET_TO_LARGE
        /// \retval STATUS_PACKET_TO_SMALL
        virtual Status Packet_Send(const void * aData, unsigned int aSize_byte) = 0;

        /// \cond en
        /// \brief  This methode read data from the driver.
        /// \param  aOut           The output buffer
        /// \param  aOutSize_byte  The size of the output buffer
        /// \param  aInfo_byte     The method put the size of read data here
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode lit des donn&eacute;es du pilote.
        /// \param  aOut           L'espace de sortie
        /// \param  aOutSize_byte  La taille de l'espace de sortie
        /// \param  aInfo_byte     La m&eacute;thode place la taille des
        ///                        donn&eacute;es lues ici
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_INVALID_SIZE
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status Read(void * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte) = 0;

        // TODO  OpenNet.Adapter.Tx_Disable
        //       Low (Cleanup) - Make it a flag in the configuration

        /// \cond en
        /// \brief  This methode disable packet transmit.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode d&eacute;sactive la transmission.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        /// \sa     Tx_Enable
        virtual Status Tx_Disable() = 0;

        /// \cond en
        /// \brief  This methode enable packet transmit.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode active la transmission.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        /// \sa     Tx_Disable
        virtual Status Tx_Enable() = 0;

        /// \cond en
        /// \brief  This methode reset the configuration.
        /// \endcond
        /// \cond fr
        /// \brief  Cette m&eacute;thode change la configuration pour la
        ///         configuration par d&eacute;faut.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        /// \sa     GetConfig, SetConfig
        virtual Status ResetConfig() = 0;

    protected:

        Adapter();

    private:

        Adapter(const Adapter &);

        const Adapter & operator = (const Adapter &);

    };

}
