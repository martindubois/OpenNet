
// Product / Produit  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Adapter.h
/// \brief      OpenNetK::Adapter

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNetK ==================================================
#include <OpenNetK/Adapter_Types.h>
#include <OpenNetK/Constants.h>
#include <OpenNetK/PacketGenerator_Types.h>
#include <OpenNetK/Types.h>

class Packet;

namespace OpenNetK
{

    class Hardware;
    class SpinLock;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class maintains information about an adapter on the
    ///         OpenNet internal network.
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         method
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe maintien les information concernant un
    ///         adaptateur sur le reseau interne OpenNet.
    /// \note   Classe noyau - Pas de constructeur, pas de destructor, pas de
    ///         method virtuel
    /// \endcond
    class Adapter
    {

    public:

        /// \cond en
        /// \brief  This structure contains information about buffer size an
        ///         IoCtl accepts.
        /// \endcond
        /// \cond fr
        /// \brief  Cette structure contient les tailles d'espace memoire
        ///         qu'un IoCtl accepte.
        /// \endcond
        /// \todo   Document the members
        typedef struct
        {
            unsigned int mIn_MaxSize_byte ;
            unsigned int mIn_MinSize_byte ;
            unsigned int mOut_MinSize_byte;
        }
        IoCtl_Info;

        /// \cond en
        /// \brief  Retrieve information about an IoCtl code
        /// \param  aCode  The IoCtl code
        /// \param  aInfo  The output buffer
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Optenir l'information au sujet d'un code IoCtl
        /// \param  aCode  Le code IoCtl
        /// \param  aInfo  L'espace memoire de sortie
        /// \retval false  Erreur
        /// \endcond
        /// \retval true   OK
        static bool IoCtl_GetInfo(unsigned int aCode, IoCtl_Info * aInfo);

        /// \cond en
        /// \brief  Connect the Hardware instance
        /// \param  aHardware [-K-;RW-] The Hardware instance
        /// \endcond
        /// \cond fr
        /// \brief  Connecter l'instance de la classe Hardware
        /// \param  aHardware [-K-;RW-] L'instance de la classe Hardware
        /// \endcond
        /// \note   Level = Thread, Thread = Init
        void SetHardware(Hardware * aHardware);

    // Internal

        // TODO  OpenNetK.Adapter
        //       Normal (Cleanup) - Definir la structure BufferInfo dans un
        //       fichier prive. En faire une classe.

        typedef struct
        {
            Buffer mBuffer;

            uint8_t              * mBase  ;
            OpenNet_BufferHeader * mHeader;
            volatile uint32_t    * mMarker;
            Packet               * mPackets;

            struct
            {
                unsigned mStopRequested : 1;

                unsigned mReserved : 31;
            }
            mFlags;

            uint32_t      mMarkerValue;
            unsigned int  mPacketInfoOffset_byte;
            volatile long mRx_Counter ;
            volatile long mTx_Counter ;

            uint8_t mReserved1[32];
        }
        BufferInfo;

        void Init(SpinLock * aZone0);

        void Buffer_SendPackets(BufferInfo * aBufferInfo);

        // TODO  OpenNetK.Adapter
        //       Renommer Buffers_Process en Interrupt_Process2
        void Buffers_Process(bool * aNeedMoreProcessing);

        void Disconnect();

        void Interrupt_Process3();

        int  IoCtl(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte);

        void Tick();

    private:

        void Buffer_InitHeader_Zone0 (OpenNet_BufferHeader * aHeader, const Buffer & aBuffer, Packet * aPackets);
        void Buffer_Queue_Zone0      (const Buffer & aBuffer);
        void Buffer_Receive_Zone0    (BufferInfo * aBufferInfo);
        void Buffer_Send_Zone0       (BufferInfo * aBufferInfo);
        void Buffer_WriteMarker_Zone0(BufferInfo * aBufferInfo);

        void Stop_Zone0();

        // ===== Buffer_State ===============================================
        void Buffer_PxCompleted_Zone0(BufferInfo * aBufferInfo);
        void Buffer_PxRunning_Zone0  (BufferInfo * aBufferInfo);
        void Buffer_RxRunning_Zone0  (BufferInfo * aBufferInfo);
        void Buffer_Stopped_Zone0    (unsigned int aIndex     );
        void Buffer_TxRunning_Zone0  (BufferInfo * aBufferInfo);

        // ===== IoCtl ======================================================
        int IoCtl_Config_Get      (      Adapter_Config * aOut);
        int IoCtl_Config_Set      (const Adapter_Config * aIn , Adapter_Config * aOut);
        int IoCtl_Connect         (const void           * aIn );
        int IoCtl_Info_Get        (      Adapter_Info   * aOut) const;
        int IoCtl_Packet_Send     (const void           * aIn , unsigned int aInSize_byte );
        int IoCtl_Packet_Send_Ex  (const void           * aIn , unsigned int aInSize_byte );
        int IoCtl_PacketGenerator_Config_Get(      PacketGenerator_Config * aOut);
        int IoCtl_PacketGenerator_Config_Set(const PacketGenerator_Config * aIn , PacketGenerator_Config * aOut);
        int IoCtl_PacketGenerator_Start     ();
        int IoCtl_PacketGenerator_Stop      ();
        int IoCtl_Start           (const Buffer         * aIn , unsigned int aInSize_byte );
        int IoCtl_State_Get       (      Adapter_State  * aOut);
        int IoCtl_Statistics_Get  (const void           * aIn , uint32_t * aOut, unsigned int aOutSize_byte) const;
        int IoCtl_Statistics_Reset();
        int IoCtl_Stop            ();

        Adapter   ** mAdapters ;
        unsigned int mAdapterNo;
        Hardware   * mHardware ;
        unsigned int mSystemId ;

        PacketGenerator_Config mPacketGenerator_Config ;
        unsigned int           mPacketGenerator_Counter;
        long                   mPacketGenerator_Pending;
        bool                   mPacketGenerator_Running;

        mutable uint32_t      mStatistics[32];

        #ifdef _KMS_WINDOWS_
            KEVENT              * mEvent           ;
            mutable LARGE_INTEGER mStatistics_Start;
        #endif

        // ===== Zone 0 =====================================================
        SpinLock * mZone0;

        unsigned int mBufferCount;
        BufferInfo   mBuffers[OPEN_NET_BUFFER_QTY];

    };

}
