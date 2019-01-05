
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Adapter.h
/// \brief   OpenNetK::Adapter

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNetK ==================================================
#include <OpenNetK/Adapter_Types.h>
#include <OpenNetK/Constants.h>
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

        /// \cond en
        /// \brief  DirectGMA Buffer information. This structur is not
        ///         documented and may change or disapear in futur version.
        /// \endcond
        /// \cond fr
        /// \brief  L'information au sujet d'un espace memoire DirectGMA.
        ///         Cette structure n'est pas documente. Elle peut changer
        ///         ou disparaitre dans une version future.
        /// \endcond
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

        static bool IoCtl_GetInfo(unsigned int aCode, void * aInfo);

        void Init(SpinLock * aZone0);

        void Buffer_SendPackets(BufferInfo * aBufferInfo);

        void Buffers_Process();

        void Disconnect();

        int  IoCtl(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte);

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
        int IoCtl_Start           (const Buffer         * aIn , unsigned int aInSize_byte );
        int IoCtl_State_Get       (      Adapter_State  * aOut);
        int IoCtl_Statistics_Get  (const void           * aIn , uint32_t * aOut, unsigned int aOutSize_byte) const;
        int IoCtl_Statistics_Reset();
        int IoCtl_Stop            ();

        Adapter   ** mAdapters ;
        unsigned int mAdapterNo;
        KEVENT     * mEvent    ;
        Hardware   * mHardware ;
        unsigned int mSystemId ;

        mutable uint32_t      mStatistics[32];
        mutable LARGE_INTEGER mStatistics_Start;

        // ===== Zone 0 =====================================================
        SpinLock * mZone0;

        unsigned int mBufferCount;
        BufferInfo   mBuffers[OPEN_NET_BUFFER_QTY];

    };

}
