
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Adapter.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNetK ==================================================
#include <OpenNetK/Constants.h>
#include <OpenNetK/Types.h>
#include <OpenNetK/Interface.h>

namespace OpenNetK
{

    class Hardware;

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
    /// \note   Classe noyau - Pas de constructer, pas de destructor, pas de
    ///         method virtuel
    /// \endcond
    class Adapter
    {

    public:

        /// \cond en
        /// \brief  Connect the hardware
        /// \param  aHardware [-K-;RW-] The new comment
        /// \endcond
        /// \cond fr
        /// \brief  Connecter le materiel
        /// \param  aHardware [-K-;RW-] Le materiel
        /// \endcond
        /// \note   Level = thread, Thread = Init
        void SetHardware(Hardware * aHardware);

    // Internal

        enum
        {
            IOCTL_RESULT_OK                = 0x00000000,

            IOCTL_RESULT_PROCESSING_NEEDED = 0xffffffe0,

            IOCTL_RESULT_ERROR             = 0xfffffffa,
            IOCTL_RESULT_INVALID_SYSTEM_ID = 0xfffffffb,
            IOCTL_RESULT_NO_BUFFER         = 0xfffffffc,
            IOCTL_RESULT_NOT_SET           = 0xfffffffd,
            IOCTL_RESULT_TOO_MANY_ADAPTER  = 0xfffffffe,
            IOCTL_RESULT_TOO_MANY_BUFFER   = 0xffffffff,
        };

        typedef struct
        {
            OpenNet_BufferInfo mBufferInfo;

            OpenNet_BufferHeader * mHeader;
            volatile uint32_t    * mMarker;

            struct
            {
                unsigned mStopRequested : 1;

                unsigned mReserved : 31;
            }
            mFlags;

            uint32_t      mMarkerValue;
            volatile long mRx_Counter ;
            volatile long mTx_Counter ;

            uint8_t mReserved1[32];
        }
        BufferInfo;

        typedef struct
        {
            unsigned int mIn_MinSize_byte ;
            unsigned int mOut_MinSize_byte;
        }
        IoCtlInfo;

        static bool IoCtl_GetInfo(unsigned int aCode, IoCtlInfo * aInfo);

        void Init();

        void Buffer_SendPackets(BufferInfo * aInfo);

        void Buffers_Process();

        void Disconnect();

        int  IoCtl(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte);

    private:

        void Buffer_InitHeader(OpenNet_BufferHeader * aHeader, const OpenNet_BufferInfo & aBufferInfo);
        void Buffer_Process   (BufferInfo * aBuffer);
        void Buffer_Queue     (const OpenNet_BufferInfo & aBufferInfo);
        void Buffer_Receive   (BufferInfo * aBuffer);
        void Buffer_Send      (BufferInfo * aBuffer);
        void Buffer_Stop      (BufferInfo * aBuffer);

        void Stop();

        // ===== IoCtl ======================================================

        int IoCtl_Config_Get (      OpenNet_Config     * aOut);
        int IoCtl_Config_Set (const OpenNet_Config     * aIn , OpenNet_Config * aOut);
        int IoCtl_Connect    (const OpenNet_Connect    * aIn );
        int IoCtl_Info_Get   (      OpenNet_Info       * aOut) const;
        int IoCtl_Packet_Send(const void               * aIn , unsigned int aInSize_byte );
        int IoCtl_Start      (const OpenNet_BufferInfo * aIn, unsigned int aInSize_byte);
        int IoCtl_State_Get  (      OpenNet_State      * aOut);
        int IoCtl_Stats_Get  (      OpenNet_Stats      * aOut) const;
        int IoCtl_Stats_Reset();
        int IoCtl_Stop       ();

        Adapter           ** mAdapters   ;
        unsigned int         mAdapterNo  ;
        unsigned int         mBufferCount;
        BufferInfo           mBuffers[OPEN_NET_BUFFER_QTY];
        KEVENT             * mEvent      ;
        Hardware           * mHardware   ;
        unsigned int         mSystemId   ;

        mutable OpenNet_Stats_Adapter         mStats        ;
        mutable OpenNet_Stats_Adapter_NoReset mStats_NoReset;

    };

}
