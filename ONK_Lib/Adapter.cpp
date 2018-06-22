
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Lib/Adapter.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================

#define INITGUID

#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>

// ===== Includes ===========================================================
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Constants.h>
#include <OpenNetK/Hardware.h>

#include <OpenNetK/Adapter.h>

// ===== Common =============================================================
#include "../Common/Version.h"

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static void SkipDangerousBoundary(uint64_t aLogical, unsigned int * aOffset_byte, unsigned int aSize_byte, unsigned int * aOutOffset_byte);

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void Adapter::SetHardware(Hardware * aHardware)
    {
        ASSERT(NULL != aHardware);

        ASSERT(NULL == mHardware);

        mHardware = aHardware;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    // aInfo [---;-W-]
    //
    // Static function - No stats
    //
    // Level   SoftInt
    // Thread  Queue
    bool Adapter::IoCtl_GetInfo(unsigned int aCode, IoCtlInfo * aInfo)
    {
        ASSERT(NULL != aInfo);

        memset(aInfo, 0, sizeof(IoCtlInfo));

        switch (aCode)
        {
        case OPEN_NET_IOCTL_CONFIG_GET :                                                       aInfo->mOut_MinSize_byte = sizeof(OpenNet_Config    ); break;
        case OPEN_NET_IOCTL_CONFIG_SET : aInfo->mIn_MinSize_byte = sizeof(OpenNet_Config    ); aInfo->mOut_MinSize_byte = sizeof(OpenNet_Config    ); break;
        case OPEN_NET_IOCTL_CONNECT    : aInfo->mIn_MinSize_byte = sizeof(OpenNet_Connect   );                                                        break;
        case OPEN_NET_IOCTL_INFO_GET   :                                                       aInfo->mOut_MinSize_byte = sizeof(OpenNet_Info      ); break;
        case OPEN_NET_IOCTL_PACKET_SEND:                                                                                                              break;
        case OPEN_NET_IOCTL_START      : aInfo->mIn_MinSize_byte = sizeof(OpenNet_BufferInfo);                                                        break;
        case OPEN_NET_IOCTL_STATE_GET  :                                                       aInfo->mOut_MinSize_byte = sizeof(OpenNet_State     ); break;
        case OPEN_NET_IOCTL_STATS_GET  :                                                       aInfo->mOut_MinSize_byte = sizeof(OpenNet_Stats     ); break;
        case OPEN_NET_IOCTL_STATS_RESET:                                                                                                              break;
        case OPEN_NET_IOCTL_STOP       :                                                                                                              break;

        default : return false;
        }

        return true;
    }

    // Level   Thread
    // Thread  Initialisation
    void Adapter::Init()
    {
        memset(&mStats        , 0, sizeof(mStats        ));
        memset(&mStats_NoReset, 0, sizeof(mStats_NoReset));

        mAdapters    = NULL;
        mAdapterNo   = OPEN_NET_ADAPTER_NO_UNKNOWN;
        mBufferCount =    0;
        mEvent       = NULL;
        mHardware    = NULL;
        mSystemId    =    0;
    }

    // aBuffer [-K-;RW-]
    //
    // Level   SoftInt
    // Thread  SoftInt
    void Adapter::Buffer_SendPackets(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer                                 );
        ASSERT(NULL != aBuffer->mHeader                        );
        ASSERT(   0 <  aBuffer->mHeader->mPacketInfoOffset_byte);
        ASSERT(   0 <  aBuffer->mHeader->mPacketQty            );

        ASSERT(OPEN_NET_ADAPTER_NO_QTY >  mAdapterNo);
        ASSERT(NULL                    != mHardware );

        uint32_t  lAdapterBit = 1 << mAdapterNo;
        uint8_t * lBase       = reinterpret_cast<uint8_t *>(aBuffer->mHeader);

        OpenNet_PacketInfo * lPacketInfo = reinterpret_cast<OpenNet_PacketInfo *>(lBase + aBuffer->mHeader->mPacketInfoOffset_byte);

        for (unsigned int i = 0; i < aBuffer->mHeader->mPacketQty; i++)
        {
            ASSERT(0 < lPacketInfo[i].mPacketOffset_byte);

            switch (lPacketInfo[i].mPacketState)
            {
            case OPEN_NET_PACKET_STATE_RECEIVED :
            case OPEN_NET_PACKET_STATE_RECEIVING:
                // TODO  ONK_Lib.Adapter.PartialBuffer  Add statistics
                break;

            case OPEN_NET_PACKET_STATE_PROCESSED:
                lPacketInfo[i].mPacketState = OPEN_NET_PACKET_STATE_SENDING;
                // no break;

            case OPEN_NET_PACKET_STATE_SENDING:
                if (0 != (lPacketInfo[i].mToSendTo & lAdapterBit))
                {
                    mHardware->Packet_Send(aBuffer->mBufferInfo.mBuffer_PA + lPacketInfo[i].mPacketOffset_byte, lPacketInfo[i].mPacketSize_byte, &aBuffer->mTx_Counter);
                    mStats.mTx_Packet++;
                }
                break;

            default: ASSERT(false);
            }
        }

        mStats.mBuffer_SendPackets++;
    }

    // Level    SoftInt
    // Threada  Queue or SoftInt
    void Adapter::Buffers_Process()
    {
        ASSERT(OPEN_NET_BUFFER_QTY >= mBufferCount);

        for (unsigned int i = 0; i < mBufferCount; i++)
        {
            ASSERT(NULL != mBuffers[i].mHeader);

            switch (mBuffers[i].mHeader->mBufferState)
            {
            case OPEN_NET_BUFFER_STATE_PROCESSED : Buffer_Send(mBuffers + i); break;
            case OPEN_NET_BUFFER_STATE_PROCESSING:                            break;

            case OPEN_NET_BUFFER_STATE_RECEIVING:
                if (0 == mBuffers[i].mRx_Counter)
                {
                    Buffer_Process(mBuffers + i);
                }
                break;

            case OPEN_NET_BUFFER_STATE_SENDING:
                if (0 == mBuffers[i].mTx_Counter)
                {
                    if (mBuffers[i].mFlags.mStopRequested) { Buffer_Stop   (mBuffers + i); }
                    else                                   { Buffer_Receive(mBuffers + i); }
                }
                break;

            case OPEN_NET_BUFFER_STATE_STOPPED:
                if (i == (mBufferCount - 1))
                {
                    mBufferCount--;
                }
                break;

            default: ASSERT(false);
            }
        }

        mStats.mBuffers_Process++;
    }

    // Level   Thread
    // Thread  User
    void Adapter::Disconnect()
    {
        // AdapterNo is not set if Disconnect is called by Connect on an
        // error

        ASSERT(NULL != mAdapters);
        ASSERT(NULL != mEvent   );
        ASSERT(   0 != mSystemId);

        if (0 < mBufferCount)
        {
            Stop();
        }

        mAdapters  = NULL                       ;
        mAdapterNo = OPEN_NET_ADAPTER_NO_UNKNOWN;
        mEvent     = NULL                       ;
        mSystemId  =                           0;
    }

    // aIn  [---;R--]
    // aOut [---;-W-]
    //
    // Return  See IOCTL_RESULT_...
    //
    // Level   SoftInt
    // Thread  Queue
    int Adapter::IoCtl(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte)
    {
        (void)(aOutSize_byte);

        int lResult = IOCTL_RESULT_NOT_SET;

        switch (aCode)
        {
        case OPEN_NET_IOCTL_CONFIG_GET : lResult = IoCtl_Config_Get (reinterpret_cast<      OpenNet_Config     *>(aOut)); break;
        case OPEN_NET_IOCTL_CONFIG_SET : lResult = IoCtl_Config_Set (reinterpret_cast<const OpenNet_Config     *>(aIn ), reinterpret_cast<OpenNet_Config *>(aOut)); break;
        case OPEN_NET_IOCTL_CONNECT    : lResult = IoCtl_Connect    (reinterpret_cast<const OpenNet_Connect    *>(aIn )); break;
        case OPEN_NET_IOCTL_INFO_GET   : lResult = IoCtl_Info_Get   (reinterpret_cast<      OpenNet_Info       *>(aOut)); break;
        case OPEN_NET_IOCTL_PACKET_SEND: lResult = IoCtl_Packet_Send(                                             aIn  , aInSize_byte ); break;
        case OPEN_NET_IOCTL_START      : lResult = IoCtl_Start      (reinterpret_cast<const OpenNet_BufferInfo *>(aIn ), aInSize_byte ); break;
        case OPEN_NET_IOCTL_STATE_GET  : lResult = IoCtl_State_Get  (reinterpret_cast<      OpenNet_State      *>(aOut)); break;
        case OPEN_NET_IOCTL_STATS_GET  : lResult = IoCtl_Stats_Get  (reinterpret_cast<      OpenNet_Stats      *>(aOut)); break;
        case OPEN_NET_IOCTL_STATS_RESET: lResult = IoCtl_Stats_Reset(); break;
        case OPEN_NET_IOCTL_STOP       : lResult = IoCtl_Stop       (); break;

        default: ASSERT(false);
        }

        mStats.mIoCtl++;

        mStats_NoReset.mIoCtl_Last        = aCode  ;
        mStats_NoReset.mIoCtl_Last_Result = lResult;

        return lResult;
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    // aHeader     [---;-W-]
    // aBufferInfo [---;R--]
    //
    // Levels   SoftInt or Thread
    // Threads  Queue
    void Adapter::Buffer_InitHeader(OpenNet_BufferHeader * aHeader, const OpenNet_BufferInfo & aBufferInfo)
    {
        ASSERT(NULL !=   aHeader               );
        ASSERT(NULL != (&aBufferInfo)          );
        ASSERT(   0 <    aBufferInfo.mPacketQty);

        ASSERT(NULL != mHardware);

        OpenNet_PacketInfo * lPacketInfo      = reinterpret_cast<OpenNet_PacketInfo *>(aHeader + 1);
        unsigned int         lPacketQty       = aBufferInfo.mPacketQty;
        unsigned int         lPacketSize_byte = mHardware->GetPacketSize();

        ASSERT(OPEN_NET_PACKET_SIZE_MAX_byte >= lPacketSize_byte);
        ASSERT(OPEN_NET_PACKET_SIZE_MIN_byte <= lPacketSize_byte);

        unsigned int lPacketOffset_byte = sizeof(OpenNet_BufferHeader) + (sizeof(OpenNet_PacketInfo) * lPacketQty);

        memset(aHeader, 0, lPacketOffset_byte);

        aHeader->mBufferState           = OPEN_NET_BUFFER_STATE_SENDING;
        aHeader->mPacketInfoOffset_byte = sizeof(OpenNet_BufferHeader);
        aHeader->mPacketQty             = lPacketQty;
        aHeader->mPacketSize_byte       = lPacketSize_byte;

        for (unsigned int i = 0; i < lPacketQty; i++)
        {
            lPacketInfo[i].mPacketState = OPEN_NET_PACKET_STATE_SENDING;

            SkipDangerousBoundary(aBufferInfo.mBuffer_PA, &lPacketOffset_byte, lPacketSize_byte, &lPacketInfo[i].mPacketOffset_byte);
        }

        mStats.mBuffer_InitHeader++;
    }

    // aBuffer [---;R--]
    //
    // Level   SoftInt
    // Thread  SoftInt
    void Adapter::Buffer_Process(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer         );
        ASSERT(NULL != aBuffer->mHeader);
        ASSERT(NULL != aBuffer->mMarker);

        aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_PROCESSING;

        aBuffer->mMarkerValue++;

        (*aBuffer->mMarker) = aBuffer->mMarkerValue;

        mStats.mBuffer_Process++;
    }

    // aBufferInfo [---;R--]
    //
    // Level   SoftInt or Thread
    // Thread  Queue
    void Adapter::Buffer_Queue(const OpenNet_BufferInfo & aBufferInfo)
    {
        ASSERT(NULL != (&aBufferInfo)        );
        ASSERT(   0 <  aBufferInfo.mPacketQty);

        ASSERT(OPEN_NET_BUFFER_QTY > mBufferCount);

        memset(mBuffers + mBufferCount, 0, sizeof(mBuffers[mBufferCount]));

        mBuffers[mBufferCount].mBufferInfo = aBufferInfo;

        PHYSICAL_ADDRESS lPA;

        lPA.QuadPart = aBufferInfo.mBuffer_PA;

        unsigned int lSize_byte = sizeof(OpenNet_BufferHeader) + sizeof(OpenNet_PacketInfo) * aBufferInfo.mPacketQty;

        mBuffers[mBufferCount].mHeader = reinterpret_cast<OpenNet_BufferHeader *>(MmMapIoSpace(lPA, lSize_byte, MmNonCached));
        ASSERT(NULL != mBuffers[mBufferCount].mHeader);

        lPA.QuadPart = aBufferInfo.mMarker_PA;

        mBuffers[mBufferCount].mMarker = reinterpret_cast<uint32_t *>(MmMapIoSpace(lPA, PAGE_SIZE, MmNonCached));
        ASSERT(NULL != mBuffers[mBufferCount].mMarker);

        Buffer_InitHeader(mBuffers[mBufferCount].mHeader, aBufferInfo);
        
        mBufferCount++;

        mStats.mBuffer_Queue++;
    }

    // aBuffer [-K-;R--]
    //
    // Level    SoftInt
    // Threads  Queue or SoftInt
    void Adapter::Buffer_Receive(BufferInfo * aBuffer)
    {
        ASSERT(NULL                          != aBuffer                                 );
        ASSERT(NULL                          != aBuffer->mHeader                        );
        ASSERT(OPEN_NET_BUFFER_STATE_SENDING == aBuffer->mHeader->mBufferState          );
        ASSERT(0                             <  aBuffer->mHeader->mPacketInfoOffset_byte);
        ASSERT(0                             <  aBuffer->mHeader->mPacketQty            );

        ASSERT(NULL != mHardware);

        uint8_t * lBase = reinterpret_cast<uint8_t *>(aBuffer->mHeader);

        OpenNet_PacketInfo * lPacketInfo = reinterpret_cast<OpenNet_PacketInfo *>(lBase + aBuffer->mHeader->mPacketInfoOffset_byte);

        for (unsigned int i = 0; i < aBuffer->mHeader->mPacketQty; i++)
        {
            ASSERT(0 < lPacketInfo[i].mPacketOffset_byte);

            mHardware->Packet_Receive(aBuffer->mBufferInfo.mBuffer_PA + lPacketInfo[i].mPacketOffset_byte, lPacketInfo + i, &aBuffer->mRx_Counter);
        }

        aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_RECEIVING;

        mStats.mBuffer_Receive++;
    }

    // aBuffer [-K-;R--]
    //
    // Level   SoftInt
    // Thread  SoftInt
    void Adapter::Buffer_Send(BufferInfo * aBuffer)
    {
        ASSERT(NULL                            != aBuffer                       );
        ASSERT(NULL                            != aBuffer->mHeader              );
        ASSERT(OPEN_NET_BUFFER_STATE_PROCESSED == aBuffer->mHeader->mBufferState);

        if (NULL != mAdapters)
        {
            for (unsigned int i = 0; i < OPEN_NET_ADAPTER_NO_QTY; i++)
            {
                if (NULL != mAdapters[i])
                {
                    // TODO  ONL_Lib.Adapter
                    //       Here, the Buffer_SendPackets of the other
                    //       adapter is called without holding the mZone0 of
                    //       this other adapter. Worst, mZone adapter of the
                    //       firs adapter is locked, what could cause a dead
                    //       lock if we try to lock the mZone of the other
                    //       adapter. A solution could be to use a shared
                    //       lock once the adapter is locked.
                    mAdapters[i]->Buffer_SendPackets(aBuffer);
                }
            }

            aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_SENDING;
        }
        else
        {
            aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_STOPPED;
        }

        mStats.mBuffer_Send++;
    }

    // Level   SoftInt
    // Thread  SoftInt
    void Adapter::Buffer_Stop(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer         );
        ASSERT(NULL != aBuffer->mHeader);

        aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_STOPPED;

        mStats.mBuffer_Stop++;
    }

    // Level   Thread or SoftInt
    // Thread  Queue or User
    void Adapter::Stop()
    {
        ASSERT(0 < mBufferCount);

        for (unsigned int i = 0; i < mBufferCount; i++)
        {
            mBuffers[i].mFlags.mStopRequested = true;
        }
    }

    // ===== IoCtl ==========================================================

    int Adapter::IoCtl_Config_Get(OpenNet_Config * aOut)
    {
        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        mHardware->GetConfig(aOut);

        mStats.mIoCtl_Config_Get++;

        return sizeof(OpenNet_Config);
    }

    int Adapter::IoCtl_Config_Set(const OpenNet_Config * aIn, OpenNet_Config * aOut)
    {
        ASSERT(NULL != aIn );
        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        mHardware->SetConfig(*aIn);
        mHardware->GetConfig(aOut);

        mStats.mIoCtl_Config_Set++;

        return sizeof(OpenNet_Config);
    }

    int Adapter::IoCtl_Connect(const OpenNet_Connect * aIn)
    {
        ASSERT(NULL != aIn);

        ASSERT(   0 != aIn->mEvent       );
        ASSERT(NULL != aIn->mSharedMemory);

        ASSERT(NULL                        == mAdapters );
        ASSERT(OPEN_NET_ADAPTER_NO_UNKNOWN == mAdapterNo);
        ASSERT(NULL                        == mEvent    );
        ASSERT(                          0 == mSystemId );

        mStats.mIoCtl_Connect++;

        if (0 == aIn->mSystemId)
        {
            return IOCTL_RESULT_INVALID_SYSTEM_ID;
        }

        mEvent    = reinterpret_cast<KEVENT   *>(aIn->mEvent       );
        mAdapters = reinterpret_cast<Adapter **>(aIn->mSharedMemory);
        mSystemId =                              aIn->mSystemId     ;

        for (unsigned int i = 0; i < OPEN_NET_ADAPTER_NO_QTY; i++)
        {
            if (NULL == mAdapters[i])
            {
                mAdapters[i] = this;
                mAdapterNo = i;

                return IOCTL_RESULT_OK;
            }
        }

        Disconnect();

        return IOCTL_RESULT_TOO_MANY_ADAPTER;
    }

    int Adapter::IoCtl_Info_Get(OpenNet_Info * aOut) const
    {
        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        mHardware->GetInfo(aOut);

        mStats.mIoCtl_Info_Get++;

        return sizeof(OpenNet_Info);
    }

    // TODO  ONK_Lib.Adapter.ErrorHandling
    //       Verify if aIn = NULL and aInSize_byte = 0 cause problem.
    int Adapter::IoCtl_Packet_Send(const void * aIn, unsigned int aInSize_byte)
    {
        ASSERT(NULL != mHardware);

        mHardware->Packet_Send(aIn, aInSize_byte);

        mStats.mIoCtl_Packet_Send++;

        return IOCTL_RESULT_OK;
    }

    int Adapter::IoCtl_Start(const OpenNet_BufferInfo * aIn, unsigned int aInSize_byte)
    {
        ASSERT(NULL                       != aIn         );
        ASSERT(sizeof(OpenNet_BufferInfo) <= aInSize_byte);

        ASSERT(OPEN_NET_BUFFER_QTY >= mBufferCount);

        mStats.mIoCtl_Start++;

        unsigned int lCount = aInSize_byte / sizeof(OpenNet_BufferInfo);

        if (OPEN_NET_BUFFER_QTY < (mBufferCount + lCount))
        {
            return IOCTL_RESULT_TOO_MANY_BUFFER;
        }

        for (unsigned int i = 0; i < lCount; i++)
        {
            Buffer_Queue(aIn[i]);
        }

        return IOCTL_RESULT_PROCESSING_NEEDED;
    }

    int Adapter::IoCtl_State_Get(OpenNet_State * aOut)
    {
        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        memset(aOut, 0, sizeof(OpenNet_State));

        aOut->mAdapterNo   = mAdapterNo  ;
        aOut->mBufferCount = mBufferCount;
        aOut->mSystemId    = mSystemId   ;

        mHardware->GetState(aOut);

        mStats.mIoCtl_State_Get++;

        return sizeof(OpenNet_State);
    }

    int Adapter::IoCtl_Stats_Get(OpenNet_Stats * aOut) const
    {
        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        memcpy(&aOut->mAdapter        , &mStats        , sizeof(mStats        ));
        memcpy(&aOut->mAdapter_NoReset, &mStats_NoReset, sizeof(mStats_NoReset));

        mHardware->Stats_Get(aOut);

        mStats.mIoCtl_Stats_Get++;
        
        return sizeof(OpenNet_Stats);
    }

    int Adapter::IoCtl_Stats_Reset()
    {
        ASSERT(NULL != mHardware);

        memset(&mStats, 0, sizeof(mStats));

        mHardware->Stats_Reset();

        mStats_NoReset.mIoCtl_Stats_Reset++;

        return IOCTL_RESULT_OK;
    }

    int Adapter::IoCtl_Stop()
    {
        if (0 >= mBufferCount)
        {
            return IOCTL_RESULT_NO_BUFFER;
        }

        Stop();

        mStats.mIoCtl_Stop++;

        return IOCTL_RESULT_OK;
    }

}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aOffset_byte    [---;RW-]
// aOutOffset_byte [---;-W-]
//
// Levels  SoftInt or Thread
// Thread  Queue
void SkipDangerousBoundary(uint64_t aLogical, unsigned int * aOffset_byte, unsigned int aSize_byte, unsigned int * aOutOffset_byte)
{
    ASSERT(NULL                                  != aOffset_byte   );
    ASSERT(                                    0 <  aSize_byte     );
    ASSERT(OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte >  aSize_byte     );
    ASSERT(NULL                                  != aOutOffset_byte);

    uint64_t lBegin = aLogical + (*aOffset_byte);
    uint64_t lEnd   = lBegin + aSize_byte - 1;

    if ((lBegin & OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte) == (lEnd & OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte))
    {
        (*aOutOffset_byte) = (*aOffset_byte);
    }
    else
    {
        uint64_t lOffset_byte = OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte - (lBegin % OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte);

        (*aOutOffset_byte) = (*aOffset_byte) + static_cast<unsigned int>(lOffset_byte);
    }

    (*aOffset_byte) = (*aOutOffset_byte) + aSize_byte;
}
