
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

#include <OpenNetK/Hardware.h>

#include <OpenNetK/Adapter.h>

// ===== Common =============================================================
#include "../Common/Version.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define SIZE_64_KB (64 * 1024)

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static void Skip64KByteBoundary(uint64_t aLogical, unsigned int * aOffset_byte, unsigned int aSize_byte, unsigned int * aOutOffset_byte);

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

    void Adapter::Init(CompletePendingRequest aCompletePendingRequest, void * aContext)
    {
        ASSERT(NULL != aCompletePendingRequest);
        ASSERT(NULL != aContext               );

        memset(&mStats, 0, sizeof(mStats));

        mAdapters               = NULL;
        mAdapterNo              = OPEN_NET_ADAPTER_NO_UNKNOWN;
        mBufferCount            =    0;
        mCompletePendingRequest = aCompletePendingRequest;
        mContext                = aContext;
        mEvent                  = NULL;
        mHardware               = NULL;
        mSystemId               =    0;
    }

    // aBuffer [-K-;RW-]
    //
    // Level   SoftInt
    // Thread  SoftInt
    //
    // TODO  Test
    void Adapter::Buffer_SendPackets(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer         );
        ASSERT(NULL != aBuffer->mHeader);

        ASSERT(OPEN_NET_ADAPTER_NO_QTY >  mAdapterNo);
        ASSERT(NULL                    != mHardware );

        uint32_t  lAdapterBit = 1 << mAdapterNo;
        uint8_t * lBase       = reinterpret_cast<uint8_t *>(aBuffer->mHeader);

        OpenNet_PacketInfo * lPacketInfo = reinterpret_cast<OpenNet_PacketInfo *>(lBase + aBuffer->mHeader->mPacketInfoOffset_byte);

        for (unsigned int i = 0; i < aBuffer->mHeader->mPacketQty; i++)
        {
            switch (lPacketInfo[i].mPacketState)
            {
            case OPEN_NET_PACKET_STATE_RECEIVED :
            case OPEN_NET_PACKET_STATE_RECEIVING:
                break;

            case OPEN_NET_PACKET_STATE_PROCESSED:
                lPacketInfo[i].mPacketState = OPEN_NET_PACKET_STATE_SENDING;
                // no break;

            case OPEN_NET_PACKET_STATE_SENDING:
                if (0 != (lPacketInfo[i].mToSendTo & lAdapterBit))
                {
                    mHardware->Packet_Send(aBuffer->mBufferInfo.mBuffer_PA + lPacketInfo[i].mPacketOffset_byte, lPacketInfo[i].mPacketSize_byte, &aBuffer->mSendCounter);
                }
                break;

            default: ASSERT(false);
            }
        }
    }

    // Level    SoftInt
    // Threada  Queue or SoftInt
    void Adapter::Buffers_Process()
    {
        for (unsigned int i = 0; i < mBufferCount; i++)
        {
            ASSERT(NULL != mBuffers[i].mHeader);

            switch (mBuffers[i].mHeader->mBufferState)
            {
            case OPEN_NET_BUFFER_STATE_PROCESSED:
                // TODO  Test
                Buffer_Send(mBuffers + i);
                break;

            case OPEN_NET_BUFFER_STATE_RECEIVING:
                // TODO  Test
                if (0 == mBuffers[i].mReceiveCounter)
                {
                    Buffer_Process(mBuffers + i);
                }
                break;

            case OPEN_NET_BUFFER_STATE_SENDING:
                if (0 == mBuffers[i].mSendCounter)
                {
                    if (mBuffers[i].mFlags.mMarkedForRetrieval)
                    {
                        // TODO  Test
                        Buffer_Retrieve();
                    }
                    else
                    {
                        Buffer_Receive(mBuffers + i);
                    }
                }
                break;

            case OPNE_NET_BUFFER_STATE_PROCESSING:
                // TODO  Test
                break;

            default: ASSERT(false);
            }
        }
    }

    void Adapter::Disconnect()
    {
        ASSERT(NULL != mAdapters);
        ASSERT(NULL != mEvent   );
        ASSERT(   0 != mSystemId);

        mAdapters  = NULL                       ;
        mAdapterNo = OPEN_NET_ADAPTER_NO_UNKNOWN;
        mEvent     = NULL                       ;
        mSystemId  =                           0;
    }

    int Adapter::IoCtl(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte)
    {
        int lResult;

        switch (aCode)
        {
        case OPEN_NET_IOCTL_BUFFER_QUEUE   : lResult = IoCtl_Buffer_Queue   (reinterpret_cast<const OpenNet_BufferInfo *>(aIn ), aInSize_byte ); break;
        case OPEN_NET_IOCTL_BUFFER_RETRIEVE: lResult = IoCtl_Buffer_Retrieve(reinterpret_cast<      OpenNet_BufferInfo *>(aOut), aOutSize_byte); break;
        case OPEN_NET_IOCTL_CONFIG_GET     : lResult = IoCtl_Config_Get     (reinterpret_cast<      OpenNet_Config     *>(aOut)); break;
        case OPEN_NET_IOCTL_CONFIG_SET     : lResult = IoCtl_Config_Set     (reinterpret_cast<const OpenNet_Config     *>(aIn ), reinterpret_cast<OpenNet_Config *>(aOut)); break;
        case OPEN_NET_IOCTL_CONNECT        : lResult = IoCtl_Connect        (reinterpret_cast<const OpenNet_Connect    *>(aIn )); break;
        case OPEN_NET_IOCTL_INFO_GET       : lResult = IoCtl_Info_Get       (reinterpret_cast<      OpenNet_Info       *>(aOut)); break;
        case OPEN_NET_IOCTL_PACKET_SEND    : lResult = IoCtl_Packet_Send    (                                             aIn  , aInSize_byte ); break;
        case OPEN_NET_IOCTL_STATE_GET      : lResult = IoCtl_State_Get      (reinterpret_cast<      OpenNet_State      *>(aOut)); break;
        case OPEN_NET_IOCTL_STATS_GET      : lResult = IoCtl_Stats_Get      (reinterpret_cast<      OpenNet_Stats      *>(aOut)); break;
        case OPEN_NET_IOCTL_STATS_RESET    : lResult = IoCtl_Stats_Reset    (); break;

        default: lResult = IOCTL_RESULT_INVALID_CODE;
        }

        mStats.mIoCtl++;
        mStats.mIoCtl_Last        = aCode  ;
        mStats.mIoCtl_Last_Result = lResult;

        return lResult;
    }

    unsigned int Adapter::IoCtl_InSize_GetMin(unsigned int aCode) const
    {
        unsigned int lResult;

        switch (aCode)
        {
        case OPEN_NET_IOCTL_BUFFER_QUEUE: lResult = sizeof(OpenNet_BufferInfo); break;
        case OPEN_NET_IOCTL_CONFIG_SET  : lResult = sizeof(OpenNet_Config    ); break;
        case OPEN_NET_IOCTL_CONNECT     : lResult = sizeof(OpenNet_Connect   ); break;

        default: lResult = 0;
        }

        return lResult;
    }

    unsigned int Adapter::IoCtl_OutSize_GetMin(unsigned int aCode) const
    {
        unsigned int lResult;

        switch (aCode)
        {
        case OPEN_NET_IOCTL_BUFFER_RETRIEVE: lResult = sizeof(OpenNet_BufferInfo); break;
        case OPEN_NET_IOCTL_CONFIG_GET     : lResult = sizeof(OpenNet_Config    ); break;
        case OPEN_NET_IOCTL_CONFIG_SET     : lResult = sizeof(OpenNet_Config    ); break;
        case OPEN_NET_IOCTL_INFO_GET       : lResult = sizeof(OpenNet_Info      ); break;
        case OPEN_NET_IOCTL_STATE_GET      : lResult = sizeof(OpenNet_State     ); break;
        case OPEN_NET_IOCTL_STATS_GET      : lResult = sizeof(OpenNet_Stats     ); break;

        default: lResult = 0;
        }

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

        memset(aHeader, 0, sizeof(lPacketOffset_byte));

        aHeader->mBufferState           = OPEN_NET_BUFFER_STATE_SENDING;
        aHeader->mPacketInfoOffset_byte = sizeof(OpenNet_BufferHeader);
        aHeader->mPacketQty             = lPacketQty;
        aHeader->mPacketSize_byte       = lPacketSize_byte;

        for (unsigned int i = 0; i < lPacketQty; i++)
        {
            lPacketInfo[i].mPacketOffset_byte = 0;
            lPacketInfo[i].mPacketState       = OPEN_NET_PACKET_STATE_SENDING;

            Skip64KByteBoundary(aBufferInfo.mBuffer_PA, &lPacketOffset_byte, lPacketSize_byte, &lPacketInfo[i].mPacketOffset_byte);
        }
    }

    // aBuffer [---;R--]
    //
    // Level   SoftInt
    // Thread  SoftInt
    //
    // TODO  Test
    void Adapter::Buffer_Process(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer         );
        ASSERT(NULL != aBuffer->mHeader);
        ASSERT(NULL != aBuffer->mMarker);

        aBuffer->mHeader->mBufferState = OPNE_NET_BUFFER_STATE_PROCESSING;

        (*aBuffer->mMarker) = OPEN_NET_MARKER_VALUE;
    }

    // aBufferInfo [---;R--]
    //
    // Level   SoftInt or Thread
    // Thread  Queue
    void Adapter::Buffer_Queue(const OpenNet_BufferInfo & aBufferInfo)
    {
        ASSERT(NULL != (&aBufferInfo));

        ASSERT(OPEN_NET_BUFFER_QTY > mBufferCount);

        memset(mBuffers + mBufferCount, 0, sizeof(mBuffers[mBufferCount]));

        mBuffers[mBufferCount].mBufferInfo = aBufferInfo;

        PHYSICAL_ADDRESS lPA;

        lPA.QuadPart = aBufferInfo.mBuffer_PA;

        unsigned int lSize_byte = sizeof(OpenNet_BufferHeader) + sizeof(OpenNet_PacketInfo) * aBufferInfo.mPacketQty;

        mBuffers[mBufferCount].mHeader = reinterpret_cast<OpenNet_BufferHeader *>(MmMapIoSpace(lPA, lSize_byte, MmNonCached));
        ASSERT(NULL != mBuffers[mBufferCount].mHeader);

        lPA.QuadPart = aBufferInfo.mBuffer_PA;

        mBuffers[mBufferCount].mMarker = reinterpret_cast<uint32_t *>(MmMapIoSpace(lPA, PAGE_SIZE, MmNonCached));
        ASSERT(NULL != mBuffers[mBufferCount].mMarker);

        Buffer_InitHeader(mBuffers[mBufferCount].mHeader, aBufferInfo);
        
        mBufferCount++;
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

            mHardware->Packet_Receive(aBuffer->mBufferInfo.mBuffer_PA + lPacketInfo[i].mPacketOffset_byte, lPacketInfo + i, &aBuffer->mReceiveCounter);
        }

        aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_RECEIVING;
    }

    // Level   SoftInt
    // Thread  SoftInt
    //
    // TODO  Test
    void Adapter::Buffer_Retrieve()
    {
        ASSERT(OPEN_NET_BUFFER_QTY >= mBufferCount           );
        ASSERT(mBufferCount        >  mBufferRetrieveCount   );
        ASSERT(NULL                != mBufferRetrieveOutput  );
        ASSERT(mBufferCount        >= mBufferRetrieveQty     );
        ASSERT(NULL                != mCompletePendingRequest);
        ASSERT(NULL                != mContext               );

        mBufferRetrieveCount++;

        if (mBufferRetrieveQty <= mBufferRetrieveCount)
        {
            unsigned int lOffset = mBufferCount - mBufferRetrieveQty;

            for (unsigned int i = lOffset; i < mBufferCount; i++)
            {
                mBufferRetrieveOutput[i - lOffset] = mBuffers[i].mBufferInfo;
            }

            mBufferCount = lOffset;

            CompletePendingRequest lCompletePendingRequest = mCompletePendingRequest;
            void                 * lContext                = mContext               ;

            mCompletePendingRequest = NULL;
            mContext                = NULL;

            lCompletePendingRequest(lContext, sizeof(OpenNet_BufferInfo) * mBufferRetrieveQty);
        }
    }

    // aBuffer [-K-;R--]
    //
    // Level   SoftInt
    // Thread  SoftInt
    //
    // TODO  Test
    void Adapter::Buffer_Send(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer);

        ASSERT(NULL != mAdapters);

        for (unsigned int i = 0; i < OPEN_NET_ADAPTER_NO_QTY; i++)
        {
            if (NULL != mAdapters[i])
            {
                mAdapters[i]->Buffer_SendPackets(aBuffer);
            }
        }
    }

    // ===== IoCtl ==========================================================

    int Adapter::IoCtl_Buffer_Queue(const OpenNet_BufferInfo * aIn, unsigned int aInSize_byte)
    {
        ASSERT(NULL                       != aIn         );
        ASSERT(sizeof(OpenNet_BufferInfo) <= aInSize_byte);

        ASSERT(OPEN_NET_BUFFER_QTY >= mBufferCount);

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

    int Adapter::IoCtl_Buffer_Retrieve(OpenNet_BufferInfo * aOut, unsigned int aOutSize_byte)
    {
        ASSERT(NULL                       != aOut         );
        ASSERT(sizeof(OpenNet_BufferInfo) <= aOutSize_byte);

        memset(aOut, 0, aOutSize_byte);

        unsigned int lCount = aOutSize_byte / sizeof(OpenNet_BufferInfo);

        if (mBufferCount < lCount)
        {
            lCount = mBufferCount;
        }

        mBufferRetrieveCount  = 0     ;
        mBufferRetrieveOutput = aOut  ;
        mBufferRetrieveQty    = lCount;

        for (unsigned int i = (mBufferCount - lCount); i < mBufferCount; i++)
        {
            mBuffers->mFlags.mMarkedForRetrieval = true;
        }

        return IOCTL_RESULT_PENDING;
    }

    int Adapter::IoCtl_Config_Get(OpenNet_Config * aOut)
    {
        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        mHardware->GetConfig(aOut);

        return sizeof(OpenNet_Config);
    }

    int Adapter::IoCtl_Config_Set(const OpenNet_Config * aIn, OpenNet_Config * aOut)
    {
        ASSERT(NULL != aIn );
        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        mHardware->SetConfig(*aIn);
        mHardware->GetConfig(aOut);

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

        return sizeof(OpenNet_Info);
    }

    int Adapter::IoCtl_Packet_Send(const void * aIn, unsigned int aInSize_byte)
    {
        ASSERT(NULL != mHardware);

        mHardware->Packet_Send(aIn, aInSize_byte);

        return IOCTL_RESULT_OK;
    }

    int Adapter::IoCtl_State_Get(OpenNet_State * aOut)
    {
        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        memset(aOut, 0, sizeof(OpenNet_State));

        aOut->mAdapterNo = mAdapterNo;
        aOut->mSystemId  = mSystemId ;

        mHardware->GetState(aOut);

        return sizeof(OpenNet_State);
    }

    int Adapter::IoCtl_Stats_Get(OpenNet_Stats * aOut) const
    {
        ASSERT(NULL != aOut);

        memcpy(aOut, &mStats, sizeof(mStats));

        mStats.mStats_Get++;
        
        return sizeof(mStats);
    }

    int Adapter::IoCtl_Stats_Reset()
    {
        memset(&mStats, 0, sizeof(mStats) / 2);

        mStats.mStats_Reset++;

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
void Skip64KByteBoundary(uint64_t aLogical, unsigned int * aOffset_byte, unsigned int aSize_byte, unsigned int * aOutOffset_byte)
{
    ASSERT(NULL       != aOffset_byte   );
    ASSERT(0          <  aSize_byte     );
    ASSERT(SIZE_64_KB >  aSize_byte     );
    ASSERT(NULL       != aOutOffset_byte);

    uint64_t lBegin = aLogical + (*aOffset_byte);
    uint64_t lEnd   = lBegin + aSize_byte - 1;

    if ((lBegin & SIZE_64_KB) == (lEnd & SIZE_64_KB))
    {
        (*aOutOffset_byte) = (*aOffset_byte);
    }
    else
    {
        uint64_t lOffset_byte = SIZE_64_KB - (lBegin % SIZE_64_KB);

        (*aOutOffset_byte) = (*aOffset_byte) + static_cast<unsigned int>(lOffset_byte);
    }

    (*aOffset_byte) = (*aOutOffset_byte) + aSize_byte;
}
