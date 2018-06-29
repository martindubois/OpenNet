
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Lib/Adapter.cpp

// TODO  ONK_Lib.Adapter  Move the IOCTL_RESULT_... constant into the
//                        common/OpenNetK/IoCtl.h file and use an enum

// TODO  ONK_Lib.Adapter  Move the IoCtlInfo type declaration into the
//                        common/OpenNetK/IoCtl.h

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
#include <OpenNetK/Debug.h>
#include <OpenNetK/Hardware.h>
#include <OpenNetK/SpinLock.h>

#include <OpenNetK/Adapter.h>

// ===== Common =============================================================
#include "../Common/Version.h"

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static void SkipDangerousBoundary(uint64_t aLogical, unsigned int * aOffset_byte, unsigned int aSize_byte, volatile unsigned int * aOutOffset_byte);

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

    // aZone0 [-K-;RW-]
    //
    // Level   Thread
    // Thread  Initialisation
    void Adapter::Init(SpinLock * aZone0)
    {
        ASSERT(NULL != aZone0);

        memset(&mStats        , 0, sizeof(mStats        ));
        memset(&mStats_NoReset, 0, sizeof(mStats_NoReset));

        mAdapters    = NULL;
        mAdapterNo   = OPEN_NET_ADAPTER_NO_UNKNOWN;
        mBufferCount =    0;
        mEvent       = NULL;
        mHardware    = NULL;
        mSystemId    =    0;
        mZone0       = aZone0;
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
        volatile uint8_t * lBase       = reinterpret_cast<volatile uint8_t *>(aBuffer->mHeader);

        volatile OpenNet_PacketInfo * lPacketInfo = reinterpret_cast<volatile OpenNet_PacketInfo *>(lBase + aBuffer->mHeader->mPacketInfoOffset_byte);

        for (unsigned int i = 0; i < aBuffer->mHeader->mPacketQty; i++)
        {
            ASSERT(0 < lPacketInfo[i].mPacketOffset_byte);

            switch (lPacketInfo[i].mPacketState)
            {
            case OPEN_NET_PACKET_STATE_RX_COMPLETED:
            case OPEN_NET_PACKET_STATE_RX_RUNNING  :
                // TODO  ONK_Lib.Adapter.PartialBuffer  Add statistics
                break;

            case OPEN_NET_PACKET_STATE_PX_COMPLETED:
                lPacketInfo[i].mPacketState = OPEN_NET_PACKET_STATE_TX_RUNNING;
                // no break;

            case OPEN_NET_PACKET_STATE_TX_RUNNING:
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
        ASSERT(NULL != mZone0);

        mZone0->Lock();

            ASSERT(OPEN_NET_BUFFER_QTY >= mBufferCount);

            for (unsigned int i = 0; i < mBufferCount; i++)
            {
                ASSERT(NULL != mBuffers[i].mHeader);

                switch (mBuffers[i].mHeader->mBufferState)
                {
                case OPEN_NET_BUFFER_STATE_PX_COMPLETED  : Buffer_PxCompleted_Zone0(mBuffers + i); break;
                case OPEN_NET_BUFFER_STATE_PX_RUNNING    : Buffer_PxRunning_Zone0  (mBuffers + i); break;
                case OPEN_NET_BUFFER_STATE_RX_PROGRAMMING:                                         break;
                case OPEN_NET_BUFFER_STATE_RX_RUNNING    : Buffer_RxRunning_Zone0  (mBuffers + i); break;
                case OPEN_NET_BUFFER_STATE_STOPPED       : Buffer_Stopped_Zone0    (           i); break;
                case OPEN_NET_BUFFER_STATE_TX_PROGRAMMING:                                         break;
                case OPEN_NET_BUFFER_STATE_TX_RUNNING    : Buffer_TxRunning_Zone0  (mBuffers + i); break;

                default:
                    // The buffer is clearly corrupted! We don't write to it
                    // and if possible we simply forget about it.

                    // TODO  ONK_Lib.Adapter  Add statistic counter
                    if (i == (mBufferCount - 1))
                    {
                        // TODO  ONK_Lib.Adapter  Add statistic counter
                        DbgPrintEx(DPFLTR_IHVDRIVER_ID, DEBUG_STATE_CHANGE, "%u %u Corrupted ==> Released" DEBUG_EOL, mAdapterNo, i);
                        mBufferCount--;
                    }
                }
            }

        mZone0->Unlock();

        mStats.mBuffers_Process++;
    }

    // AdapterNo may be OPEN_NET_ADAPTER_NO_UNKNOW if Disconnect is called
    // from Connect after an error occured.
    //
    // Level   Thread
    // Thread  User
    void Adapter::Disconnect()
    {
        ASSERT(NULL != mAdapters);
        ASSERT(NULL != mEvent   );
        ASSERT(   0 != mSystemId);
        ASSERT(NULL != mZone0   );

        mZone0->Lock();

            if (0 < mBufferCount)
            {
                Stop_Zone0();
            }

        mZone0->Unlock();

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
    void Adapter::Buffer_InitHeader_Zone0(volatile OpenNet_BufferHeader * aHeader, const OpenNet_BufferInfo & aBufferInfo)
    {
        ASSERT(NULL !=   aHeader               );
        ASSERT(NULL != (&aBufferInfo)          );
        ASSERT(   0 <    aBufferInfo.mPacketQty);

        ASSERT(NULL != mHardware);

        volatile OpenNet_PacketInfo * lPacketInfo      = reinterpret_cast<volatile OpenNet_PacketInfo *>(aHeader + 1);
        unsigned int                  lPacketQty       = aBufferInfo.mPacketQty;
        unsigned int                  lPacketSize_byte = mHardware->GetPacketSize();

        ASSERT(OPEN_NET_PACKET_SIZE_MAX_byte >= lPacketSize_byte);
        ASSERT(OPEN_NET_PACKET_SIZE_MIN_byte <= lPacketSize_byte);

        unsigned int lPacketOffset_byte = sizeof(OpenNet_BufferHeader) + (sizeof(OpenNet_PacketInfo) * lPacketQty);

        memset((OpenNet_BufferHeader *)(aHeader), 0, lPacketOffset_byte); // volatile_cast

        aHeader->mBufferState           = OPEN_NET_BUFFER_STATE_TX_RUNNING;
        aHeader->mPacketInfoOffset_byte = sizeof(OpenNet_BufferHeader);
        aHeader->mPacketQty             = lPacketQty;
        aHeader->mPacketSize_byte       = lPacketSize_byte;

        for (unsigned int i = 0; i < lPacketQty; i++)
        {
            lPacketInfo[i].mPacketState = OPEN_NET_PACKET_STATE_TX_RUNNING;

            SkipDangerousBoundary(aBufferInfo.mBuffer_PA, &lPacketOffset_byte, lPacketSize_byte, &lPacketInfo[i].mPacketOffset_byte);
        }

        mStats.mBuffer_InitHeader++;
    }

    // aBufferInfo [---;R--]
    //
    // Level   SoftInt or Thread
    // Thread  Queue
    void Adapter::Buffer_Queue_Zone0(const OpenNet_BufferInfo & aBufferInfo)
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

        Buffer_InitHeader_Zone0(mBuffers[mBufferCount].mHeader, aBufferInfo);
        
        mBufferCount++;

        mStats.mBuffer_Queue++;
    }

    // aBuffer [-K-;R--]
    //
    // Level    SoftInt
    // Threads  Queue or SoftInt
    void Adapter::Buffer_Receive_Zone0(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer                                 );
        ASSERT(NULL != aBuffer->mHeader                        );
        ASSERT(0    <  aBuffer->mHeader->mPacketInfoOffset_byte);
        ASSERT(0    <  aBuffer->mHeader->mPacketQty            );

        ASSERT(NULL != mHardware);
        ASSERT(NULL != mZone0   );

        volatile uint8_t * lBase = reinterpret_cast<volatile uint8_t *>(aBuffer->mHeader);

        volatile OpenNet_PacketInfo * lPacketInfo = reinterpret_cast<volatile OpenNet_PacketInfo *>(lBase + aBuffer->mHeader->mPacketInfoOffset_byte);

        mZone0->Unlock();

            for (unsigned int i = 0; i < aBuffer->mHeader->mPacketQty; i++)
            {
                ASSERT(0 < lPacketInfo[i].mPacketOffset_byte);

                mHardware->Packet_Receive(aBuffer->mBufferInfo.mBuffer_PA + lPacketInfo[i].mPacketOffset_byte, lPacketInfo + i, &aBuffer->mRx_Counter);
            }

        mZone0->Lock();

        mStats.mBuffer_Receive++;
    }

    // aBuffer [-K-;R--]
    //
    // Level   SoftInt
    // Thread  SoftInt
    void Adapter::Buffer_Send_Zone0(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer);

        ASSERT(NULL != mAdapters);
        ASSERT(NULL != mZone0   );

        mZone0->Unlock();

            for (unsigned int i = 0; i < OPEN_NET_ADAPTER_NO_QTY; i++)
            {
                if (NULL != mAdapters[i])
                {
                    mAdapters[i]->Buffer_SendPackets(aBuffer);
                }
            }

        mZone0->Unlock();

        mStats.mBuffer_Send++;
    }

    void Adapter::Buffer_WriteMarker_Zone0(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer         );
        ASSERT(NULL != aBuffer->mMarker);

        aBuffer->mMarkerValue++;

        (*aBuffer->mMarker) = aBuffer->mMarkerValue;
    }

    // Level   Thread or SoftInt
    // Thread  Queue or User
    void Adapter::Stop_Zone0()
    {
        ASSERT(0 < mBufferCount);

        for (unsigned int i = 0; i < mBufferCount; i++)
        {
            mBuffers[i].mFlags.mStopRequested = true;
        }
    }

    // ===== Buffer_State ===================================================
    // aBuffer [---;RW-]
    //
    // Level   SoftInt
    // Thread  SoftInt

    // TODO  ONK_Lib.Adapetr
    //       Verifier si les Interlocked sont necessaire
    void Adapter::Buffer_PxCompleted_Zone0(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer         );
        ASSERT(NULL != aBuffer->mHeader);

        if (NULL == mAdapters)
        {
            if (OPEN_NET_BUFFER_STATE_PX_COMPLETED == InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&aBuffer->mHeader->mBufferState), OPEN_NET_BUFFER_STATE_STOPPED, OPEN_NET_BUFFER_STATE_PX_COMPLETED))
            {
                DbgPrintEx(DPFLTR_IHVDRIVER_ID, DEBUG_STATE_CHANGE, "%u %p PX_COMPLETED ==> STOPPED" DEBUG_EOL, mAdapterNo, aBuffer);

                Buffer_WriteMarker_Zone0(aBuffer);
            }
        }
        else
        {
            // Here, we use a temporary state because Buffer_Send_Zone0
            // release the gate to avoid deadlock with the other adapter's
            // gates.
            if (OPEN_NET_BUFFER_STATE_PX_COMPLETED == InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&aBuffer->mHeader->mBufferState), OPEN_NET_BUFFER_STATE_TX_PROGRAMMING, OPEN_NET_BUFFER_STATE_PX_COMPLETED))
            {
                DbgPrintEx(DPFLTR_IHVDRIVER_ID, DEBUG_STATE_CHANGE, "%u %p PX_COMPLETED ==> TX_PROGRAMMING" DEBUG_EOL, mAdapterNo, aBuffer);

                Buffer_Send_Zone0(aBuffer);

                ASSERT(OPEN_NET_BUFFER_STATE_TX_PROGRAMMING == aBuffer->mHeader->mBufferState);

                DbgPrintEx(DPFLTR_IHVDRIVER_ID, DEBUG_STATE_CHANGE, "%u %p TX_PROGRAMMING ==> TX_RUNNING" DEBUG_EOL, mAdapterNo, aBuffer);
                aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_TX_RUNNING;
            }
        }
    }

    // We do not put assert on the buffer state because the GPU may change it
    // at any time.
    void Adapter::Buffer_PxRunning_Zone0(BufferInfo * aBuffer)
    {
        ASSERT(NULL != aBuffer         );
        ASSERT(NULL != aBuffer->mHeader);

        if (aBuffer->mFlags.mStopRequested)
        {
            DbgPrintEx(DPFLTR_IHVDRIVER_ID, DEBUG_STATE_CHANGE, "%u %p PX_RUNNING ==> STOPPED" DEBUG_EOL, mAdapterNo, aBuffer);
            aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_STOPPED;
        }
    }

    void Adapter::Buffer_RxRunning_Zone0(BufferInfo * aBuffer)
    {
        ASSERT(NULL                             != aBuffer                       );
        ASSERT(NULL                             != aBuffer->mHeader              );
        ASSERT(OPEN_NET_BUFFER_STATE_RX_RUNNING == aBuffer->mHeader->mBufferState);

        if (0 == aBuffer->mRx_Counter)
        {
            DbgPrintEx(DPFLTR_IHVDRIVER_ID, DEBUG_STATE_CHANGE, "%u %p RX_RUNNING ==> PX_RUNNING" DEBUG_EOL, mAdapterNo, aBuffer);
            aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_PX_RUNNING;
            Buffer_WriteMarker_Zone0(aBuffer);
        }
    }

    void Adapter::Buffer_Stopped_Zone0(unsigned int aIndex)
    {
        ASSERT(OPEN_NET_BUFFER_QTY > aIndex);

        ASSERT(OPEN_NET_BUFFER_QTY >= mBufferCount);

        if (aIndex == (mBufferCount - 1))
        {
            DbgPrintEx(DPFLTR_IHVDRIVER_ID, DEBUG_STATE_CHANGE, "%u %u STOPPED ==> Released" DEBUG_EOL, mAdapterNo, aIndex);
            mBufferCount--;
        }
    }

    void Adapter::Buffer_TxRunning_Zone0(BufferInfo * aBuffer)
    {
        ASSERT(NULL                             != aBuffer                       );
        ASSERT(NULL                             != aBuffer->mHeader              );
        ASSERT(OPEN_NET_BUFFER_STATE_TX_RUNNING == aBuffer->mHeader->mBufferState);

        if (0 == aBuffer->mTx_Counter)
        {
            if (aBuffer->mFlags.mStopRequested)
            {
                DbgPrintEx(DPFLTR_IHVDRIVER_ID, DEBUG_STATE_CHANGE, "%u %p TX_RUNNING ==> STOPPED" DEBUG_EOL, mAdapterNo, aBuffer);
                aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_STOPPED;

                Buffer_WriteMarker_Zone0(aBuffer);
            }
            else
            {
                // Here, we use a temporary state because Buffer_Receivd_Zone
                // release the gate to avoid deadlock with the Hardware's
                // gates.
                DbgPrintEx(DPFLTR_IHVDRIVER_ID, DEBUG_STATE_CHANGE, "%u %p TX_RUNNING ==> RX_PROGRAMMING" DEBUG_EOL, mAdapterNo, aBuffer);
                aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_RX_PROGRAMMING;

                Buffer_Receive_Zone0(aBuffer);

                ASSERT(OPEN_NET_BUFFER_STATE_RX_PROGRAMMING == aBuffer->mHeader->mBufferState);

                DbgPrintEx(DPFLTR_IHVDRIVER_ID, DEBUG_STATE_CHANGE, "%u %p RX_PROGRAMMING ==> RX_RUNNING" DEBUG_EOL, mAdapterNo, aBuffer);
                aBuffer->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_RX_RUNNING;
            }
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

        ASSERT(NULL != mZone0);

        mStats.mIoCtl_Start++;

        unsigned int lCount = aInSize_byte / sizeof(OpenNet_BufferInfo);

        int lResult;

        mZone0->Lock();

            ASSERT(OPEN_NET_BUFFER_QTY >= mBufferCount);

            if (OPEN_NET_BUFFER_QTY >= (mBufferCount + lCount))
            {
                for (unsigned int i = 0; i < lCount; i++)
                {
                    Buffer_Queue_Zone0(aIn[i]);
                }

                lResult = IOCTL_RESULT_PROCESSING_NEEDED;
            }
            else
            {
                lResult = IOCTL_RESULT_TOO_MANY_BUFFER;
            }

        mZone0->Unlock();

        return lResult;
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
        ASSERT(NULL != mZone0);

        int lResult;

        mZone0->Lock();

            if (0 < mBufferCount)
            {
                Stop_Zone0();

                lResult = IOCTL_RESULT_OK;
            }
            else
            {
                lResult = IOCTL_RESULT_NO_BUFFER;
            }

        mZone0->Unlock();

        mStats.mIoCtl_Stop++;

        return lResult;
    }

}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aOffset_byte    [---;RW-]
// aOutOffset_byte [---;-W-]
//
// Levels  SoftInt or Thread
// Thread  Queue
void SkipDangerousBoundary(uint64_t aLogical, unsigned int * aOffset_byte, unsigned int aSize_byte, volatile unsigned int * aOutOffset_byte)
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
