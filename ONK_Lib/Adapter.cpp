
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Lib/Adapter.cpp

#define __CLASS__     "Adapter::"
#define __NAMESPACE__ "OpenNetK::"

// CONFIG  _TRACE_BUFFER_STATE_CHANGE_
//         When defined, the driver trace the buffer state change.

// #define _TRACE_BUFFER_STATE_CHANGE_

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/OS.h>

#include <OpenNetK/Hardware.h>
#include <OpenNetK/Packet.h>
#include <OpenNetK/SpinLock.h>

#include <OpenNetK/Adapter.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/IoCtl.h"
#include "../Common/OpenNetK/Adapter_Statistics.h"
#include "../Common/Version.h"

// Macros
/////////////////////////////////////////////////////////////////////////////

#ifdef _TRACE_BUFFER_STATE_CHANGE_

    #ifdef _KMS_LINUX_
        #define TRACE_BUFFER_STATE_CHANGE(B,F,T) printk( KERN_INFO "%s - A%u B%u - %s ==> %s" DEBUG_EOL, __FUNCTION__, mAdapterNo, (B), (F), (T))
    #endif

    #ifdef _KMS_WINDOWS_
        #define TRACE_BUFFER_STATE_CHANGE(B,F,T) DbgPrintEx( DEBUG_ID, DEBUG_STATE_CHANGE, __FUNCTION__ " - A%u B%u - %s ==> %s" DEBUG_EOL, mAdapterNo, (B), (F), (T))
    #endif

#else
    #define TRACE_BUFFER_STATE_CHANGE(B,F,T)
#endif

// Constants
/////////////////////////////////////////////////////////////////////////////

// ===== Buffer state =======================================================

// See ONK_Lib/_DocDev/BufferStates.graphml
#define BUFFER_STATE_INVALID        (0)
#define BUFFER_STATE_EVENT_PENDING  (1)
#define BUFFER_STATE_PX_RUNNING     (2)
#define BUFFER_STATE_RX_PROGRAMMING (3)
#define BUFFER_STATE_RX_RUNNING     (4)
#define BUFFER_STATE_STOPPED        (5)
#define BUFFER_STATE_TX_PROGRAMMING (6)
#define BUFFER_STATE_TX_RUNNING     (7)
#define BUFFER_STATE_QTY            (8)

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static void SkipDangerousBoundary(uint64_t aIn_PA, unsigned int * aOffset_byte, unsigned int aSize_byte, volatile unsigned int * aOutOffset_byte);

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    bool Adapter::IoCtl_GetInfo(unsigned int aCode, OpenNetK_IoCtl_Info * aInfo)
    {
        // TRACE_DEBUG "%s( 0x%08x,  )" DEBUG_EOL, __FUNCTION__, aCode TRACE_END;

        ASSERT(NULL != aInfo);

        memset(aInfo, 0, sizeof(OpenNetK_IoCtl_Info));

        switch (aCode)
        {
        case IOCTL_CONFIG_GET      :
            aInfo->mOut_MinSize_byte = sizeof(Adapter_Config);
            break;
        case IOCTL_CONFIG_SET      :
            aInfo->mIn_MaxSize_byte  = sizeof(Adapter_Config);
            aInfo->mIn_MinSize_byte  = sizeof(Adapter_Config);
            aInfo->mOut_MinSize_byte = sizeof(Adapter_Config);
            break;
        case IOCTL_CONNECT         :
            aInfo->mIn_MaxSize_byte  = sizeof(IoCtl_Connect_In );
            aInfo->mIn_MinSize_byte  = sizeof(IoCtl_Connect_In );
            aInfo->mOut_MinSize_byte = sizeof(IoCtl_Connect_Out);
            break;
        case IOCTL_EVENT_WAIT      :
            aInfo->mIn_MaxSize_byte  = sizeof(IoCtl_Event_Wait_In);
            aInfo->mIn_MinSize_byte  = sizeof(IoCtl_Event_Wait_In);
            aInfo->mOut_MinSize_byte = sizeof(OpenNetK::Event[32]);

            #ifdef _KMS_LINUX_
                aInfo->mOut_MinSize_byte = sizeof(OpenNetK::Event[32]);
            #endif

            #ifdef _KMS_WINDOWS_
                aInfo->mOut_MinSize_byte = sizeof(OpenNetK::Event);
            #endif
            break;
        case IOCTL_INFO_GET        :
            aInfo->mOut_MinSize_byte = sizeof(Adapter_Info);
            break;
        case IOCTL_LICENSE_SET     :
            aInfo->mIn_MaxSize_byte  = sizeof(IoCtl_License_Set_In );
            aInfo->mIn_MinSize_byte  = sizeof(IoCtl_License_Set_In );
            aInfo->mOut_MinSize_byte = sizeof(IoCtl_License_Set_Out);
            break;
        case IOCTL_PACKET_SEND_EX  :
            aInfo->mIn_MaxSize_byte  = sizeof(IoCtl_Packet_Send_Ex_In) + PACKET_SIZE_MAX_byte;
            aInfo->mIn_MinSize_byte  = sizeof(IoCtl_Packet_Send_Ex_In);
            break;
        case IOCTL_PACKET_GENERATOR_CONFIG_GET:
            aInfo->mOut_MinSize_byte = sizeof(PacketGenerator_Config);
            break;
        case IOCTL_PACKET_GENERATOR_CONFIG_SET:
            aInfo->mIn_MaxSize_byte  = sizeof(PacketGenerator_Config);
            aInfo->mIn_MinSize_byte  = sizeof(PacketGenerator_Config);
            aInfo->mOut_MinSize_byte = sizeof(PacketGenerator_Config);
            break;
        case IOCTL_START           :
            aInfo->mIn_MaxSize_byte  = sizeof(Buffer) * OPEN_NET_BUFFER_QTY;
            aInfo->mIn_MinSize_byte  = sizeof(Buffer);
            break;
        case IOCTL_STATE_GET       :
            aInfo->mOut_MinSize_byte = sizeof(Adapter_State);
            break;
        case IOCTL_STATISTICS_GET  :
            aInfo->mIn_MaxSize_byte  = sizeof(IoCtl_Statistics_Get_In);
            aInfo->mIn_MinSize_byte  = sizeof(IoCtl_Statistics_Get_In);

            #ifdef _KMS_LINUX_
                aInfo->mOut_MinSize_byte = sizeof( uint32_t ) * 128;
            #endif
            break;

        case IOCTL_EVENT_WAIT_CANCEL     :
        case IOCTL_PACKET_DROP           :
        case IOCTL_PACKET_GENERATOR_START:
        case IOCTL_PACKET_GENERATOR_STOP :
        case IOCTL_STATISTICS_RESET      :
        case IOCTL_STOP                  :
        case IOCTL_TX_DISABLE            :
        case IOCTL_TX_ENABLE             :
            break;

        default : return false;
        }

        return true;
    }

    void Adapter::FileCleanup( void * aFileObject )
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT( NULL != aFileObject );

        if ( mPacketGenerator_FileObject == aFileObject )
        {
            mPacketGenerator_FileObject = NULL;
        }

        if ( mConnect_FileObject == aFileObject )
        {
            Disconnect();
        }
    }

    void Adapter::SetHardware(Hardware * aHardware)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aHardware);

        ASSERT(NULL == mHardware);

        mHardware = aHardware;
    }

    void Adapter::SetOSDep( OpenNetK_OSDep * aOSDep )
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aOSDep              );
        ASSERT(NULL != aOSDep->GetTimeStamp);

        ASSERT( NULL == mOSDep );

        mOSDep = aOSDep;

        mStatistics_Start_us = mOSDep->GetTimeStamp();
    }

    // CRITICAL PATH  BufferEvent
    unsigned int Adapter::Event_GetPendingCount() const
    {
        ASSERT( EVENT_QTY > mEvent_In  );
        ASSERT( EVENT_QTY > mEvent_Out );

        return ( mEvent_In + EVENT_QTY - mEvent_Out ) % EVENT_QTY;
    }

    void Adapter::Event_RegisterCallback(Adapter::Event_Callback aCallback, void * aContext)
    {
        ASSERT(NULL != aCallback);

        mEvent_Callback = aCallback;
        mEvent_Context  = aContext ;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    // aZone0 [-K-;RW-] The spinlock
    //
    // Level   Thread
    // Thread  Initialisation
    void Adapter::Init(SpinLock * aZone0)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aZone0);

        memset(&mPacketGenerator_Config        ,    0, sizeof(mPacketGenerator_Config));
        memset(&mPacketGenerator_Config.mPacket, 0xff,                               6);

        memset(&mStatistics, 0, sizeof(mStatistics));

        mEvent_Callback = NULL;
        mEvent_In       =    0;
        mEvent_Out      =    0;

        mPacketGenerator_Config.mAllowedIndexRepeat = REPEAT_COUNT_MAX;
        mPacketGenerator_Config.mPacketPer100ms     =                1;
        mPacketGenerator_Config.mPacketSize_byte    =             1024;
        mPacketGenerator_FileObject                 =             NULL;

        mAdapters            = NULL              ;
        mAdapterNo           = ADAPTER_NO_UNKNOWN;
        mBuffer.mCount       =                  0;
        mConnect_FileObject  = NULL              ;
        mHardware            = NULL              ;
        mStatistics_Start_us =                  0;
        mSystemId            =                  0;
        mZone0               = aZone0            ;
    }

    // aBuffer [-K-;RW-]
    //
    // Level  SoftInt

    // CRITICAL PATH  Interrupt.Tx
    //                1 to AdapterQty / buffer
    void Adapter::Buffer_SendPackets(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo                        );
        ASSERT(NULL != aBufferInfo->mBase_XA              );
        ASSERT(   0 <  aBufferInfo->mPacketInfoOffset_byte);
        ASSERT(NULL != aBufferInfo->mPackets              );
        ASSERT(   0 <  aBufferInfo->mBuffer.mPacketQty    );

        ASSERT(ADAPTER_NO_QTY >  mAdapterNo);
        ASSERT(NULL           != mHardware );

        if (!mHardware->Tx_IsEnabled())
        {
            return;
        }

        uint32_t  lAdapterBit = 1 << mAdapterNo;
        bool      lLocked     = false          ;

        OpenNet_PacketInfo * lPacketInfo_XA = reinterpret_cast<OpenNet_PacketInfo *>(aBufferInfo->mBase_XA + aBufferInfo->mPacketInfoOffset_byte);

            unsigned int lPacketQty = 0;

            for (unsigned int i = 0; i < aBufferInfo->mBuffer.mPacketQty; i++)
            {
                switch (aBufferInfo->mPackets[i].mState)
                {
                case Packet::STATE_RX_COMPLETED:
                    // TODO  ONK_Lib.Adapter
                    //       Normal (Optimisation) - Use burst

                    aBufferInfo->mPackets[i].mSendTo = lPacketInfo_XA[i].mSendTo;

                    // TODO  ONK_Lib.Adapter.PartialBuffer
                    //       Low (Feature)
                    if (0 == (OPEN_NET_PACKET_PROCESSED & aBufferInfo->mPackets[i].mSendTo))
                    {
                        mStatistics[ADAPTER_STATS_NOT_PROCESSED_packet]++;
                    }

                    aBufferInfo->mPackets[i].mState = Packet::STATE_TX_RUNNING;
                    // no break;

                case Packet::STATE_TX_RUNNING:
                    if (0 != (aBufferInfo->mPackets[i].mSendTo & lAdapterBit))
                    {
                        if (!lLocked)
                        {
                            // Locking the hardware may delay processing of
                            // other packets received by other network
                            // adapters or sent from application or packet
                            // generator. We only lock the hardware when we
                            // know the buffer contains packet to be send by
                            // this network adapter.
                            mHardware->Lock();
                            lLocked = true;
                        }

                        lPacketQty++;
                        mHardware->Packet_Send_NoLock(aBufferInfo->mPackets[i].GetData_PA(), aBufferInfo->mPackets[i].GetData_XA(), aBufferInfo->mPackets[i].GetSize(), &aBufferInfo->mTx_Counter);
                    }
                    break;

                default:
                    TRACE_ERROR "%s - aBufferInfo->mPackets[ %u ].mState = %d, lPacketInfo_XA[ %u ].mSendTo = 0x%08x\n", __FUNCTION__, i, aBufferInfo->mPackets[ i ].mState, i, lPacketInfo_XA[i].mSendTo TRACE_END;
                    ASSERT(false);
                }
            }

        if (lLocked)
        {
            mHardware->Unlock_AfterSend(&aBufferInfo->mTx_Counter, lPacketQty);
        }

        mStatistics[ADAPTER_STATS_BUFFER_SEND_PACKETS]++;
        mStatistics[ADAPTER_STATS_TX_packet          ]+= lPacketQty;
    }

    // AdapterNo may be OPEN_NET_ADAPTER_NO_UNKNOW if Disconnect is called
    // from Connect after an error occured.
    //
    // Level   Thread
    // Thread  User
    void Adapter::Disconnect()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mAdapters          );
        ASSERT(NULL != mConnect_FileObject);
        ASSERT(   0 != mSystemId          );
        ASSERT(NULL != mZone0             );

        uint32_t lFlags = mZone0->LockFromThread();

            if (0 < mBuffer.mCount)
            {
                Stop_Zone0();
            }

        mZone0->UnlockFromThread( lFlags );

        mAdapters[ mAdapterNo ] = NULL;

        // OpenNetK_OSDep::MapSharedMemory ==> OpenNetK_OSDep::UnmapSharedMemory  See IoCtl_Connect
        mOSDep->UnmapSharedMemory( mOSDep->mContext );

        mAdapters           = NULL              ;
        mAdapterNo          = ADAPTER_NO_UNKNOWN;
        mConnect_FileObject = NULL              ;
        mSystemId           =                  0;
    }

    // Level  SoftInt

    // CRITICAL PATH  Interrupt
    //                1 / hardware interrupt + 1 / tick
    void Adapter::Interrupt_Process2(bool * aNeedMoreProcessing)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mZone0);

        mZone0->Lock();

            ASSERT(OPEN_NET_BUFFER_QTY >= mBuffer.mCount);

            BufferCountAndIndex lBuffer;

            for (unsigned int i = 0; i < mBuffer.mCount; i ++)
            {
                lBuffer = mBuffer;

                Interrupt_Process2_Px_Zone0();
                Interrupt_Process2_Rx_Zone0();
                Interrupt_Process2_Tx_Zone0();

                if (0 < mBuffer.mCount)
                {
                    switch (mBuffers[mBuffer.mCount - 1].mState)
                    {
                    case BUFFER_STATE_STOPPED: Buffer_Release_Zone0(); break;

                    case BUFFER_STATE_EVENT_PENDING :
                    case BUFFER_STATE_PX_RUNNING    :
                    case BUFFER_STATE_RX_PROGRAMMING:
                    case BUFFER_STATE_RX_RUNNING    :
                    case BUFFER_STATE_TX_PROGRAMMING:
                    case BUFFER_STATE_TX_RUNNING    :
                        break;

                    default: ASSERT(false);
                    }
                }

                if ((lBuffer.mCount == mBuffer.mCount) && (lBuffer.mPx == mBuffer.mPx) && (lBuffer.mRx == mBuffer.mRx) && (lBuffer.mTx == mBuffer.mTx))
                {
                    break;
                }
            }

        mZone0->Unlock();

        // We cannot call the event callback while holding the mZone0 lock
        // because the callback is calling Adapter::IoCtl and Adapter::IoCtl
        // also acquire this lock. Also, calling the callback while
        // processing packet delay the packet processing. More, waiting here
        // help to return more than one event at the same time.
        if (mEvent_Pending)
        {
            if (NULL != mEvent_Callback)
            {
                mEvent_Callback(mEvent_Context);
            }

            mEvent_Pending = false;
        }

        if ( NULL != mPacketGenerator_FileObject )
        {
            // If the packet generator is running, we request the execution
            // of the third level of the interrupt processing. This level is
            // responsible for generating packet. This way, the packet
            // generation does not delay packet processing.
            (*aNeedMoreProcessing) = true;
        }

        mStatistics[ADAPTER_STATS_BUFFERS_PROCESS]++;
    }

    void Adapter::Interrupt_Process3()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mHardware                                   );
        ASSERT(   0 <  mPacketGenerator_Config.mAllowedIndexRepeat );
        ASSERT(   0 <  mPacketGenerator_Config.mPacketPer100ms     );
        ASSERT(  64 <= mPacketGenerator_Config.mPacketSize_byte    );

        unsigned int lRepeatCount = mPacketGenerator_Config.mAllowedIndexRepeat;

        if (mPacketGenerator_Config.mPacketPer100ms < lRepeatCount)
        {
            lRepeatCount = (mPacketGenerator_Config.mPacketPer100ms / 10) + 1;
        }

        mStatistics[ADAPTER_STATS_PACKET_GENERATOR_REPEAT_COUNT] = lRepeatCount;

        while (mPacketGenerator_Config.mPacketPer100ms > mPacketGenerator_Counter)
        {
            if (0 < mPacketGenerator_Config.mIndexOffset_byte)
            {
                (*reinterpret_cast<uint32_t *>(mPacketGenerator_Config.mPacket + mPacketGenerator_Config.mIndexOffset_byte))++;
            }

            if (!mHardware->Packet_Send(mPacketGenerator_Config.mPacket, mPacketGenerator_Config.mPacketSize_byte, lRepeatCount))
            {
                mStatistics[ADAPTER_STATS_PACKET_GENERATOR_BREAK] ++;
                break;
            }

            mPacketGenerator_Counter += lRepeatCount;

            mStatistics[ADAPTER_STATS_PACKET_GENERATOR_ITERATION] ++;
        }

        mStatistics[ADAPTER_STATS_INTERRUPT_PROCESS_3]++;
    }

    // aFileObject [-K-;---]
    // aIn         [--O;R--]
    // aOut        [--O;-W-]
    //
    // Return  See IOCTL_RESULT_...
    //
    // Level   SoftInt
    // Thread  Queue
    int Adapter::IoCtl( void * aFileObject, unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte)
    {
        // TRACE_DEBUG "%s( , 0x%08x, , %u bytes, , %u bytes )" DEBUG_EOL, __FUNCTION__, aCode, aInSize_byte, aOutSize_byte TRACE_END;

        (void)(aOutSize_byte);

        int lResult = IOCTL_RESULT_NOT_SET;

        switch (aCode)
        {
        case IOCTL_CONFIG_GET                 : lResult = IoCtl_Config_Get                (reinterpret_cast<Adapter_Config *>(aOut)); break;
        case IOCTL_INFO_GET                   : lResult = IoCtl_Info_Get                  (reinterpret_cast<Adapter_Info   *>(aOut)); break;

        case IOCTL_CONFIG_SET                 : lResult = IoCtl_Config_Set                (reinterpret_cast<const Adapter_Config         *>(aIn), reinterpret_cast<Adapter_Config         *>(aOut)); break;
        case IOCTL_PACKET_GENERATOR_CONFIG_SET: lResult = IoCtl_PacketGenerator_Config_Set(reinterpret_cast<const PacketGenerator_Config *>(aIn), reinterpret_cast<PacketGenerator_Config *>(aOut)); break;

        case IOCTL_CONNECT                    : lResult = IoCtl_Connect                   (aIn, aOut, aFileObject); break;

        case IOCTL_EVENT_WAIT                 : lResult = IoCtl_Event_Wait                (aIn, reinterpret_cast<Event    *>(aOut), aOutSize_byte); break;
        case IOCTL_STATISTICS_GET             : lResult = IoCtl_Statistics_Get            (aIn, reinterpret_cast<uint32_t *>(aOut), aOutSize_byte); break;

        case IOCTL_EVENT_WAIT_CANCEL          : lResult = IoCtl_Event_Wait_Cancel         (); break;
        case IOCTL_PACKET_DROP                : lResult = IoCtl_Packet_Drop               (); break;
        case IOCTL_PACKET_GENERATOR_STOP      : lResult = IoCtl_PacketGenerator_Stop      (); break;
        case IOCTL_STATISTICS_RESET           : lResult = IoCtl_Statistics_Reset          (); break;
        case IOCTL_STOP                       : lResult = IoCtl_Stop                      (); break;
        case IOCTL_TX_DISABLE                 : lResult = IoCtl_Tx_Disable                (); break;
        case IOCTL_TX_ENABLE                  : lResult = IoCtl_Tx_Enable                 (); break;

        case IOCTL_LICENSE_SET                : lResult = IoCtl_License_Set               (aIn, aOut); break;

        case IOCTL_PACKET_SEND_EX             : lResult = IoCtl_Packet_Send_Ex            (aIn, aInSize_byte ); break;

        case IOCTL_PACKET_GENERATOR_CONFIG_GET: lResult = IoCtl_PacketGenerator_Config_Get(reinterpret_cast<PacketGenerator_Config *>(aOut)); break;
        case IOCTL_STATE_GET                  : lResult = IoCtl_State_Get                 (reinterpret_cast<Adapter_State          *>(aOut)); break;

        case IOCTL_PACKET_GENERATOR_START     : lResult = IoCtl_PacketGenerator_Start     (aFileObject); break;

        case IOCTL_START                      : lResult = IoCtl_Start                     (reinterpret_cast<const Buffer *>(aIn ), aInSize_byte); break;

        default: ASSERT(false);
        }

        return lResult;
    }

    // CRITICAL PATH  Interrupt
    //                1 / tick
    void Adapter::Tick()
    {
        mPacketGenerator_Counter = 0;

        if (0 < mEvaluation_ms)
        {
            mEvaluation_ms -= 100;
        }
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    // aHeader_MA [---;-W-]
    // aBuffer    [---;R--]
    // aPackets   [---;-W-]
    //
    // Levels   SoftInt or Thread
    // Threads  Queue
    void Adapter::Buffer_InitHeader_Zone0(OpenNet_BufferHeader * aHeader_XA, const Buffer & aBuffer, Packet * aPackets)
    {
        // TRACE_DEBUG "%s( , ,  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL !=   aHeader_XA        );
        ASSERT(NULL != (&aBuffer)          );
        ASSERT(   0 <    aBuffer.mPacketQty);
        ASSERT(NULL !=   aPackets          );

        ASSERT(NULL != mHardware);

        uint8_t            * lBase_XA         = reinterpret_cast<uint8_t            *>(aHeader_XA    );
        OpenNet_PacketInfo * lPacketInfo_XA   = reinterpret_cast<OpenNet_PacketInfo *>(aHeader_XA + 1);
        unsigned int         lPacketQty       = aBuffer.mPacketQty;
        unsigned int         lPacketSize_byte = mHardware->GetPacketSize();

        ASSERT(PACKET_SIZE_MAX_byte >= lPacketSize_byte);
        ASSERT(PACKET_SIZE_MIN_byte <= lPacketSize_byte);

        unsigned int lPacketOffset_byte = sizeof(OpenNet_BufferHeader) + (sizeof(OpenNet_PacketInfo) * lPacketQty);

        memset(aHeader_XA, 0, lPacketOffset_byte);

        aHeader_XA->mEvents                = OPEN_NET_BUFFER_PROCESSED;
        aHeader_XA->mPacketInfoOffset_byte = sizeof(OpenNet_BufferHeader);
        aHeader_XA->mPacketQty             = lPacketQty;
        aHeader_XA->mPacketSize_byte       = lPacketSize_byte;

        for (unsigned int i = 0; i < lPacketQty; i++)
        {
            uint32_t lOffset_byte;

            SkipDangerousBoundary(aBuffer.mBuffer_PA, &lPacketOffset_byte, lPacketSize_byte, &lOffset_byte);

            aPackets[i].Init(aBuffer.mBuffer_PA + lOffset_byte, lBase_XA + lOffset_byte, lPacketInfo_XA + i);

            lPacketInfo_XA[i].mOffset_byte = lOffset_byte             ;
            lPacketInfo_XA[i].mSendTo      = OPEN_NET_PACKET_PROCESSED;
        }

        mStatistics[ADAPTER_STATS_BUFFER_INIT_HEADER] ++;
    }

    // aBuffer [---;R--]
    //
    // Return
    //  false  Error
    //  true   OK
    //
    // Level   SoftInt or Thread
    // Thread  Queue
    //
    // Buffer_Queue_Zone0 ==> Buffer_Release_Zone0
    bool Adapter::Buffer_Queue_Zone0(const Buffer & aBuffer)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT( NULL != ( & aBuffer )      );
        ASSERT(    0 <  aBuffer.mPacketQty );

        ASSERT( OPEN_NET_BUFFER_QTY >  mBuffer.mCount         );
        ASSERT( NULL                != mOSDep                 );
        ASSERT( NULL                != mOSDep->AllocateMemory );
        ASSERT( NULL                != mOSDep->FreeMemory     );
        ASSERT( NULL                != mOSDep->MapBuffer      );

        BufferInfo * lB = mBuffers + mBuffer.mCount;

        memset( lB, 0, sizeof( BufferInfo ) );

        lB->mBuffer                = aBuffer;
        lB->mPacketInfoOffset_byte = sizeof( OpenNet_BufferHeader );

        // MapBuffer ==> UnmapBuffer  See Buffer_Release_Zone0
        lB->mBase_XA = reinterpret_cast< uint8_t * >( mOSDep->MapBuffer( mOSDep->mContext, & lB->mBuffer.mBuffer_PA, lB->mBuffer.mBuffer_DA, lB->mBuffer.mSize_byte, lB->mBuffer.mMarker_PA, reinterpret_cast< volatile void * * >( & lB->mMarker_MA ) ) );
        if ( NULL == lB->mBase_XA )
        {
            return false;
        }

        lB->mHeader_XA = reinterpret_cast< OpenNet_BufferHeader * >( lB->mBase_XA );
        lB->mState     = BUFFER_STATE_TX_RUNNING;

        // AllocateMemory ==> FreeMemory  See Buffer_Release_Zone0
        lB->mPackets = reinterpret_cast< Packet * >( mOSDep->AllocateMemory( sizeof(Packet) * lB->mBuffer.mPacketQty ) );
        ASSERT( NULL != lB->mPackets );

        Buffer_InitHeader_Zone0( lB->mHeader_XA, lB->mBuffer, lB->mPackets );
        
        mBuffer.mCount++;

        mStatistics[ADAPTER_STATS_BUFFER_QUEUE] ++;

        return true;
    }

    // Buffer_Queue_Zone0 ==> Buffer_Release_Zone0
    void Adapter::Buffer_Release_Zone0()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(                   0 <  mBuffer.mCount      );
        ASSERT( OPEN_NET_BUFFER_QTY >= mBuffer.mCount      );
        ASSERT( NULL                != mOSDep              );
        ASSERT( NULL                != mOSDep->FreeMemory  );
        ASSERT( NULL                != mOSDep->UnmapBuffer );

        mBuffer.mCount--;

        BufferInfo * lB = mBuffers + mBuffer.mCount;

        ASSERT( NULL != lB->mBase_XA );
        ASSERT( NULL != lB->mPackets );

        // AllocateMemory ==> FreeMemory  See Buffer_Queue_Zone0
        mOSDep->FreeMemory( lB->mPackets );

        // MapBuffer ==> UnmapBuffer  See Buffer_Queue_zone0
        mOSDep->UnmapBuffer( mOSDep->mContext, lB->mBase_XA, lB->mBuffer.mSize_byte, lB->mMarker_MA );
    }

    // aBufferInfo [-K-;R--]
    //
    // Level  SoftInt

    // CRITICAL PATH  Interrupt.Rx
    //                1 / buffer
    void Adapter::Buffer_Receive_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo                        );
        ASSERT(NULL != aBufferInfo->mPackets              );
        ASSERT(NULL != aBufferInfo->mBuffer.mPacketQty    );

        ASSERT(NULL != mHardware);
        ASSERT(NULL != mZone0   );

        mZone0->Unlock();

            mHardware->Lock();

                for (unsigned int i = 0; i < aBufferInfo->mBuffer.mPacketQty; i++)
                {
                    mHardware->Packet_Receive_NoLock(aBufferInfo->mPackets + i, &aBufferInfo->mRx_Counter);
                }

            mHardware->Unlock_AfterReceive(&aBufferInfo->mRx_Counter, aBufferInfo->mBuffer.mPacketQty);

        mZone0->Lock();

        mStatistics[ADAPTER_STATS_BUFFER_RECEIVE] ++;
    }

    // aBufferInfo [-K-;R--]
    //
    // Level  SoftInt

    // CRITICAL PATH  Interrupt.Tx
    //                1 / buffer
    void Adapter::Buffer_Send_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo);

        ASSERT(NULL != mAdapters);
        ASSERT(NULL != mZone0   );

        mZone0->Unlock();

            // TODO  ONK_Lib.Adapter
            //       Normal (Optimisation) - Avoid walking all the list at
            //       all iteration

            for (unsigned int i = 0; i < ADAPTER_NO_QTY; i++)
            {
                if (NULL != mAdapters[i])
                {
                    mAdapters[i]->Buffer_SendPackets(aBufferInfo);
                }
            }

        mZone0->Lock();

        mStatistics[ADAPTER_STATS_BUFFER_SEND] ++;
    }

    // aBufferInfo [---;RW-]

    // CRITICAL PATH  Interrupt.Rx
    //                1 / buffer
    void Adapter::Buffer_WriteMarker_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo );

        aBufferInfo->mMarkerValue++;

        if ( NULL != aBufferInfo->mMarker_MA )
        {
            (*aBufferInfo->mMarker_MA) = aBufferInfo->mMarkerValue;
        }
    }

    // aType
    // aData32
    // aData64
    //
    // CRITICAL_PATH  BufferEvent  1 / Buffer event
    void Adapter::Event_Report_Zone0(OpenNetK::Event_Type aType, uint32_t aData)
    {
        ASSERT(OpenNetK::EVENT_TYPE_QTY > aType);

        ASSERT(EVENT_QTY > mEvent_In            );
        ASSERT(EVENT_QTY > mEvent_Out           );
        ASSERT(NULL      != mOSDep              );
        ASSERT(NULL      != mOSDep->GetTimeStamp);

        mEvents[mEvent_In].mData         = aData;
        mEvents[mEvent_In].mTimestamp_us = mOSDep->GetTimeStamp();
        mEvents[mEvent_In].mType         = aType;

        mEvent_In = (mEvent_In + 1) % EVENT_QTY;
        if (mEvent_Out == mEvent_In)
        {
            // The ring is full, we drop the oldest event
            mEvent_Out = (mEvent_Out + 1) % EVENT_QTY;
        }

        mEvent_Pending = true;
    }

    // CRITICAL PATH  Interrupt
    //                1 / buffer
    void Adapter::Interrupt_Process2_Px_Zone0()
    {
        if ( mBuffer.mCount > mBuffer.mPx )
        {
            switch ( mBuffers[ mBuffer.mPx ].mState )
            {
            case BUFFER_STATE_PX_RUNNING : Buffer_PxRunning_Zone0  ( mBuffers + mBuffer.mPx ); break;

            case BUFFER_STATE_EVENT_PENDING:
            case BUFFER_STATE_STOPPED:
                mBuffer.mPx = (mBuffer.mPx + 1) % mBuffer.mCount;
                break;

            case BUFFER_STATE_RX_PROGRAMMING :
            case BUFFER_STATE_RX_RUNNING     :
            case BUFFER_STATE_TX_PROGRAMMING :
            case BUFFER_STATE_TX_RUNNING     :
                break;

            default: ASSERT(false);
            }
        }
        else
        {
            mBuffer.mPx = 0;
        }
    }

    // CRITICAL PATH  Interrupt
    //                1 / buffer
    void Adapter::Interrupt_Process2_Rx_Zone0()
    {
        if ( mBuffer.mCount > mBuffer.mRx )
        {
            switch ( mBuffers[ mBuffer.mRx ].mState )
            {
            case BUFFER_STATE_RX_RUNNING : Buffer_RxRunning_Zone0( mBuffers + mBuffer.mRx ); break;

            case BUFFER_STATE_EVENT_PENDING:
            case BUFFER_STATE_STOPPED:
                mBuffer.mRx = (mBuffer.mRx + 1) % mBuffer.mCount;
                break;

            case BUFFER_STATE_PX_RUNNING     :
            case BUFFER_STATE_RX_PROGRAMMING :
            case BUFFER_STATE_TX_PROGRAMMING :
            case BUFFER_STATE_TX_RUNNING     :
                break;

            default: ASSERT(false);
            }
        }
        else
        {
            mBuffer.mRx = 0;
        }
    }

    // CRITICAL PATH  Interrupt
    //                1 / buffer
    void Adapter::Interrupt_Process2_Tx_Zone0()
    {
        if ( mBuffer.mCount > mBuffer.mTx )
        {
            switch ( mBuffers[ mBuffer.mTx ].mState )
            {
            case BUFFER_STATE_EVENT_PENDING: Buffer_EventPending_Zone0(mBuffers + mBuffer.mTx); break;
            case BUFFER_STATE_TX_RUNNING   : Buffer_TxRunning_Zone0   (mBuffers + mBuffer.mTx); break;

            case BUFFER_STATE_STOPPED:
                mBuffer.mTx = (mBuffer.mTx + 1) % mBuffer.mCount;
                break;

            case BUFFER_STATE_PX_RUNNING     :
            case BUFFER_STATE_RX_PROGRAMMING :
            case BUFFER_STATE_RX_RUNNING     :
            case BUFFER_STATE_TX_PROGRAMMING :
                break;

            default: ASSERT(false);
            }
        }
        else
        {
            mBuffer.mTx = 0;
        }
    }

    // Level   Thread or SoftInt
    // Thread  Queue or User
    void Adapter::Stop_Zone0()
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(0 < mBuffer.mCount);

        for (unsigned int i = 0; i < mBuffer.mCount; i++)
        {
            mBuffers[i].mFlags.mStopRequested = true;
        }
    }


    // ===== Buffer_ State ==================================================
    // aBufferInfo [---;R--]
    //
    // Level  SoftInt

    // CRITICAL PATH  Interrupt
    //                1 / buffer

    void Adapter::Buffer_EventPending_Zone0(BufferInfo * aBufferInfo)
    {
        ASSERT(NULL != aBufferInfo            );
        ASSERT(NULL != aBufferInfo->mHeader_XA);

        ASSERT(BUFFER_STATE_EVENT_PENDING == aBufferInfo->mState);

        ASSERT(mBuffer.mCount > mBuffer.mTx);

        if (aBufferInfo->mFlags.mStopRequested)
        {
            Buffer_Enter_Stopped_Zone0(aBufferInfo, mBuffer.mTx, "EVENT_PENDING");
        }
        else
        {
            aBufferInfo->mEvents = aBufferInfo->mHeader_XA->mEvents;
            if (0 != (OPEN_NET_BUFFER_RESERVED & aBufferInfo->mEvents))
            {
                TRACE_ERROR "Buffer_EventPending_Zone0 - A%u B%u - Corrupted" DEBUG_EOL, mAdapterNo, mBuffer.mTx TRACE_END;

                mStatistics[ADAPTER_STATS_CORRUPTED_BUFFER] ++;

                Buffer_Enter_Stopped_Zone0(aBufferInfo, mBuffer.mTx, "EVENT_PENDING");
            }
            else if (0 == (OPEN_NET_BUFFER_EVENT & aBufferInfo->mEvents))
            {
                Buffer_Enter_RxProgramming_Zone0(aBufferInfo, mBuffer.mTx, "EVENT_PENDING");
            }
        }

        mBuffer.mTx = (mBuffer.mTx + 1) % mBuffer.mCount;
    }

    void Adapter::Buffer_PxRunning_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo            );
        ASSERT(NULL != aBufferInfo->mHeader_XA);

        ASSERT(BUFFER_STATE_PX_RUNNING == aBufferInfo->mState);

        ASSERT( mBuffer.mCount > mBuffer.mPx );

        // We do not put assert on the buffer state because the GPU may
        // change it at any time.

        if (aBufferInfo->mFlags.mStopRequested)
        {
            Buffer_Enter_Stopped_Zone0(aBufferInfo, mBuffer.mPx, "PX_RUNNING");

            mBuffer.mPx = (mBuffer.mPx + 1) % mBuffer.mCount;
        }
        else
        {
            aBufferInfo->mEvents = aBufferInfo->mHeader_XA->mEvents;
            if (0 != (OPEN_NET_BUFFER_RESERVED & aBufferInfo->mEvents))
            {
                TRACE_ERROR "%s - A%u B%u - Corrupted" DEBUG_EOL, __FUNCTION__, mAdapterNo, mBuffer.mPx TRACE_END;

                mStatistics[ADAPTER_STATS_CORRUPTED_BUFFER] ++;

                Buffer_Enter_Stopped_Zone0(aBufferInfo, mBuffer.mPx, "PX_RUNNING");

                mBuffer.mPx = (mBuffer.mPx + 1) % mBuffer.mCount;
            }
            else if (0 != (OPEN_NET_BUFFER_PROCESSED & aBufferInfo->mEvents))
            {
                if (0 != (OPEN_NET_BUFFER_EVENT & aBufferInfo->mEvents))
                {
                    Event_Report_Zone0(OpenNetK::EVENT_TYPE_BUFFER, mBuffer.mPx);
                }

                if (NULL == mAdapters)
                {
                    Buffer_Enter_Stopped_Zone0(aBufferInfo, mBuffer.mPx, "PX_RUNNING");

                    Buffer_WriteMarker_Zone0(aBufferInfo);
                }
                else
                {
                    // Here, we use a temporary state because Buffer_Send_Zone0
                    // release the gate to avoid deadlock with the other adapter's
                    // gates.

                    TRACE_BUFFER_STATE_CHANGE(mBuffer.mPx, "PX_COMPLETED", "TX_PROGRAMMING");
                    aBufferInfo->mState = BUFFER_STATE_TX_PROGRAMMING;

                    Buffer_Send_Zone0(aBufferInfo);

                    ASSERT(BUFFER_STATE_TX_PROGRAMMING == aBufferInfo->mState);

                    TRACE_BUFFER_STATE_CHANGE(mBuffer.mPx, "TX_PROGRAMMING", "TX_RUNNING");
                    aBufferInfo->mState = BUFFER_STATE_TX_RUNNING;

                    Buffer_TxRunning_Zone0(aBufferInfo);
                }

                mBuffer.mPx = (mBuffer.mPx + 1) % mBuffer.mCount;
            }
        }
    }

    // TODO  ONK_Lib.Adapter
    //       Normal (Feature) - Ajouter la possibilite de remplacer le
    //       traitement OpenCL par un "Forward" fixe. Cela implique
    //       l'allocation de buffer dans la memoire de l'ordinateur par le
    //       pilote lui meme.

    // aBufferInfo [---;RW-]
    void Adapter::Buffer_RxRunning_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL                    != aBufferInfo             );
        ASSERT(NULL                    != aBufferInfo->mHeader_XA );
        ASSERT(BUFFER_STATE_RX_RUNNING == aBufferInfo->mState     );

        ASSERT( mBuffer.mCount > mBuffer.mRx );

        if (0 == aBufferInfo->mRx_Counter)
        {
            TRACE_BUFFER_STATE_CHANGE(mBuffer.mRx, "RX_RUNNING", "PX_RUNNING");
            aBufferInfo->mState = BUFFER_STATE_PX_RUNNING;

            aBufferInfo->mHeader_XA->mEvents = 0;
            Buffer_WriteMarker_Zone0(aBufferInfo);

            mBuffer.mRx = (mBuffer.mRx + 1) % mBuffer.mCount;
        }
    }

    // aBufferInfo [---;RW-]
    void Adapter::Buffer_TxRunning_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL                    != aBufferInfo             );
        ASSERT(NULL                    != aBufferInfo->mHeader_XA );
        ASSERT(BUFFER_STATE_TX_RUNNING == aBufferInfo->mState     );

        ASSERT( mBuffer.mCount > mBuffer.mTx );

        if (0 == aBufferInfo->mTx_Counter)
        {
            if (aBufferInfo->mFlags.mStopRequested || ((!mLicenseOk) && (0 >= mEvaluation_ms)))
            {
                Buffer_Enter_Stopped_Zone0(aBufferInfo, mBuffer.mTx, "TX_RUNNING");

                Buffer_WriteMarker_Zone0(aBufferInfo);
            }
            else
            {
                if (0 == (OPEN_NET_BUFFER_EVENT & aBufferInfo->mEvents))
                {
                    Buffer_Enter_RxProgramming_Zone0(aBufferInfo, mBuffer.mTx, "TX_RUNNING");
                }
                else
                {
                    TRACE_BUFFER_STATE_CHANGE(mBuffer.mTx, "TX_RUNNING", "EVENT_PENDING");
                    aBufferInfo->mState = BUFFER_STATE_EVENT_PENDING;
                }
            }

            mBuffer.mTx = (mBuffer.mTx + 1) % mBuffer.mCount;
        }
    }

    // ===== Buffer_Enter_ State ============================================
    // aBufferInfo
    // aIndex       Used for traces
    // aFrom        Used for traces
    //
    // Level  SoftInt

    // CRITICAL PATH  Interrupt.Rx
    void Adapter::Buffer_Enter_RxProgramming_Zone0(BufferInfo * aBufferInfo, unsigned int aIndex, const char * aFrom)
    {
        ASSERT(NULL != aBufferInfo);
        ASSERT(NULL != aFrom      );

        // Here, we use a temporary state because Buffer_Receivd_Zone release
        // the gate to avoid deadlock with the Hardware's gates.

        TRACE_BUFFER_STATE_CHANGE(aIndex, aFrom, "RX_PROGRAMMING");
        aBufferInfo->mState = BUFFER_STATE_RX_PROGRAMMING;

        Buffer_Receive_Zone0(aBufferInfo);

        ASSERT(BUFFER_STATE_RX_PROGRAMMING == aBufferInfo->mState);

        TRACE_BUFFER_STATE_CHANGE(aIndex, "RX_PROGRAMMING", "RX_RUNNING");
        aBufferInfo->mState = BUFFER_STATE_RX_RUNNING;

        (void)(aFrom );
        (void)(aIndex);
    }

    void Adapter::Buffer_Enter_Stopped_Zone0(BufferInfo * aBufferInfo, unsigned int aIndex, const char * aFrom)
    {
        ASSERT(NULL != aBufferInfo);
        ASSERT(NULL != aFrom      );

        TRACE_BUFFER_STATE_CHANGE(aIndex, aFrom, "STOPPED");
        aBufferInfo->mState = BUFFER_STATE_STOPPED;

        (void)(aFrom );
        (void)(aIndex);
    }

    // ===== IoCtl ==========================================================

    int Adapter::IoCtl_Config_Get(Adapter_Config * aOut)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        mHardware->GetConfig(aOut);

        mStatistics[ADAPTER_STATS_IOCTL_CONFIG_GET] ++;

        return sizeof(Adapter_Config);
    }

    int Adapter::IoCtl_Config_Set(const Adapter_Config * aIn, Adapter_Config * aOut)
    {
        // TRACE_DEBUG "%s( ,  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aIn );
        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        mHardware->SetConfig(*aIn);
        mHardware->GetConfig(aOut);

        mStatistics[ADAPTER_STATS_IOCTL_CONFIG_SET] ++;

        return sizeof(Adapter_Config);
    }

    int Adapter::IoCtl_Connect( const void * aIn, void * aOut, void * aFileObject )
    {
        ASSERT(NULL != aIn         );
        ASSERT(NULL != aOut        );
        ASSERT(NULL != aFileObject );

        mStatistics[ADAPTER_STATS_IOCTL_CONNECT] ++;

        const IoCtl_Connect_In  * lIn  = reinterpret_cast<const IoCtl_Connect_In  *>(aIn );
              IoCtl_Connect_Out * lOut = reinterpret_cast<      IoCtl_Connect_Out *>(aOut);

        if ( NULL == lIn->mSharedMemory )
        {
            TRACE_ERROR "%s - IOCTL_RESULT_INVALID_PARAMETER\n", __FUNCTION__ TRACE_END;
            return IOCTL_RESULT_INVALID_PARAMETER;
        }

        if (0 == lIn->mSystemId)
        {
            TRACE_ERROR "%s - IOCTL_RESULT_INVALID_SYSTEM_ID\n", __FUNCTION__ TRACE_END;
            return IOCTL_RESULT_INVALID_SYSTEM_ID;
        }

        if ( NULL != mAdapters )
        {
            TRACE_ERROR "%s - IOCTL_RESULT_ALREADY_CONNECTED\n", __FUNCTION__ TRACE_END;
            return IOCTL_RESULT_ALREADY_CONNECTED;
        }

        ASSERT(ADAPTER_NO_UNKNOWN == mAdapterNo);
        ASSERT(                 0 == mSystemId );

        mSystemId = lIn->mSystemId;

        // OpenNetK::MapShareadMemory ==> OpenNetK::UnmapSharedMemory  See Disconnect
        mAdapters = reinterpret_cast< OpenNetK::Adapter * * >( mOSDep->MapSharedMemory( mOSDep->mContext, lIn->mSharedMemory, SHARED_MEMORY_SIZE_byte ) );
        if ( NULL == mAdapters )
        {
            TRACE_ERROR "%s - OpenNetK_OSdep::MapSharedMemory( , ,  ) failed\n", __FUNCTION__ TRACE_END;
            return IOCTL_RESULT_SYSTEM_ERROR;
        }

        for (unsigned int i = 0; i < ADAPTER_NO_QTY; i++)
        {
            if (NULL == mAdapters[i])
            {
                mAdapters[i] = this;
                mAdapterNo = i;

                mConnect_FileObject = aFileObject;

                lOut->mAdapterNo = i;

                return sizeof(IoCtl_Connect_Out);
            }
        }

        Disconnect();

        TRACE_ERROR "%s - IOCTL_RESULT_TOO_MANY_ADAPTER\n", __FUNCTION__ TRACE_END;
        return IOCTL_RESULT_TOO_MANY_ADAPTER;
    }

    // CRITICAL PATH  BufferEvent
    int Adapter::IoCtl_Event_Wait(const void * aIn, OpenNetK::Event * aOut, unsigned int aOutSize_byte)
    {
        ASSERT(NULL                    != aIn          );
        ASSERT(NULL                    != aOut         );
        ASSERT(sizeof(OpenNetK::Event) <  aOutSize_byte);

        const IoCtl_Event_Wait_In * lIn           = reinterpret_cast<const IoCtl_Event_Wait_In *>(aIn);
        unsigned int                lOutSize_byte = aOutSize_byte;

        if (sizeof(OpenNetK::Event) > lIn->mOutputSize_byte)
        {
            TRACE_ERROR "IoCtl_Event_Wait - IOCTL_RESULT_INVALID_PARAMETER" DEBUG_EOL TRACE_END;
            return IOCTL_RESULT_INVALID_PARAMETER;
        }

        if (lIn->mOutputSize_byte < lOutSize_byte)
        {
            lOutSize_byte = lIn->mOutputSize_byte;
        }

        unsigned int lCount = lOutSize_byte / sizeof(OpenNetK::Event);
        unsigned int lIndex = 0;

        uint32_t lFlags = mZone0->LockFromThread();

            while (mEvent_In != mEvent_Out)
            {
                aOut[lIndex] = mEvents[mEvent_Out];

                mEvent_Out = (mEvent_Out + 1) % EVENT_QTY;

                lIndex++;

                if (lCount == lIndex)
                {
                    break;
                }
            }

        mZone0->UnlockFromThread(lFlags);

        return (0 == lIndex) ? IOCTL_RESULT_WAIT : (sizeof(OpenNetK::Event) * lIndex);
    }

    int Adapter::IoCtl_Event_Wait_Cancel()
    {
        ASSERT(NULL != mZone0);

        uint32_t lFlags = mZone0->LockFromThread();

            Event_Report_Zone0(OpenNetK::EVENT_TYPE_WAIT_CANCEL, 0);

        mZone0->UnlockFromThread(lFlags);

        if (mEvent_Pending)
        {
            if (NULL != mEvent_Callback)
            {
                mEvent_Callback(mEvent_Context);
            }

            mEvent_Pending = false;
        }

        return IOCTL_RESULT_OK;
    }

    int Adapter::IoCtl_Info_Get(Adapter_Info * aOut) const
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        mHardware->GetInfo(aOut);

        mStatistics[ADAPTER_STATS_IOCTL_INFO_GET] ++;

        return sizeof(Adapter_Info);
    }

    int Adapter::IoCtl_License_Set(const void * aIn, void * aOut)
    {
        TRACE_DEBUG "%s( ,  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aIn );
        ASSERT(NULL != aOut);

        const IoCtl_License_Set_In * lIn = reinterpret_cast<const IoCtl_License_Set_In *>(aIn);

        mHardware->GetInfo(&mInfo);

        uint64_t lKey = 26;

        lKey *= mInfo.mMaxSpeed_Mb_s;

        for (unsigned int i = 0; i < sizeof(mInfo.mEthernetAddress.mAddress); i++)
        {
            lKey +=  mInfo.mEthernetAddress.mAddress[i];
            lKey *= (mInfo.mEthernetAddress.mAddress[i] + 1);
        }

        lKey /= 1024;
        lKey &= 0xffffffff;

        TRACE_DEBUG "IoCtl_License_Set - 0x%08x == 0x%08x" DEBUG_EOL, lIn->mKey, lKey TRACE_END;

        mLicenseOk = (lIn->mKey == lKey);

        IoCtl_License_Set_Out * lOut = reinterpret_cast<IoCtl_License_Set_Out *>(aOut);

        memset(lOut, 0, sizeof(IoCtl_License_Set_Out));

        lOut->mFlags.mLicenseOk = mLicenseOk;

        return sizeof(IoCtl_License_Set_Out);
    }

    int Adapter::IoCtl_Packet_Drop()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT( NULL != mHardware );

        if ( ! mHardware->Packet_Drop() )
        {
            TRACE_ERROR "%s - Hardware::Packet_Drop() failed\n", __FUNCTION__ TRACE_END;
            return IOCTL_RESULT_CANNOT_DROP;
        }

        return IOCTL_RESULT_OK;
    }

    int Adapter::IoCtl_Packet_Send_Ex(const void * aIn, unsigned int aInSize_byte)
    {
        // TRACE_DEBUG "%s( , %u aInSize_byte )" DEBUG_EOL, __FUNCTION__, aInSize_byte TRACE_END;

        ASSERT(NULL                            != aIn         );
        ASSERT(sizeof(IoCtl_Packet_Send_Ex_In) <= aInSize_byte);

        const IoCtl_Packet_Send_Ex_In * lIn = reinterpret_cast<const IoCtl_Packet_Send_Ex_In *>(aIn);

        if (   (               0 >= lIn->mRepeatCount )
            || (REPEAT_COUNT_MAX <  lIn->mRepeatCount ) )
        {
            TRACE_ERROR "%s - IOCTL_RESULT_INVALID_PARAMETER\n", __FUNCTION__ TRACE_END;
            return IOCTL_RESULT_INVALID_PARAMETER;
        }

        unsigned int lSize_byte = aInSize_byte - sizeof(IoCtl_Packet_Send_Ex_In);
        if ( lIn->mSize_byte < lSize_byte )
        {
            lSize_byte = lIn->mSize_byte;
        }

        if (!mHardware->Packet_Send(lIn + 1, lSize_byte, lIn->mRepeatCount))
        {
            TRACE_ERROR "IoCtl_Packet_Send_Ex - Hardware::PacketSend( , %u byte, %u ) failed\n", lSize_byte, lIn->mRepeatCount TRACE_END;
            return IOCTL_RESULT_CANNOT_SEND;
        }

        mStatistics[ADAPTER_STATS_IOCTL_PACKET_SEND] += lIn->mRepeatCount;

        return IOCTL_RESULT_OK;
    }

    int Adapter::IoCtl_PacketGenerator_Config_Get(PacketGenerator_Config * aOut)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aOut);

        memcpy(aOut, &mPacketGenerator_Config, sizeof(mPacketGenerator_Config));

        return sizeof(mPacketGenerator_Config);
    }

    int Adapter::IoCtl_PacketGenerator_Config_Set(const PacketGenerator_Config * aIn, PacketGenerator_Config * aOut)
    {
        // TRACE_DEBUG "%s( ,  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT( NULL != aIn  );
        ASSERT( NULL != aOut );

        ASSERT( NULL != mHardware );

        memcpy(&mPacketGenerator_Config, aIn, sizeof(mPacketGenerator_Config));

        unsigned int lPacketSize_byte = mHardware->GetPacketSize();

        if      (               0 >= mPacketGenerator_Config.mAllowedIndexRepeat) { mPacketGenerator_Config.mAllowedIndexRepeat =                1; }
        else if (REPEAT_COUNT_MAX <  mPacketGenerator_Config.mAllowedIndexRepeat) { mPacketGenerator_Config.mAllowedIndexRepeat = REPEAT_COUNT_MAX; }

        if (0 >= mPacketGenerator_Config.mPacketPer100ms) { mPacketGenerator_Config.mPacketPer100ms = 1; }

        if      (              64 > mPacketGenerator_Config.mPacketSize_byte) { mPacketGenerator_Config.mPacketSize_byte =               64; }
        else if (lPacketSize_byte < mPacketGenerator_Config.mPacketSize_byte) { mPacketGenerator_Config.mPacketSize_byte = lPacketSize_byte; }

        if (mPacketGenerator_Config.mPacketSize_byte < (mPacketGenerator_Config.mIndexOffset_byte + sizeof(uint32_t))) { mPacketGenerator_Config.mIndexOffset_byte = 0; }

        memcpy(aOut, &mPacketGenerator_Config, sizeof(mPacketGenerator_Config));

        return sizeof(mPacketGenerator_Config);
    }

    int Adapter::IoCtl_PacketGenerator_Start( void * aFileObject )
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT( NULL != aFileObject );

        if ( NULL != mPacketGenerator_FileObject )
        {
            TRACE_ERROR "%s - IOCTL_RESULT_RUNNING\n", __FUNCTION__ TRACE_END;
            return IOCTL_RESULT_RUNNING;
        }

        mPacketGenerator_Counter    =           0;
        mPacketGenerator_Pending    =           0;
        mPacketGenerator_FileObject = aFileObject;

        if (0 < mPacketGenerator_Config.mIndexOffset_byte)
        {
            (*reinterpret_cast<uint32_t *>(mPacketGenerator_Config.mPacket + mPacketGenerator_Config.mIndexOffset_byte)) = 0;
        }

        return IOCTL_RESULT_OK;
    }

    int Adapter::IoCtl_PacketGenerator_Stop()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        if ( NULL == mPacketGenerator_FileObject )
        {
            TRACE_ERROR "%s - IOCTL_RESULT_STOPPED\n", __FUNCTION__ TRACE_END;
            return IOCTL_RESULT_STOPPED;
        }

        mPacketGenerator_FileObject = NULL;

        return IOCTL_RESULT_OK;
    }

    int Adapter::IoCtl_Start(const Buffer * aIn, unsigned int aInSize_byte)
    {
        // TRACE_DEBUG "%s( , %u bytes )" DEBUG_EOL, __FUNCTION__, aInSize_byte TRACE_END;

        ASSERT(NULL                     != aIn         );
        ASSERT(sizeof(OpenNetK::Buffer) <= aInSize_byte);

        ASSERT(NULL != mZone0);

        unsigned int lCount = aInSize_byte / sizeof(Buffer);

        OpenNetK_IoCtl_Result lResult;

        uint32_t lFlags = mZone0->LockFromThread();

            // TODO  ONK_Lib.Adapter
            //       Normal (Feature) - Refuse to start if the number of
            //       buffer may cause a descriptor overrun.

            ASSERT(OPEN_NET_BUFFER_QTY >= mBuffer.mCount);

            mBuffer.mPx = 0;
            mBuffer.mRx = 0;
            mBuffer.mTx = 0;

            if (OPEN_NET_BUFFER_QTY >= (mBuffer.mCount + lCount))
            {
                for (unsigned int i = 0; i < lCount; i++)
                {
                    if ( ( NULL == aIn[ i ].mBuffer_DA ) && ( 0 == aIn[ i ].mBuffer_PA ) )
                    {
                        break;
                    }

                    if ( ! Buffer_Queue_Zone0(aIn[i]) )
                    {
                        while ( 0 < mBuffer.mCount )
                        {
                            Buffer_Release_Zone0();
                        }

                        TRACE_ERROR "%s - IOCTL_RESULT_CANNOT_MAP_BUFFER\n", __FUNCTION__ TRACE_END;
                        lResult = IOCTL_RESULT_CANNOT_MAP_BUFFER;
                        break;
                    }
                }

                if ( 0 < mBuffer.mCount )
                {
                    if (!mLicenseOk)
                    {
                        mEvaluation_ms = 5 * 60 * 1000; // 5 minutes
                    }

                    lResult = IOCTL_RESULT_PROCESSING_NEEDED;
                }
                else
                {
                    TRACE_ERROR "%s - IOCTL_RESULT_NO_BUFFER\n", __FUNCTION__ TRACE_END;
                    lResult = IOCTL_RESULT_NO_BUFFER;
                }
            }
            else
            {
                TRACE_ERROR "%s - IOCTL_RESULT_TOO_MANY_BUFFER\n", __FUNCTION__ TRACE_END;
                lResult = IOCTL_RESULT_TOO_MANY_BUFFER;
            }

        mZone0->UnlockFromThread( lFlags );

        return lResult;
    }

    int Adapter::IoCtl_State_Get(Adapter_State * aOut)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        memset(aOut, 0, sizeof(Adapter_State));

        aOut->mAdapterNo     = mAdapterNo    ;
        aOut->mBufferCount   = mBuffer.mCount;
        aOut->mEvaluation_ms = mEvaluation_ms;
        aOut->mSystemId      = mSystemId     ;

        aOut->mFlags.mLicenseOk  = mLicenseOk;
        aOut->mFlags.mTx_Enabled = mHardware->Tx_IsEnabled();

        mHardware->GetState(aOut);

        mStatistics[ADAPTER_STATS_IOCTL_STATE_GET] ++;

        return sizeof(Adapter_State);
    }

    int Adapter::IoCtl_Statistics_Get(const void * aIn, uint32_t * aOut, unsigned int aOutSize_byte) const
    {
        // TRACE_DEBUG "%s( , , %u bytes )" DEBUG_EOL, __FUNCTION__, aOutSize_byte TRACE_END;

        ASSERT(NULL != aIn);

        ASSERT(NULL != mHardware           );
        ASSERT(NULL != mOSDep              );
        ASSERT(NULL != mOSDep->GetTimeStamp);

        const IoCtl_Statistics_Get_In * lIn = reinterpret_cast<const IoCtl_Statistics_Get_In *>(aIn);

        uint32_t                 * lOut          = aOut;
        unsigned int               lOutSize_byte = aOutSize_byte;
        int                        lResult_byte  = 0;

        bool lReset = lIn->mFlags.mReset;

        if ( lIn->mOutputSize_byte < lOutSize_byte )
        {
            lOutSize_byte = lIn->mOutputSize_byte;
        }

        uint64_t lNow_us = mOSDep->GetTimeStamp();

        mStatistics[ADAPTER_STATS_RUNNING_TIME_ms] = static_cast<unsigned int>((lNow_us - mStatistics_Start_us) / 1000);

        if (sizeof(mStatistics) <= lOutSize_byte)
        {
            memcpy(lOut, &mStatistics, sizeof(mStatistics));
            lResult_byte  += sizeof(mStatistics);
            lOut          += sizeof(mStatistics) / sizeof(uint32_t);
            lOutSize_byte -= sizeof(mStatistics);
        }
        else
        {
            if (0 < lOutSize_byte)
            {
                memcpy(lOut, &mStatistics, lOutSize_byte);
                lResult_byte += lOutSize_byte;
                lOut          = NULL;
                lOutSize_byte = 0;
            }
        }

        if (lReset)
        {
            mStatistics_Start_us = lNow_us;

            memset(&mStatistics, 0, ADAPTER_STATS_RESET_QTY * sizeof(uint32_t));

            mStatistics[ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET] ++;
        }

        lResult_byte += mHardware->Statistics_Get(lOut, lOutSize_byte, lReset);

        mStatistics[ADAPTER_STATS_IOCTL_STATISTICS_GET] ++;
        
        return lResult_byte;
    }

    int Adapter::IoCtl_Statistics_Reset()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mHardware);

        mStatistics_Start_us = mOSDep->GetTimeStamp();

        memset(&mStatistics, 0, ADAPTER_STATS_RESET_QTY * sizeof( uint32_t ) );

        mHardware->Statistics_Reset();

        mStatistics[ADAPTER_STATS_IOCTL_STATISTICS_RESET] ++;

        return IOCTL_RESULT_OK;
    }

    int Adapter::IoCtl_Stop()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mZone0);

        OpenNetK_IoCtl_Result lResult;

        uint32_t lFlags = mZone0->LockFromThread();

            mEvent_In  = 0;
            mEvent_Out = 0;

            if (0 < mBuffer.mCount)
            {
                Stop_Zone0();

                lResult = IOCTL_RESULT_OK;
            }
            else
            {
                TRACE_ERROR "%s - IOCTL_RESULT_NO_BUFFER\n", __FUNCTION__ TRACE_END;
                lResult = IOCTL_RESULT_NO_BUFFER;
            }

        mZone0->UnlockFromThread( lFlags );

        return lResult;
    }

    // TODO  ONK_Lib.Adapter
    //       Normal (Feature) - Re-enable Tx when the disabler disconnect
    int Adapter::IoCtl_Tx_Disable()
    {
        ASSERT(NULL != mHardware);

        mHardware->Tx_Disable();

        return IOCTL_RESULT_OK;
    }

    int Adapter::IoCtl_Tx_Enable()
    {
        ASSERT(NULL != mHardware);

        mHardware->Tx_Enable();

        return IOCTL_RESULT_OK;
    }

}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aIn_PA          [---;---]
// aOffset_byte    [---;RW-]
// aOutOffset_byte [---;-W-]
//
// Levels  SoftInt or Thread
// Thread  Queue
void SkipDangerousBoundary(uint64_t aIn_PA, unsigned int * aOffset_byte, unsigned int aSize_byte, volatile unsigned int * aOutOffset_byte)
{
    ASSERT(NULL                                  != aOffset_byte   );
    ASSERT(                                    0 <  aSize_byte     );
    ASSERT(OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte >  aSize_byte     );
    ASSERT(NULL                                  != aOutOffset_byte);

    uint64_t lBegin_PA = aIn_PA + (*aOffset_byte);
    uint64_t lEnd_PA   = lBegin_PA + aSize_byte - 1;

    if ((lBegin_PA & OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte) == (lEnd_PA & OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte))
    {
        (*aOutOffset_byte) = (*aOffset_byte);
    }
    else
    {
        uint64_t lOffset_byte = OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte - (lBegin_PA % OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte);

        (*aOutOffset_byte) = (*aOffset_byte) + static_cast<unsigned int>(lOffset_byte);
    }

    (*aOffset_byte) = (*aOutOffset_byte) + aSize_byte;
}
