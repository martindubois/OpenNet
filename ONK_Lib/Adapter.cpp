
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Lib/Adapter.cpp

#define __CLASS__     "Adapter::"
#define __NAMESPACE__ "OpenNetK::"

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/OS.h>
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Constants.h>
#include <OpenNetK/Debug.h>
#include <OpenNetK/Hardware.h>
#include <OpenNetK/IoCtl.h>
#include <OpenNetK/Packet.h>
#include <OpenNetK/SpinLock.h>

#include <OpenNetK/Adapter.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/IoCtl.h"
#include "../Common/OpenNetK/Adapter_Statistics.h"
#include "../Common/Version.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define TAG 'LKNO'

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static void SkipDangerousBoundary(uint64_t aLogical, unsigned int * aOffset_byte, unsigned int aSize_byte, volatile unsigned int * aOutOffset_byte);

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
            aInfo->mIn_MaxSize_byte  = sizeof(IoCtl_Connect_In);
            aInfo->mIn_MinSize_byte  = sizeof(IoCtl_Connect_In);
            break;
        case IOCTL_INFO_GET        :
            aInfo->mOut_MinSize_byte = sizeof(Adapter_Info);
            break;
        case IOCTL_PACKET_SEND     :
            aInfo->mIn_MaxSize_byte  = PACKET_SIZE_MAX_byte;
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

        case IOCTL_PACKET_DROP           :
        case IOCTL_PACKET_GENERATOR_START:
        case IOCTL_PACKET_GENERATOR_STOP :
        case IOCTL_STATISTICS_RESET:
        case IOCTL_STOP            :
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
            mPacketGenerator_FileObject == NULL;
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

        ASSERT( NULL != aOSDep );

        ASSERT( NULL == mOSDep );

        mOSDep = aOSDep;
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

        mPacketGenerator_Config.mAllowedIndexRepeat = REPEAT_COUNT_MAX;
        mPacketGenerator_Config.mPacketPer100ms     =                1;
        mPacketGenerator_Config.mPacketSize_byte    =             1024;
        mPacketGenerator_FileObject                 =             NULL;

        mAdapters    = NULL;
        mAdapterNo   = ADAPTER_NO_UNKNOWN;
        mBufferCount =    0;
        mConnect_FileObject = NULL;
        mHardware    = NULL;
        mSystemId    =    0;
        mZone0       = aZone0;

        #ifdef _KMS_WINDOWS_

            KeQuerySystemTimePrecise(&mStatistics_Start);

        #endif
    }

    // aBuffer [-K-;RW-]
    //
    // CRITICAL PATH - Buffer
    //
    // Level   SoftInt
    // Thread  SoftInt
    void Adapter::Buffer_SendPackets(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo                                 );
        ASSERT(NULL != aBufferInfo->mBase                          );
        ASSERT(   0 <  aBufferInfo->mPacketInfoOffset_byte         );
        ASSERT(NULL != aBufferInfo->mPackets                       );
        ASSERT(   0 <  aBufferInfo->mBuffer.mPacketQty             );

        ASSERT(ADAPTER_NO_QTY >  mAdapterNo);
        ASSERT(NULL           != mHardware );

        uint32_t  lAdapterBit = 1 << mAdapterNo;
        bool      lLocked     = false          ;

        OpenNet_PacketInfo * lPacketInfo = reinterpret_cast<OpenNet_PacketInfo *>(aBufferInfo->mBase + aBufferInfo->mPacketInfoOffset_byte);

            unsigned int lPacketQty = 0;

            for (unsigned int i = 0; i < aBufferInfo->mBuffer.mPacketQty; i++)
            {
                switch (aBufferInfo->mPackets[i].mState)
                {
                case Packet::STATE_RX_COMPLETED:
                    // TODO  ONK_Lib.Adapter
                    //       Normal (Performance) - Use burst

                    // TODO  ONK_Lib.Adapter
                    //       Normal (Performance) - Also cache mSize_byte
                    //       (and use burst too)
                    aBufferInfo->mPackets[i].mSendTo = lPacketInfo[i].mSendTo; // Reading DirectGMA buffer !!!

                    // TODO  OpenNetK.Adapter.PartialBuffer
                    //       Low (Feature)
                    ASSERT(0 != (OPEN_NET_PACKET_PROCESSED & aBufferInfo->mPackets[i].mSendTo));

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
                        mHardware->Packet_Send_NoLock(aBufferInfo->mBuffer.mBuffer_PA + aBufferInfo->mPackets[i].GetOffset(), aBufferInfo->mPackets[i].GetVirtualAddress(), lPacketInfo[i].mSize_byte, &aBufferInfo->mTx_Counter); // Reading DirectGMA buffer !!!
                    }
                    break;

                default:
                    // TRACE_DEBUG "%s - aBufferInfo->mPackets[ i ].mState = %d\n", __FUNCTION__, aBufferInfo->mPackets[ i ].mState TRACE_END;
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

    // CRITICAL PATH
    //
    // Level    SoftInt
    // Threads  Queue or SoftInt

    // TODO  OpenNetK.Adapter.TxOrder
    //       High - Les paquets doivent etre transmis dans l'ordre de
    //       reception.
    void Adapter::Buffers_Process(bool * aNeedMoreProcessing)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mZone0);

        mZone0->Lock();

            ASSERT(OPEN_NET_BUFFER_QTY >= mBufferCount);

            for (unsigned int i = 0; i < mBufferCount; i++)
            {
                ASSERT(NULL != mBuffers[i].mHeader);

                switch (mBuffers[i].mHeader->mBufferState) // Reading DirectGMA buffer !!!
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

                    mStatistics[ADAPTER_STATS_CORRUPTED_BUFFER]++;

                    if (i == (mBufferCount - 1))
                    {
                        TRACE_DEBUG "%s - %u %u Corrupted ==> Released" DEBUG_EOL, __FUNCTION__, mAdapterNo, i TRACE_END;

                        mStatistics[ ADAPTER_STATS_CORRUPTED_BUFFER_RELEASED ] ++;

                        Buffer_Release_Zone0();
                    }
                }
            }

        mZone0->Unlock();

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

    // AdapterNo may be OPEN_NET_ADAPTER_NO_UNKNOW if Disconnect is called
    // from Connect after an error occured.
    //
    // Level   Thread
    // Thread  User
    void Adapter::Disconnect()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mAdapters);
        ASSERT( NULL != mConnect_FileObject );
        ASSERT(   0 != mSystemId);
        ASSERT(NULL != mZone0   );

        mZone0->Lock();

            if (0 < mBufferCount)
            {
                Stop_Zone0();
            }

        mZone0->Unlock();

        mAdapters[ mAdapterNo ] = NULL;

        #ifdef _KMS_LINUX_
            // OpenNetK_OSDep::MapSharedMemory ==> OpenNetK_OSDep::UnmapSharedMemory  See IoCtl_Connect
            mOSDep->UnmapSharedMemory( mOSDep->mContext );
        #endif

        mAdapters  = NULL              ;
        mAdapterNo = ADAPTER_NO_UNKNOWN;
        mConnect_FileObject = NULL     ;
        mSystemId  =                  0;
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
    // aIn  [---;R--]
    // aOut [---;-W-]
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
        case IOCTL_CONFIG_GET      : lResult = IoCtl_Config_Get      (reinterpret_cast<      Adapter_Config *>(aOut)); break;
        case IOCTL_CONFIG_SET      : lResult = IoCtl_Config_Set      (reinterpret_cast<const Adapter_Config *>(aIn ), reinterpret_cast<Adapter_Config *>(aOut)); break;
        case IOCTL_CONNECT         : lResult = IoCtl_Connect         (                                         aIn  , aFileObject  ); break;
        case IOCTL_INFO_GET        : lResult = IoCtl_Info_Get        (reinterpret_cast<      Adapter_Info   *>(aOut)); break;
        case IOCTL_PACKET_DROP     : lResult = IoCtl_Packet_Drop     (); break;
        case IOCTL_PACKET_SEND     : lResult = IoCtl_Packet_Send     (                                         aIn  , aInSize_byte ); break;
        case IOCTL_PACKET_SEND_EX  : lResult = IoCtl_Packet_Send_Ex  (                                         aIn  , aInSize_byte ); break;
        case IOCTL_PACKET_GENERATOR_CONFIG_GET: lResult = IoCtl_PacketGenerator_Config_Get(reinterpret_cast<      PacketGenerator_Config *>(aOut)); break;
        case IOCTL_PACKET_GENERATOR_CONFIG_SET: lResult = IoCtl_PacketGenerator_Config_Set(reinterpret_cast<const PacketGenerator_Config *>(aIn ), reinterpret_cast<PacketGenerator_Config *>(aOut)); break;
        case IOCTL_PACKET_GENERATOR_START     : lResult = IoCtl_PacketGenerator_Start     ( aFileObject ); break;
        case IOCTL_PACKET_GENERATOR_STOP      : lResult = IoCtl_PacketGenerator_Stop      (); break;
        case IOCTL_START           : lResult = IoCtl_Start           (reinterpret_cast<const Buffer         *>(aIn ), aInSize_byte ); break;
        case IOCTL_STATE_GET       : lResult = IoCtl_State_Get       (reinterpret_cast<      Adapter_State  *>(aOut)); break;
        case IOCTL_STATISTICS_GET  : lResult = IoCtl_Statistics_Get  (                                         aIn  , reinterpret_cast<uint32_t *>(aOut), aOutSize_byte); break;
        case IOCTL_STATISTICS_RESET: lResult = IoCtl_Statistics_Reset(); break;
        case IOCTL_STOP            : lResult = IoCtl_Stop            (); break;

        default: ASSERT(false);
        }

        return lResult;
    }

    void Adapter::Tick()
    {
        mPacketGenerator_Counter = 0;
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    // aHeader [---;-W-]
    // aBuffer [---;R--]
    // aPackets [---;-W-]
    //
    // Levels   SoftInt or Thread
    // Threads  Queue
    void Adapter::Buffer_InitHeader_Zone0(OpenNet_BufferHeader * aHeader, const Buffer & aBuffer, Packet * aPackets)
    {
        // TRACE_DEBUG "%s( , ,  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL !=   aHeader           );
        ASSERT(NULL != (&aBuffer)          );
        ASSERT(   0 <    aBuffer.mPacketQty);
        ASSERT(NULL !=   aPackets          );

        ASSERT(NULL != mHardware);

        uint8_t                     * lBase            = reinterpret_cast<uint8_t                     *>(aHeader    );
        volatile OpenNet_PacketInfo * lPacketInfo      = reinterpret_cast<volatile OpenNet_PacketInfo *>(aHeader + 1);
        unsigned int                  lPacketQty       = aBuffer.mPacketQty;
        unsigned int                  lPacketSize_byte = mHardware->GetPacketSize();

        ASSERT(PACKET_SIZE_MAX_byte >= lPacketSize_byte);
        ASSERT(PACKET_SIZE_MIN_byte <= lPacketSize_byte);

        unsigned int lPacketOffset_byte = sizeof(OpenNet_BufferHeader) + (sizeof(OpenNet_PacketInfo) * lPacketQty);

        memset(aHeader, 0, lPacketOffset_byte);  // Writing DirectGMA buffer !

        aHeader->mBufferState           = OPEN_NET_BUFFER_STATE_TX_RUNNING; // Writing DirectGMA buffer !
        aHeader->mPacketInfoOffset_byte = sizeof(OpenNet_BufferHeader); // Writing DirectGMA buffer !
        aHeader->mPacketQty             = lPacketQty; // Writing DirectGMA buffer !
        aHeader->mPacketSize_byte       = lPacketSize_byte; // Writing DirectGMA buffer !

        for (unsigned int i = 0; i < lPacketQty; i++)
        {
            uint32_t lOffset_byte;

            SkipDangerousBoundary(aBuffer.mBuffer_PA, &lPacketOffset_byte, lPacketSize_byte, &lOffset_byte);

            aPackets[i].Init(lOffset_byte, lBase + lOffset_byte);

            lPacketInfo[i].mOffset_byte = lOffset_byte             ; // Writing DirectGMA buffer !
            lPacketInfo[i].mSendTo      = OPEN_NET_PACKET_PROCESSED; // Writing DirectGMA buffer !
        }

        mStatistics[ADAPTER_STATS_BUFFER_INIT_HEADER] ++;
    }

    // aBuffer [---;R--]
    //
    // Level   SoftInt or Thread
    // Thread  Queue
    //
    // Buffer_Queue_Zone0 ==> Buffer_Release_Zone0
    void Adapter::Buffer_Queue_Zone0(const Buffer & aBuffer)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT( NULL != ( & aBuffer )      );
        ASSERT(    0 <  aBuffer.mPacketQty );

        ASSERT( OPEN_NET_BUFFER_QTY >  mBufferCount           );
        ASSERT( NULL                != mOSDep                 );
        ASSERT( NULL                != mOSDep->AllocateMemory );
        ASSERT( NULL                != mOSDep->MapBuffer      );

        BufferInfo * lBuffer = mBuffers + mBufferCount;

        memset( lBuffer, 0, sizeof( BufferInfo ) );

        lBuffer->mBuffer                = aBuffer;
        lBuffer->mPacketInfoOffset_byte = sizeof( OpenNet_BufferHeader );

        // AllocateMemory ==> FreeMemory  See Buffer_Release_Zone0
        lBuffer->mPackets = reinterpret_cast< Packet * >( mOSDep->AllocateMemory( sizeof(Packet) * lBuffer->mBuffer.mPacketQty ) );
        ASSERT( NULL != lBuffer->mPackets );

        // MapBuffer ==> UnmapBuffer  See Buffer_Release_Zone0
        lBuffer->mBase = reinterpret_cast< uint8_t * >( mOSDep->MapBuffer( mOSDep->mContext, & lBuffer->mBuffer.mBuffer_PA, lBuffer->mBuffer.mBuffer_DA, lBuffer->mBuffer.mSize_byte ) );
        ASSERT( NULL != lBuffer->mBase );

        lBuffer->mHeader = reinterpret_cast< OpenNet_BufferHeader * >( lBuffer->mBase );

        #ifdef _KMS_WINDOWS_

            PHYSICAL_ADDRESS lPA;

            lPA.QuadPart = aBuffer.mBuffer_PA;

            mBuffers[mBufferCount].mBase = reinterpret_cast<uint8_t *>(MmMapIoSpace(lPA, aBuffer.mSize_byte, MmNonCached));
            ASSERT(NULL != mBuffers[mBufferCount].mBase);

            mBuffers[mBufferCount].mHeader = reinterpret_cast<OpenNet_BufferHeader *>(mBuffers[mBufferCount].mBase);

            lPA.QuadPart = aBuffer.mMarker_PA;

            mBuffers[mBufferCount].mMarker = reinterpret_cast<uint32_t *>(MmMapIoSpace(lPA, PAGE_SIZE, MmNonCached));
            ASSERT(NULL != mBuffers[mBufferCount].mMarker);

        #endif

        Buffer_InitHeader_Zone0( lBuffer->mHeader, lBuffer->mBuffer, lBuffer->mPackets );
        
        mBufferCount++;

        mStatistics[ADAPTER_STATS_BUFFER_QUEUE] ++;
    }

    // Buffer_Queue_Zone0 ==> Buffer_Release_Zone0
    void Adapter::Buffer_Release_Zone0()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(                   0 <  mBufferCount        );
        ASSERT( OPEN_NET_BUFFER_QTY >= mBufferCount        );
        ASSERT( NULL                != mOSDep              );
        ASSERT( NULL                != mOSDep->FreeMemory  );
        ASSERT( NULL                != mOSDep->UnmapBuffer );

        mBufferCount--;

        ASSERT( NULL != mBuffers[ mBufferCount ].mBase    );
        ASSERT( NULL != mBuffers[ mBufferCount ].mPackets );

        // AllocateMemory ==> FreeMemory  See Buffer_Queue_Zone0
        mOSDep->FreeMemory( mBuffers[ mBufferCount ].mPackets );

        // MapBuffer ==> UnmapBuffer  See Buffer_Queue_zone0
        mOSDep->UnmapBuffer( mOSDep->mContext, mBuffers[ mBufferCount ].mBase );
    }

    // aBuffer [-K-;R--]
    //
    // CRITICAL PATH - Buffer
    //
    // Level    SoftInt
    // Threads  Queue or SoftInt
    void Adapter::Buffer_Receive_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo                                 );
        ASSERT(   0 <  aBufferInfo->mPacketInfoOffset_byte         );
        ASSERT(NULL != aBufferInfo->mPackets                       );
        ASSERT(NULL != aBufferInfo->mBuffer.mPacketQty             );
        ASSERT(NULL != aBufferInfo->mHeader                        );

        ASSERT(NULL != mHardware);
        ASSERT(NULL != mZone0   );

        uint8_t * lBase = reinterpret_cast<uint8_t *>(aBufferInfo->mHeader);

        OpenNet_PacketInfo * lPacketInfo = reinterpret_cast<OpenNet_PacketInfo *>(lBase + aBufferInfo->mPacketInfoOffset_byte);

        mZone0->Unlock();

            mHardware->Lock();

                for (unsigned int i = 0; i < aBufferInfo->mBuffer.mPacketQty; i++)
                {
                    mHardware->Packet_Receive_NoLock(aBufferInfo->mBuffer.mBuffer_PA + aBufferInfo->mPackets[i].GetOffset(), aBufferInfo->mPackets + i, lPacketInfo + i, &aBufferInfo->mRx_Counter);
                }

            mHardware->Unlock_AfterReceive(&aBufferInfo->mRx_Counter, aBufferInfo->mBuffer.mPacketQty);

        mZone0->Lock();

        mStatistics[ADAPTER_STATS_BUFFER_RECEIVE] ++;
    }

    // aBuffer [-K-;R--]
    //
    // CRITICAL PATH - Buffer
    //
    // Level   SoftInt
    // Thread  SoftInt
    void Adapter::Buffer_Send_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo);

        ASSERT(NULL != mAdapters);
        ASSERT(NULL != mZone0   );

        mZone0->Unlock();

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

    // CRITICAL PATH - Buffer
    void Adapter::Buffer_WriteMarker_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo );

        aBufferInfo->mMarkerValue++;

        if ( NULL != aBufferInfo->mMarker )
        {
            (*aBufferInfo->mMarker) = aBufferInfo->mMarkerValue;
        }
    }

    // Level   Thread or SoftInt
    // Thread  Queue or User
    void Adapter::Stop_Zone0()
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(0 < mBufferCount);

        for (unsigned int i = 0; i < mBufferCount; i++)
        {
            mBuffers[i].mFlags.mStopRequested = true;
        }
    }

    // ===== Buffer_State ===================================================
    // aBufferInfo [---;RW-]
    //
    // CRITICAL PATH - Buffer
    //
    // Level   SoftInt
    // Thread  SoftInt

    void Adapter::Buffer_PxCompleted_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo         );
        ASSERT(NULL != aBufferInfo->mHeader);

        ASSERT(OPEN_NET_BUFFER_STATE_PX_COMPLETED == aBufferInfo->mHeader->mBufferState); // Reading DirectGMA buffer !!!

        if (NULL == mAdapters)
        {
            // TRACE_DEBUG "%s - %u %px PX_COMPLETED ==> STOPPED" DEBUG_EOL, __FUNCTION__, mAdapterNo, aBufferInfo TRACE_END;
            aBufferInfo->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_STOPPED;

            Buffer_WriteMarker_Zone0(aBufferInfo);
        }
        else
        {
            // Here, we use a temporary state because Buffer_Send_Zone0
            // release the gate to avoid deadlock with the other adapter's
            // gates.

            // TRACE_DEBUG "%u %px PX_COMPLETED ==> TX_PROGRAMMING" DEBUG_EOL, mAdapterNo, aBufferInfo TRACE_END;
            aBufferInfo->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_TX_PROGRAMMING; // Writing DirectGMA buffer !

            Buffer_Send_Zone0(aBufferInfo);

            ASSERT(OPEN_NET_BUFFER_STATE_TX_PROGRAMMING == aBufferInfo->mHeader->mBufferState); // Reading DirectGMA buffer !!!

            // TRACE_DEBUG "%s - %u %px TX_PROGRAMMING ==> TX_RUNNING" DEBUG_EOL, __FUNCTION__, mAdapterNo, aBufferInfo TRACE_END;
            aBufferInfo->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_TX_RUNNING; // Writing DirectGMA buffer !

            Buffer_TxRunning_Zone0(aBufferInfo);
        }
    }

    // We do not put assert on the buffer state because the GPU may change it
    // at any time.
    void Adapter::Buffer_PxRunning_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aBufferInfo         );
        ASSERT(NULL != aBufferInfo->mHeader);

        if (aBufferInfo->mFlags.mStopRequested)
        {
            // TRACE_DEBUG "%s - %u %px PX_RUNNING ==> STOPPED" DEBUG_EOL, __FUNCTION__, mAdapterNo, aBufferInfo TRACE_END;
            aBufferInfo->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_STOPPED; // Writing DirectGMA buffer !
        }
    }

    // TODO  OpenNetK.Adapter
    //       Low (Feature) - Ajouter la possibilite de remplacer le
    //       traitement OpenCL par un "Forward" fixe. Cela implique
    //       l'allocation de buffer dans la memoire de l'ordinateur par le
    //       pilote lui meme.
    void Adapter::Buffer_RxRunning_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL                             != aBufferInfo                       );
        ASSERT(NULL                             != aBufferInfo->mHeader              );
        ASSERT(OPEN_NET_BUFFER_STATE_RX_RUNNING == aBufferInfo->mHeader->mBufferState);

        if (0 == aBufferInfo->mRx_Counter)
        {
            // TRACE_DEBUG "%s - %u %p RX_RUNNING ==> PX_RUNNING" DEBUG_EOL, __FUNCTION__, mAdapterNo, aBufferInfo TRACE_END;
            aBufferInfo->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_PX_RUNNING;  // Writing DirectGMA buffer !
            Buffer_WriteMarker_Zone0(aBufferInfo);
        }
    }

    void Adapter::Buffer_Stopped_Zone0(unsigned int aIndex)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(OPEN_NET_BUFFER_QTY > aIndex);

        ASSERT(OPEN_NET_BUFFER_QTY >= mBufferCount);

        if (aIndex == (mBufferCount - 1))
        {
            // TRACE_DEBUG "%s - %u %u STOPPED ==> Released" DEBUG_EOL, __FUNCTION__, mAdapterNo, aIndex TRACE_END;

            Buffer_Release_Zone0();
        }
    }

    void Adapter::Buffer_TxRunning_Zone0(BufferInfo * aBufferInfo)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL                             != aBufferInfo                       );
        ASSERT(NULL                             != aBufferInfo->mHeader              );
        ASSERT(OPEN_NET_BUFFER_STATE_TX_RUNNING == aBufferInfo->mHeader->mBufferState); // Reading DirectGMA buffer !!!!

        if (0 == aBufferInfo->mTx_Counter)
        {
            if (aBufferInfo->mFlags.mStopRequested)
            {
                // TRACE_DEBUG "%s - %u %px TX_RUNNING ==> STOPPED" DEBUG_EOL, __FUNCTION__, mAdapterNo, aBufferInfo TRACE_END;
                aBufferInfo->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_STOPPED; // Writing DirectGMA buffer !

                Buffer_WriteMarker_Zone0(aBufferInfo);
            }
            else
            {
                // Here, we use a temporary state because Buffer_Receivd_Zone
                // release the gate to avoid deadlock with the Hardware's
                // gates.
                // TRACE_DEBUG "%s - %u %px TX_RUNNING ==> RX_PROGRAMMING" DEBUG_EOL, __FUNCTION__, mAdapterNo, aBufferInfo TRACE_END;
                aBufferInfo->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_RX_PROGRAMMING; // Writing DirectGMA buffer !

                Buffer_Receive_Zone0(aBufferInfo);

                ASSERT(OPEN_NET_BUFFER_STATE_RX_PROGRAMMING == aBufferInfo->mHeader->mBufferState); // Reading DirectGMA buffer !!!

                // TRACE_DEBUG "%s - %u %px RX_PROGRAMMING ==> RX_RUNNING" DEBUG_EOL, __FUNCTION__, mAdapterNo, aBufferInfo TRACE_END;
                aBufferInfo->mHeader->mBufferState = OPEN_NET_BUFFER_STATE_RX_RUNNING; // Writing DirectGMA buffer !
            }
        }
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

    int Adapter::IoCtl_Connect( const void * aIn, void * aFileObject )
    {
        // TRACE_DEBUG "%s( ,  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aIn);
        ASSERT(NULL != aFileObject );

        mStatistics[ADAPTER_STATS_IOCTL_CONNECT] ++;

        const IoCtl_Connect_In * lIn = reinterpret_cast<const IoCtl_Connect_In *>(aIn);

        if ( NULL == lIn->mSharedMemory )
        {
            return IOCTL_RESULT_INVALID_PARAMETER;
        }

        if (0 == lIn->mSystemId)
        {
            return IOCTL_RESULT_INVALID_SYSTEM_ID;
        }

        if ( NULL != mAdapters )
        {
            return IOCTL_RESULT_ALREADY_CONNECTED;
        }

        ASSERT(ADAPTER_NO_UNKNOWN == mAdapterNo);
        ASSERT(                 0 == mSystemId );

        mSystemId = lIn->mSystemId;

        #ifdef _KMS_LINUX_

            // OpenNetK::MapShareadMemory ==> OpenNetK::UnmapSharedMemory  See Disconnect
            mAdapters = reinterpret_cast< OpenNetK::Adapter * * >( mOSDep->MapSharedMemory( mOSDep->mContext, lIn->mSharedMemory, SHARED_MEMORY_SIZE_byte ) );
            if ( NULL == mAdapters )
            {
                printk( KERN_ERR "%s - OpenNetK_OSdep::MapSharedMemory( , , ,  ) failed\n", __FUNCTION__ );
                return IOCTL_RESULT_SYSTEM_ERROR;
            }

        #endif

        #ifdef _KMS_WINDOWS_

            mAdapters = reinterpret_cast<Adapter **>(lIn->mSharedMemory);

        #endif

        for (unsigned int i = 0; i < ADAPTER_NO_QTY; i++)
        {
            if (NULL == mAdapters[i])
            {
                mAdapters[i] = this;
                mAdapterNo = i;

                mConnect_FileObject = aFileObject;

                return IOCTL_RESULT_OK;
            }
        }

        Disconnect();

        return IOCTL_RESULT_TOO_MANY_ADAPTER;
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

    int Adapter::IoCtl_Packet_Drop()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT( NULL != mHardware );

        if ( ! mHardware->Packet_Drop() )
        {
            printk( KERN_ERR "%s - Hardware::Packet_Drop() failed\n", __FUNCTION__ );

            // TODO  OpenNetK.Adapter
            //       Create a specific result code
            return IOCTL_RESULT_NO_BUFFER;
        }

        return IOCTL_RESULT_OK;
    }

    // TODO  ONK_Lib.Adapter.ErrorHandling
    //       High (Test) - Verify if aIn = NULL and aInSize_byte = 0 cause
    //       problem.
    int Adapter::IoCtl_Packet_Send(const void * aIn, unsigned int aInSize_byte)
    {
        // TRACE_DEBUG "%s( , %u bytes )" DEBUG_EOL, __FUNCTION__, aInSize_byte TRACE_END;

        ASSERT(NULL != mHardware);

        if (!mHardware->Packet_Send(aIn, aInSize_byte))
        {
            // TODO  OpenNetK.Adapter
            //       Create a specific result code
            return IOCTL_RESULT_NO_BUFFER;
        }

        mStatistics[ADAPTER_STATS_IOCTL_PACKET_SEND] ++;

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
            return IOCTL_RESULT_INVALID_PARAMETER;
        }

        unsigned int lSize_byte = aInSize_byte - sizeof(IoCtl_Packet_Send_Ex_In);
        if ( lIn->mSize_byte < lSize_byte )
        {
            lSize_byte = lIn->mSize_byte;
        }

        if (!mHardware->Packet_Send(lIn + 1, lSize_byte, lIn->mRepeatCount))
        {
            // TODO  OpenNetK.Adapter
            //       Create a specific result code
            return IOCTL_RESULT_NO_BUFFER;
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

    // TODO  OpenNetK.Adapter
    //       Stop the packet generator when the application who started it
    //       close the connection.
    int Adapter::IoCtl_PacketGenerator_Stop()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        if ( NULL == mPacketGenerator_FileObject )
        {
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

        mStatistics[ADAPTER_STATS_IOCTL_START] ++;

        unsigned int lCount = aInSize_byte / sizeof(Buffer);

        OpenNetK_IoCtl_Result lResult;

        mZone0->Lock();

            // TODO  ONK_Lib.Adapter
            //       High - Refuse to start if the number of buffer may cause
            //       a descriptor overrun.

            ASSERT(OPEN_NET_BUFFER_QTY >= mBufferCount);

            if (OPEN_NET_BUFFER_QTY >= (mBufferCount + lCount))
            {
                for (unsigned int i = 0; i < lCount; i++)
                {
                    if ( ( NULL == aIn[ i ].mBuffer_DA ) && ( 0 == aIn[ i ].mBuffer_PA ) )
                    {
                        break;
                    }

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

    int Adapter::IoCtl_State_Get(Adapter_State * aOut)
    {
        // TRACE_DEBUG "%s(  )" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != aOut);

        ASSERT(NULL != mHardware);

        memset(aOut, 0, sizeof(Adapter_State));

        aOut->mAdapterNo   = mAdapterNo  ;
        aOut->mBufferCount = mBufferCount;
        aOut->mSystemId    = mSystemId   ;

        mHardware->GetState(aOut);

        mStatistics[ADAPTER_STATS_IOCTL_STATE_GET] ++;

        return sizeof(Adapter_State);
    }

    int Adapter::IoCtl_Statistics_Get(const void * aIn, uint32_t * aOut, unsigned int aOutSize_byte) const
    {
        // TRACE_DEBUG "%s( , , %u bytes )" DEBUG_EOL, __FUNCTION__, aOutSize_byte TRACE_END;

        ASSERT(NULL != aIn);

        ASSERT(NULL != mHardware);

        const IoCtl_Statistics_Get_In * lIn = reinterpret_cast<const IoCtl_Statistics_Get_In *>(aIn);

        uint32_t                 * lOut          = aOut;
        unsigned int               lOutSize_byte = aOutSize_byte;
        int                        lResult_byte  = 0;

        bool lReset = lIn->mFlags.mReset;

        if ( lIn->mOutputSize_byte < lOutSize_byte )
        {
            lOutSize_byte = lIn->mOutputSize_byte;
        }

        #ifdef _KMS_WINDOWS_

            LARGE_INTEGER lNow;

            KeQuerySystemTimePrecise(&lNow);

            mStatistics[ADAPTER_STATS_RUNNING_TIME_ms] = static_cast<unsigned int>((lNow.QuadPart - mStatistics_Start.QuadPart) / 10000);

        #endif

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
            #ifdef _KMS_WINDOWS_
                mStatistics_Start = lNow;
            #endif

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

        #ifdef _KMS_WINDOWS_
            KeQuerySystemTimePrecise(&mStatistics_Start);
        #endif

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

        mStatistics[ADAPTER_STATS_IOCTL_STOP] ++;

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
