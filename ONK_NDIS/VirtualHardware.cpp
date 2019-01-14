
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/VirtualHardware.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Constants.h>
#include <OpenNetK/Hardware_Statistics.h>
#include <OpenNetK/Interface.h>
#include <OpenNetK/SpinLock.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/Version.h"

// ===== ONL_NDIS ===========================================================
#include "VirtualHardware.h"

// Public
/////////////////////////////////////////////////////////////////////////////

VirtualHardware::VirtualHardware() : mTx_Callback(NULL), mTx_Context(NULL)
{
    DbgPrintEx(DEBUG_ID, DEBUG_CONSTRUCTOR, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    mConfig.mPacketSize_byte = PACKET_SIZE_MAX_byte;

    mInfo.mPacketSize_byte = PACKET_SIZE_MAX_byte;

    mInfo.mRx_Descriptors = RX_DESCRIPTOR_QTY;

    strcpy(mInfo.mComment                , "ONK_NDIS");
    strcpy(mInfo.mVersion_Driver.mComment, "ONK_NDIS");
}

// aData [---;R--] The data
// aSize_byte      The size
//
// CRITICAL PATH - Packet
void VirtualHardware::Rx_IndicatePacket(const void * aData, unsigned int aSize_byte)
{
    DbgPrintEx(DEBUG_ID, DEBUG_CONSTRUCTOR, PREFIX __FUNCTION__ "( , %u bytes )" DEBUG_EOL, aSize_byte);

    ASSERT(NULL != aData     );
    ASSERT(0    <  aSize_byte);

    ASSERT(RX_DESCRIPTOR_QTY > mRx_In );
    ASSERT(RX_DESCRIPTOR_QTY > mRx_Out);

    if (mRx_In == mRx_Out)
    {
        mStatistics[OpenNetK::HARDWARE_STATS_RX_QUEUE_DROPPED_packet] ++;
    }
    else
    {
        ASSERT(NULL != mRx_Counter   [mRx_Out]);
        ASSERT(NULL != mRx_PacketData[mRx_Out]);
        ASSERT(NULL != mRx_PacketInfo[mRx_Out]);

        memcpy(mRx_PacketData[mRx_Out]->GetVirtualAddress(), aData, aSize_byte);

        mRx_PacketData[mRx_Out]->IndicateRxCompleted();
        mRx_PacketInfo[mRx_Out]->mSize_byte = aSize_byte; // Writing DirectGMA buffer !
        mRx_PacketInfo[mRx_Out]->mSendTo    =          0; // Writing DirectGMA buffer !

        InterlockedDecrement(mRx_Counter[mRx_Out]);

        mRx_Out = (mRx_Out + 1) % RX_DESCRIPTOR_QTY;

        mStatistics[OpenNetK::HARDWARE_STATS_RX_packet     ] ++;
        mStatistics[OpenNetK::HARDWARE_STATS_RX_HOST_byte  ] += aSize_byte;
        mStatistics[OpenNetK::HARDWARE_STATS_RX_HOST_packet] ++;
    }
}

// aCallback [-K-;--X]
// aContext  [-K-;---]
void VirtualHardware::Tx_RegisterCallback(Tx_Callback * aCallback, void * aContext)
{
    DbgPrintEx(DEBUG_ID, DEBUG_CONSTRUCTOR, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aCallback);

    ASSERT(NULL == mTx_Callback);
    ASSERT(NULL == mTx_Context );

    mTx_Callback = aCallback;
    mTx_Context  = aContext ;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNetK::Adapter ==================================================

void VirtualHardware::GetState(OpenNetK::Adapter_State * aState)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aState);

    Hardware::GetState(aState);

    aState->mFlags.mFullDuplex = true ;
    aState->mFlags.mLinkUp     = true ;
    aState->mFlags.mTx_Off     = false;
    aState->mSpeed_MB_s        =  1000;
}

bool VirtualHardware::D0_Entry()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != mZone0);

    mZone0->Lock();

        mRx_In  = 0;
        mRx_Out = 0;

    mZone0->Unlock();

    return Hardware::D0_Entry();
}

// CRITICAL PATH - Packet
void VirtualHardware::Packet_Receive_NoLock(uint64_t aData, OpenNetK::Packet * aPacketData, OpenNet_PacketInfo * aPacketInfo, volatile long * aCounter)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( , , ,  )" DEBUG_EOL);

    ASSERT(NULL != aPacketData);
    ASSERT(NULL != aPacketInfo);
    ASSERT(NULL != aCounter   );

    mRx_Counter   [mRx_In] = aCounter   ;
    mRx_Data      [mRx_In] = aData      ;
    mRx_PacketData[mRx_In] = aPacketData;
    mRx_PacketInfo[mRx_In] = aPacketInfo;

    mRx_PacketData[mRx_In]->IndicateRxRunning();

    mRx_In = (mRx_In + 1) % RX_DESCRIPTOR_QTY;
}

// CRITICAL PATH - Packet
void VirtualHardware::Packet_Send_NoLock(uint64_t, const void * aVirtualAddress, unsigned int aSize_byte, volatile long * aCounter)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( , , %u,  )" DEBUG_EOL, aSize_byte);

    ASSERT(NULL != aVirtualAddress);
    ASSERT(0    <  aSize_byte     );
    ASSERT(NULL != aCounter       );

    if (NULL == mTx_Callback)
    {
        mStatistics[OpenNetK::HARDWARE_STATS_TX_DISCARDED_packet]++;
    }
    else
    {
        if (mTx_Callback(mTx_Context, aVirtualAddress, aSize_byte))
        {
            mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_byte  ] += aSize_byte;
            mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_packet] ++;
            mStatistics[OpenNetK::HARDWARE_STATS_TX_packet     ] ++;
        }
        else
        {
            mStatistics[OpenNetK::HARDWARE_STATS_TX_DISCARDED_packet] ++;
        }
    }

    InterlockedDecrement(aCounter);
}

// CRITICAL PATH
void VirtualHardware::Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aRepeatCount)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( , %u bytes, %u )" DEBUG_EOL, aSize_byte, aRepeatCount);

    ASSERT(NULL != aPacket     );
    ASSERT(   0 <  aSize_byte  );
    ASSERT(   0 <  aRepeatCount);

    ASSERT(NULL != mZone0);

    if (NULL == mTx_Callback)
    {
        mStatistics[OpenNetK::HARDWARE_STATS_TX_DISCARDED_packet] += aRepeatCount;
    }
    else
    {
        Lock();

            for (unsigned int i = 0; i < aRepeatCount; i++)
            {
                if (mTx_Callback(&mTx_Context, aPacket, aSize_byte))
                {
                    mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_byte  ] += aSize_byte;
                    mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_packet] ++;
                    mStatistics[OpenNetK::HARDWARE_STATS_TX_packet     ] ++;
                }
                else
                {
                    mStatistics[OpenNetK::HARDWARE_STATS_TX_DISCARDED_packet] ++;
                }
            }

        Unlock_AfterSend(NULL, aRepeatCount);
    }
}