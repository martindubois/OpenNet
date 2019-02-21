
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Pro1000.cpp

// REQUIREMENT  ONK_X.InterruptRateLimitation
//              The adapter driver limit the interruption rate.

#define __CLASS__     "Pro1000::"
#define __NAMESPACE__ ""

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Constants.h>
#include <OpenNetK/Hardware_Statistics.h>
#include <OpenNetK/SpinLock.h>

// ===== Common =============================================================
#include "../Common/Version.h"

// ===== ONL_Pro1000 ========================================================
#include "Pro1000.h"

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define PACKET_SIZE_byte  (9 * 1024)

// Public
/////////////////////////////////////////////////////////////////////////////

Pro1000::Pro1000()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT( 0x0e02c == sizeof( Pro1000_BAR1          ) );
    ASSERT(       4 == sizeof( Pro1000_DeviceControl ) );
    ASSERT(      16 == sizeof( Pro1000_Rx_Descriptor ) );
    ASSERT(      16 == sizeof( Pro1000_Tx_Descriptor ) );

    mConfig.mPacketSize_byte = PACKET_SIZE_byte;

    mInfo.mPacketSize_byte = PACKET_SIZE_byte;

    mInfo.mCommonBufferSize_byte += (sizeof(Pro1000_Rx_Descriptor) * RX_DESCRIPTOR_QTY); // Rx packet descriptors
    mInfo.mCommonBufferSize_byte += (sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY); // Tx packet descriptors
    mInfo.mCommonBufferSize_byte += (PACKET_SIZE_byte * PACKET_BUFFER_QTY); // Packet buffers
    mInfo.mCommonBufferSize_byte += (mInfo.mCommonBufferSize_byte / OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte) * PACKET_SIZE_byte; // Skip 64 KB boundaries

    mInfo.mRx_Descriptors = RX_DESCRIPTOR_QTY;
    mInfo.mTx_Descriptors = TX_DESCRIPTOR_QTY;

    strcpy(mInfo.mComment                  , "ONK_Pro1000");
    strcpy(mInfo.mVersion_Driver  .mComment, "ONK_Pro1000");
    strcpy(mInfo.mVersion_Hardware.mComment, "Intel Gigabit ET Dual Port Server Adapter");

    memset((void *)(&mPacketBuffer_Counter), 0, sizeof(mPacketBuffer_Counter)); // volatile_cast
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNetK::Adapter ==================================================

void Pro1000::GetState(OpenNetK::Adapter_State * aState)
{
    // TRACE_DEBUG "%s( 0x%px )" DEBUG_EOL, __FUNCTION__, aState TRACE_END;

    ASSERT(NULL != aState);

    ASSERT(NULL != mZone0);

    mZone0->Lock();

        ASSERT(NULL != mBAR1_MA);

        Pro1000_DeviceStatus lDeviceStatus;

        lDeviceStatus.mValue = mBAR1_MA->mDeviceStatus.mValue;

        aState->mFlags.mFullDuplex = lDeviceStatus.mFields.mFullDuplex;
        aState->mFlags.mLinkUp     = lDeviceStatus.mFields.mLinkUp    ;
        aState->mFlags.mTx_Off     = lDeviceStatus.mFields.mTx_Off    ;

        // TODO  ONK_Pro1000.Pro1000
        //       High (Feature) - Comprendre pourquoi la vitesse n'est pas
        //       indique correctement.

        switch (lDeviceStatus.mFields.mSpeed)
        {
        case 0x0: aState->mSpeed_MB_s =  10; break;
        case 0x1: aState->mSpeed_MB_s = 100; break;

        case 0x2:
        case 0x3:
            aState->mSpeed_MB_s = 1000;
            break;

        default:
            ASSERT(false);
        }

    mZone0->Unlock();
}

void Pro1000::ResetMemory()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    Hardware::ResetMemory();

    mZone0->Lock();

        mBAR1_MA = NULL;

    mZone0->Unlock();
}

void Pro1000::SetCommonBuffer(uint64_t aCommon_PA, void * aCommon_CA)
{
    // TRACE_DEBUG "%s( 0x%llx, 0x%px )" DEBUG_EOL, __FUNCTION__, aCommon_PA, aCommon_CA TRACE_END;

    ASSERT(NULL != aCommon_CA);

    ASSERT(NULL != mZone0);

    mZone0->Lock();

        uint64_t  lCommon_PA = aCommon_PA;
        uint8_t * lCommon_CA = reinterpret_cast<uint8_t *>(aCommon_CA);

        mRx_CA = reinterpret_cast<Pro1000_Rx_Descriptor *>(lCommon_CA);
        mRx_PA = lCommon_PA;

        lCommon_CA += sizeof(Pro1000_Rx_Descriptor) * RX_DESCRIPTOR_QTY;
        lCommon_PA += sizeof(Pro1000_Rx_Descriptor) * RX_DESCRIPTOR_QTY;

        mTx_CA = reinterpret_cast<Pro1000_Tx_Descriptor *>(lCommon_CA);
        mTx_PA = lCommon_PA;

        lCommon_CA += sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY;
        lCommon_PA += sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY;

        unsigned int i;

        for (i = 0; i < PACKET_BUFFER_QTY; i++)
        {
            SkipDangerousBoundary(&lCommon_PA, &lCommon_CA, mConfig.mPacketSize_byte, mPacketBuffer_PA + i, reinterpret_cast<uint8_t **>(mPacketBuffer_CA + i));

            ASSERT(NULL != mPacketBuffer_CA[i]);
        }

        for (i = 0; i < TX_DESCRIPTOR_QTY; i++)
        {
            mTx_CA[i].mFields.mEndOfPacket  = true;
            mTx_CA[i].mFields.mInsertCRC    = true;
            mTx_CA[i].mFields.mReportStatus = true;
        }

    mZone0->Unlock();
}

// NOT TESTED  ONK_Pro1000.Pro1000.ErrorHandling
//             Memory 0 too small
bool Pro1000::SetMemory(unsigned int aIndex, void * aMemory_MA, unsigned int aSize_byte)
{
    // TRACE_DEBUG "%s( %u, 0x%px, %u bytes )" DEBUG_EOL, __FUNCTION__, aIndex, aMemory_MA, aSize_byte TRACE_END;

    ASSERT(NULL != aMemory_MA);
    ASSERT(   0 <  aSize_byte);

    ASSERT(NULL != mZone0);

    switch (aIndex)
    {
    case 0:
        if (sizeof(Pro1000_BAR1) > aSize_byte)
        {
            return false;
        }

        mZone0->Lock();

            ASSERT(NULL == mBAR1_MA);

            mBAR1_MA = reinterpret_cast< volatile Pro1000_BAR1 * >( aMemory_MA );

            Interrupt_Disable_Zone0();

            mInfo.mEthernetAddress.mAddress[0] = mBAR1_MA->mRx_AddressLow0 .mFields.mA;
            mInfo.mEthernetAddress.mAddress[1] = mBAR1_MA->mRx_AddressLow0 .mFields.mB;
            mInfo.mEthernetAddress.mAddress[2] = mBAR1_MA->mRx_AddressLow0 .mFields.mC;
            mInfo.mEthernetAddress.mAddress[3] = mBAR1_MA->mRx_AddressLow0 .mFields.mD;
            mInfo.mEthernetAddress.mAddress[4] = mBAR1_MA->mRx_AddressHigh0.mFields.mE;
            mInfo.mEthernetAddress.mAddress[5] = mBAR1_MA->mRx_AddressHigh0.mFields.mF;

        mZone0->Unlock();
        break;
    }

    return Hardware::SetMemory(aIndex, aMemory_MA, aSize_byte);
}

void Pro1000::D0_Entry()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mZone0);

    mZone0->Lock();

        ASSERT(NULL != mBAR1_MA);

        mRx_In  = 0;
        mRx_Out = 0;

        mTx_In  = 0;
        mTx_Out = 0;

        memset(&mTx_Counter, 0, sizeof(mTx_Counter));

        mPacketBuffer_In = 0;

        Reset_Zone0();

        Pro1000_DeviceControl lCTRL;

        lCTRL.mValue = mBAR1_MA->mDeviceControl.mValue;

        lCTRL.mFields.mInvertLossOfSignal   = false;
        // lCTRL.mFields.mRx_FlowControlEnable = true ;
        lCTRL.mFields.mSetLinkUp            = true ;
        // lCTRL.mFields.mTx_FlowControlEnable = true ;

        mBAR1_MA->mDeviceControl.mValue = lCTRL.mValue;

        mBAR1_MA->mGeneralPurposeInterruptEnable.mFields.mExtendedInterruptAutoMaskEnable = true;

        mBAR1_MA->mInterruptVectorAllocation[0].mFields.mVector0Valid = true;
        mBAR1_MA->mInterruptVectorAllocation[0].mFields.mVector1Valid = true;
        mBAR1_MA->mInterruptVectorAllocation[0].mFields.mVector2Valid = true;
        mBAR1_MA->mInterruptVectorAllocation[0].mFields.mVector3Valid = true;

        mBAR1_MA->mInterruptVectorAllocationMisc.mFields.mVector32Valid = true;
        mBAR1_MA->mInterruptVectorAllocationMisc.mFields.mVector33Valid = true;

        mBAR1_MA->mInterruptTrottle[0].mFields.mInterval_us = 100;

        // mBAR1->mFlowControlReceiveThresholdHigh.mFields.mReceiveThresholdHigh = 3000;

        // mBAR1->mFlowControlReceiveThresholdLow.mFields.mReceiveThresholdLow = 2000;
        // mBAR1->mFlowControlReceiveThresholdLow.mFields.mXOnEnable           = true;

        // mBAR1->mFlowControlRefreshThreshold.mFields.mValue = 0x8000;

        Rx_Config_Zone0();
        Tx_Config_Zone0();

    mZone0->Unlock();

    Hardware::D0_Entry();
}

bool Pro1000::D0_Exit()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    Interrupt_Disable();

    return Hardware::D0_Exit();
}

void Pro1000::Interrupt_Disable()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mZone0);

    Hardware::Interrupt_Disable();

    mZone0->Lock();

        Interrupt_Disable_Zone0();

    mZone0->Unlock();
}

void Pro1000::Interrupt_Enable()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mZone0);

    Hardware::Interrupt_Enable();

    mZone0->Lock();

        ASSERT(NULL != mBAR1_MA);

        mBAR1_MA->mInterruptMaskSet.mFields.mTx_DescriptorWritten = true;
        mBAR1_MA->mInterruptMaskSet.mFields.mRx_DescriptorWritten = true;

    mZone0->Unlock();
}

// CRITICAL PATH
bool Pro1000::Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing)
{
    // TRACE_DEBUG "%s( %u, 0x%px )" DEBUG_EOL, __FUNCTION__, aMessageId, aNeedMoreProcessing TRACE_END;

    ASSERT(NULL != aNeedMoreProcessing);

    ASSERT(NULL != mBAR1_MA);

    (void)(aMessageId);

    uint32_t lValue = mBAR1_MA->mInterruptCauseRead.mValue; // Reading hardware !!!
    (void)(lValue);

    (*aNeedMoreProcessing) = true;

    mStatistics[OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS] ++;

    mStatistics[OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS_LAST_MESSAGE_ID] = aMessageId;

    return true;
}

// CRITICAL PATH
void Pro1000::Interrupt_Process2(bool * aNeedMoreProcessing)
{
    // TRACE_DEBUG "%s( 0x%p )" DEBUG_EOL, __FUNCTION__, aNeedMoreProcessing TRACE_END;

    ASSERT(NULL != aNeedMoreProcessing);

    ASSERT(NULL != mZone0);

    mZone0->Lock();

        Rx_Process_Zone0();
        Tx_Process_Zone0();

    mZone0->Unlock();

    Hardware::Interrupt_Process2(aNeedMoreProcessing);
}

bool Pro1000::Packet_Drop()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT( PACKET_BUFFER_QTY > mPacketBuffer_In );

    bool lResult;

    Lock();

    lResult = ( ( 0 >= mPacketBuffer_Counter[ mPacketBuffer_In ] ) && ( Rx_GetAvailableDescriptor_Zone0() > 0 ) );
    if ( lResult )
    {
        volatile long * lCounter = mPacketBuffer_Counter + mPacketBuffer_In;

        Packet_Receive_NoLock( mPacketBuffer_PA[ mPacketBuffer_In ], & mPacketData, & mPacketInfo, lCounter );

        mPacketBuffer_In = (mPacketBuffer_In + 1) % PACKET_BUFFER_QTY;

        Unlock_AfterReceive( lCounter, 1 );
    }
    else
    {
        Unlock();
    }

    return lResult;
}

// CRITICAL PATH - Packet
void Pro1000::Packet_Receive_NoLock(uint64_t aData_PA, OpenNetK::Packet * aPacketData, OpenNet_PacketInfo * aPacketInfo_MA, volatile long * aCounter)
{
    // TRACE_DEBUG "%s( 0x%llx, 0x%px, 0x%px, 0x%px )" DEBUG_EOL, __FUNCTION__, aData_PA, aPacketData, aPacketInfo, aCounter TRACE_END;

    ASSERT(NULL != aPacketData   );
    ASSERT(NULL != aPacketInfo_MA);
    ASSERT(NULL != aCounter      );

    ASSERT(NULL              != mRx_CA);
    ASSERT(RX_DESCRIPTOR_QTY >  mRx_In);

    mRx_Counter      [mRx_In] = aCounter      ;
    mRx_PacketData   [mRx_In] = aPacketData   ;
    mRx_PacketInfo_MA[mRx_In] = aPacketInfo_MA;

    mRx_PacketData[mRx_In]->IndicateRxRunning();

    memset((Pro1000_Rx_Descriptor *)(mRx_CA) + mRx_In, 0, sizeof(Pro1000_Rx_Descriptor)); // volatile_cast

    mRx_CA[mRx_In].mLogicalAddress = aData_PA;

    mRx_In = (mRx_In + 1) % RX_DESCRIPTOR_QTY;
}

// CRITICAL PATH - Packet
void Pro1000::Packet_Send_NoLock(uint64_t aData_PA, const void *, unsigned int aSize_byte, volatile long * aCounter)
{
    // TRACE_DEBUG "%s( 0x%llx, , %u bytes, 0x%px )" DEBUG_EOL, __FUNCTION__, aData_PA, aSize_byte, aCounter TRACE_END;

    ASSERT(0 != aData_PA  );
    ASSERT(0 <  aSize_byte);

    ASSERT(NULL              != mTx_CA);
    ASSERT(TX_DESCRIPTOR_QTY >  mTx_In);

    mTx_Counter[mTx_In] = aCounter;

    mTx_CA[mTx_In].mFields.mDescriptorDone = false     ;
    mTx_CA[mTx_In].mFields.mSize_byte      = aSize_byte;
    mTx_CA[mTx_In].mLogicalAddress         = aData_PA  ;

    mTx_In = (mTx_In + 1) % TX_DESCRIPTOR_QTY;
}

// CRITICAL PATH
bool Pro1000::Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aRepeatCount)
{
    // TRACE_DEBUG "%s( 0x%px, %u bytes, %u )" DEBUG_EOL, __FUNCTION__, aPacket, aSize_byte, aRepeatCount TRACE_END;

    ASSERT( NULL != aPacket      );
    ASSERT(    0 <  aSize_byte   );
    ASSERT(    0 <  aRepeatCount );

    ASSERT(NULL != mTx_CA);
    ASSERT(NULL != mZone0);

    bool lResult;

    Lock();

    ASSERT(NULL != mBAR1_MA);

    ASSERT(PACKET_BUFFER_QTY >  mPacketBuffer_In                  );
    ASSERT(NULL              != mPacketBuffer_CA[mPacketBuffer_In]);

    lResult = ((0 >= mPacketBuffer_Counter[mPacketBuffer_In]) && (Tx_GetAvailableDescriptor_Zone0() >= aRepeatCount));
    if (lResult)
    {
        memcpy(mPacketBuffer_CA[mPacketBuffer_In], aPacket, aSize_byte);

        volatile long * lCounter   = mPacketBuffer_Counter + mPacketBuffer_In;
        uint64_t        lPacket_PA = mPacketBuffer_PA[mPacketBuffer_In];

        for (unsigned int i = 0; i < aRepeatCount; i++)
        {
            Packet_Send_NoLock(lPacket_PA, NULL, aSize_byte, lCounter);
        }

        mPacketBuffer_In = (mPacketBuffer_In + 1) % PACKET_BUFFER_QTY;

        Unlock_AfterSend(lCounter, aRepeatCount);
    }
    else
    {
        Unlock_AfterSend(NULL, 0);
    }

    return lResult;
}

unsigned int Pro1000::Statistics_Get(uint32_t * aOut, unsigned int aOutSize_byte, bool aReset)
{
    // TRACE_DEBUG "%s( 0x%px, %u bytes, %s )" DEBUG_EOL, __FUNCTION__, aOut, aOutSize_byte, aReset ? "true" : "false" TRACE_END;

    Statistics_Update();

    return Hardware::Statistics_Get(aOut, aOutSize_byte, aReset);
}

void Pro1000::Statistics_Reset()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    Statistics_Update();

    Hardware::Statistics_Reset();
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// CRITICAL PATH - Buffer
void Pro1000::Unlock_AfterReceive_Internal()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL              != mBAR1_MA);
    ASSERT(RX_DESCRIPTOR_QTY >  mRx_In  );

    Pro1000_Rx_DescriptorTail lReg;

    lReg.mValue = 0;
    lReg.mFields.mValue = mRx_In;

    mBAR1_MA->mRx_DescriptorTail0.mValue = lReg.mValue; // Writing hardware !
}

// CRITICAL PATH - Buffer
void Pro1000::Unlock_AfterSend_Internal()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__, TRACE_END;

    ASSERT(NULL              != mBAR1_MA);
    ASSERT(RX_DESCRIPTOR_QTY > mTx_In   );

    Pro1000_Tx_DescriptorTail lReg;

    lReg.mValue = 0;
    lReg.mFields.mValue = mTx_In;

    mBAR1_MA->mTx_DescriptorTail0.mValue = lReg.mValue; // Writing hardware !
}

// Private
/////////////////////////////////////////////////////////////////////////////

void Pro1000::Interrupt_Disable_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_MA);

    mBAR1_MA->mInterruptMaskClear.mValue = 0xffffffff;
}

void Pro1000::Reset_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_MA);

    Interrupt_Disable_Zone0();

    mBAR1_MA->mDeviceControl.mFields.mReset = true;

    while (mBAR1_MA->mDeviceControl.mFields.mReset)
    {
        // TRACE_DEBUG "%s - Waiting ..." DEBUG_EOL, __FUNCTION__ TRACE_END;
    }

    Interrupt_Disable_Zone0();

    uint32_t lValue = mBAR1_MA->mInterruptCauseRead.mValue;
    (void)(lValue);
}

// Level   Thread
// Thread  Init
void Pro1000::Rx_Config_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_MA                );
    ASSERT(   0 <  mConfig.mPacketSize_byte);

    for (unsigned int i = 0; i < (sizeof(mBAR1_MA->mMulticastTableArray) / sizeof(mBAR1_MA->mMulticastTableArray[0])); i++)
    {
        mBAR1_MA->mMulticastTableArray[i] = 0;
    }

    // TODO  OpenNet.Adapter
    //       Low (Feature) - Add configuration field for:
    //       BroadcastAcceptMode, MulticastPromiscuousEnabled,
    //       PassMacControlFrames, StoreBadPackets, UnicastPromiscuousEnabled
    mBAR1_MA->mRx_Control.mFields.mBroadcastAcceptMode         = true ;
    mBAR1_MA->mRx_Control.mFields.mDiscardPauseFrames          = true ;
    mBAR1_MA->mRx_Control.mFields.mLongPacketEnabled           = true ;
    mBAR1_MA->mRx_Control.mFields.mMulticastPromiscuousEnabled = true ;
    mBAR1_MA->mRx_Control.mFields.mPassMacControlFrames        = false;
    mBAR1_MA->mRx_Control.mFields.mStoreBadPackets             = false;
    mBAR1_MA->mRx_Control.mFields.mStripEthernetCRC            = true ;
    mBAR1_MA->mRx_Control.mFields.mUnicastPromiscuousEnabled   = true ;

    mBAR1_MA->mRx_DmaMaxOutstandingData.mFields.mValue_256_bytes = 0xfff;

    mBAR1_MA->mRx_LongPacketMaximumLength.mFields.mValue_byte = mConfig.mPacketSize_byte;

    mBAR1_MA->mRx_DescriptorBaseAddressHigh0 = (mRx_PA >> 32) & 0xffffffff;
    mBAR1_MA->mRx_DescriptorBaseAddressLow0  =  mRx_PA        & 0xffffffff;

    mBAR1_MA->mRx_DescriptorRingLength0.mFields.mValue_byte = sizeof(Pro1000_Rx_Descriptor) * RX_DESCRIPTOR_QTY;

    mBAR1_MA->mRx_PacketBufferSize.mFields.mValue_KB = 84;

    mBAR1_MA->mRx_SplitAndReplicationControl.mFields.mDescriptorType     =    0;
    mBAR1_MA->mRx_SplitAndReplicationControl.mFields.mDropEnabled        = true;
    mBAR1_MA->mRx_SplitAndReplicationControl.mFields.mHeaderSize_64bytes =    0;
    mBAR1_MA->mRx_SplitAndReplicationControl.mFields.mPacketSize_KB      = mConfig.mPacketSize_byte / 1024;

    mBAR1_MA->mRx_DescriptorControl0.mFields.mHostThreshold      =   16;
    mBAR1_MA->mRx_DescriptorControl0.mFields.mPrefetchThreshold  =   16;
    mBAR1_MA->mRx_DescriptorControl0.mFields.mWriteBackThreshold =   16;
    mBAR1_MA->mRx_DescriptorControl0.mFields.mQueueEnable        = true;

    mBAR1_MA->mRx_Control.mFields.mEnable = true;
}

// CRITICAL PATH
//
// Level   SoftInt
// Thread  SoftInt
void Pro1000::Rx_Process_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL              != mRx_CA );
    ASSERT(RX_DESCRIPTOR_QTY >  mRx_In );
    ASSERT(RX_DESCRIPTOR_QTY >  mRx_Out);

    while (mRx_In != mRx_Out)
    {
        if (!mRx_CA[mRx_Out].mFields.mDescriptorDone)
        {
            break;
        }

        ASSERT(NULL != mRx_Counter      [mRx_Out]);
        ASSERT(NULL != mRx_PacketData   [mRx_Out]);
        ASSERT(NULL != mRx_PacketInfo_MA[mRx_Out]);

        mRx_PacketData   [mRx_Out]->IndicateRxCompleted();
        mRx_PacketInfo_MA[mRx_Out]->mSize_byte = mRx_CA[mRx_Out].mSize_byte;
        mRx_PacketInfo_MA[mRx_Out]->mSendTo    =                          0;

        ( * mRx_Counter[ mRx_Out ] ) --;

        mRx_Out = (mRx_Out + 1) % RX_DESCRIPTOR_QTY;

        mStatistics[OpenNetK::HARDWARE_STATS_RX_packet] ++;
    }
}

unsigned int Pro1000::Rx_GetAvailableDescriptor_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(RX_DESCRIPTOR_QTY > mTx_In );
    ASSERT(RX_DESCRIPTOR_QTY > mTx_Out);

    return ((mRx_Out + RX_DESCRIPTOR_QTY - mRx_In - 1) % RX_DESCRIPTOR_QTY);
}

void Pro1000::Statistics_Update()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_MA);

    mStatistics[OpenNetK::HARDWARE_STATS_RX_BMC_MANAGEMENT_DROPPED_packet      ] += mBAR1_MA->mRx_BmcManagementDropper_packet     ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_CIRCUIT_BREAKER_DROPPED_packet     ] += mBAR1_MA->mRx_CircuitBreakerDropped_packet    ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_HOST_byte                          ] += mBAR1_MA->mRx_HostGoodLow_byte                ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_HOST_packet                        ] += mBAR1_MA->mRx_ToHost_packet                   ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_LENGTH_ERRORS_packet               ] += mBAR1_MA->mRx_LengthErrors_packet             ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_MANAGEMENT_DROPPED_packet          ] += mBAR1_MA->mRx_ManagementDropped_packet        ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_MISSED_packet                      ] += mBAR1_MA->mRx_Missed_packet                   ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_NO_BUFFER_packet                   ] += mBAR1_MA->mRx_NoBuffer_packet                 ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_OVERSIZE_packet                    ] += mBAR1_MA->mRx_Oversize_packet                 ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_QUEUE_DROPPED_packet               ] += mBAR1_MA->mRx_QueueDropPacket0.mFields.mValue ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_UNDERSIZE_packet                   ] += mBAR1_MA->mRx_Undersize_packet                ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_XOFF_packet                        ] += mBAR1_MA->mRx_XOff_packet                     ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_XON_packet                         ] += mBAR1_MA->mRx_XOn_packet                      ;

    mStatistics[OpenNetK::HARDWARE_STATS_TX_DEFER_EVENTS                       ] += mBAR1_MA->mTx_DeferEvents                     ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_DISCARDED_packet                   ] += mBAR1_MA->mTx_Discarded_packet                ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_byte                          ] += mBAR1_MA->mTx_HostGoodLow_byte                ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_packet                        ] += mBAR1_MA->mTx_HostGood_packet                 ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_CIRCUIT_BREAKER_DROPPED_packet] += mBAR1_MA->mTx_HostCircuitBreakerDropped_packet;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_NO_CRS_packet                      ] += mBAR1_MA->mTx_NoCrs_packet                    ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_XOFF_packet                        ] += mBAR1_MA->mTx_XOff_packet                     ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_XON_packet                         ] += mBAR1_MA->mTx_XOn_packet                      ;
}

// Level   Thread
// Thread  Init
void Pro1000::Tx_Config_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_MA);

    mBAR1_MA->mTx_DescriptorBaseAddressHigh0 = (mTx_PA >> 32) & 0xffffffff;
    mBAR1_MA->mTx_DescriptorBaseAddressLow0  =  mTx_PA        & 0xffffffff;

    mBAR1_MA->mTx_DescriptorRingLength0.mFields.mValue_bytes = sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY;

    mBAR1_MA->mTx_PacketBufferSize.mFields.mValue_KB = 20;

    Pro1000_Tx_DescriptorControl lTXDCTL;

    lTXDCTL.mValue = mBAR1_MA->mTx_DescriptorControl0.mValue;

    lTXDCTL.mFields.mHostThreshold      =   16;
    lTXDCTL.mFields.mPrefetchThreshold  =   16;
    lTXDCTL.mFields.mWriteBackThreshold =   16;
    lTXDCTL.mFields.mQueueEnable        = true;

    mBAR1_MA->mTx_DescriptorControl0.mValue = lTXDCTL.mValue;

    mBAR1_MA->mTx_Control.mFields.mEnable = true;
}

// CRITICAL PATH
//
// Level   SoftInt
// Thread  SoftInt
void Pro1000::Tx_Process_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL              != mTx_CA );
    ASSERT(TX_DESCRIPTOR_QTY >  mTx_In );
    ASSERT(TX_DESCRIPTOR_QTY >  mTx_Out);

    while (mTx_In != mTx_Out)
    {
        if (!mTx_CA[mTx_Out].mFields.mDescriptorDone)
        {
            break;
        }

        if (NULL != mTx_Counter[mTx_Out])
        {
            ( * mTx_Counter[ mTx_Out ] ) --;
        }

        mTx_Out = (mTx_Out + 1) % TX_DESCRIPTOR_QTY;

        mStatistics[OpenNetK::HARDWARE_STATS_TX_packet] ++;
    }
}

unsigned int Pro1000::Tx_GetAvailableDescriptor_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(TX_DESCRIPTOR_QTY > mTx_In );
    ASSERT(TX_DESCRIPTOR_QTY > mTx_Out);

    return ((mTx_Out + TX_DESCRIPTOR_QTY - mTx_In - 1) % TX_DESCRIPTOR_QTY);
}
