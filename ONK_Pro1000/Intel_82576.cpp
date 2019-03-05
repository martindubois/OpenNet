
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Intel_82576.cpp

#define __CLASS__     "Intel_82576::"
#define __NAMESPACE__ ""

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Hardware_Statistics.h>
#include <OpenNetK/SpinLock.h>

// ===== ONL_Pro1000 ========================================================
#include "Intel_82576.h"

// Public
/////////////////////////////////////////////////////////////////////////////

Intel_82576::Intel_82576() : mBAR1_82576_MA(NULL)
{
    ASSERT(0x0e02c == sizeof(Intel_82576_BAR1));
}

// ===== OpenNetK::Adapter ==================================================

// NOT TESTED  ONK_Intel.Intel.ErrorHandling
//             Memory 0 too small
bool Intel_82576::SetMemory(unsigned int aIndex, void * aMemory_MA, unsigned int aSize_byte)
{
    // TRACE_DEBUG "%s( %u, 0x%p, %u bytes )" DEBUG_EOL, __FUNCTION__, aIndex, aMemory_MA, aSize_byte TRACE_END;

    ASSERT(NULL != aMemory_MA);
    ASSERT(   0 <  aSize_byte);

    ASSERT(NULL != mZone0);

    switch (aIndex)
    {
    case 0:
        if (sizeof(Intel_82576_BAR1) > aSize_byte)
        {
            return false;
        }

        uint32_t lFlags = mZone0->LockFromThread();

            ASSERT(NULL == mBAR1_82576_MA);

            mBAR1_82576_MA = reinterpret_cast< volatile Intel_82576_BAR1 * >( aMemory_MA );

            mInfo.mEthernetAddress.mAddress[0] = mBAR1_82576_MA->mRx_AddressLow0 .mFields.mA;
            mInfo.mEthernetAddress.mAddress[1] = mBAR1_82576_MA->mRx_AddressLow0 .mFields.mB;
            mInfo.mEthernetAddress.mAddress[2] = mBAR1_82576_MA->mRx_AddressLow0 .mFields.mC;
            mInfo.mEthernetAddress.mAddress[3] = mBAR1_82576_MA->mRx_AddressLow0 .mFields.mD;
            mInfo.mEthernetAddress.mAddress[4] = mBAR1_82576_MA->mRx_AddressHigh0.mFields.mE;
            mInfo.mEthernetAddress.mAddress[5] = mBAR1_82576_MA->mRx_AddressHigh0.mFields.mF;

        mZone0->UnlockFromThread( lFlags );
        break;
    }

    return Intel::SetMemory(aIndex, aMemory_MA, aSize_byte);
}

void Intel_82576::D0_Entry()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mZone0);

    Intel::D0_Entry();

    uint32_t lFlags = mZone0->LockFromThread();

        ASSERT(NULL != mBAR1_82576_MA);

        mBAR1_82576_MA->mGeneralPurposeInterruptEnable.mFields.mExtendedInterruptAutoMaskEnable = true;

        mBAR1_82576_MA->mInterruptVectorAllocation[0].mFields.mVector0Valid = true;
        mBAR1_82576_MA->mInterruptVectorAllocation[0].mFields.mVector1Valid = true;
        mBAR1_82576_MA->mInterruptVectorAllocation[0].mFields.mVector2Valid = true;
        mBAR1_82576_MA->mInterruptVectorAllocation[0].mFields.mVector3Valid = true;

        mBAR1_82576_MA->mInterruptVectorAllocationMisc.mFields.mVector32Valid = true;
        mBAR1_82576_MA->mInterruptVectorAllocationMisc.mFields.mVector33Valid = true;

        mBAR1_82576_MA->mInterruptTrottle[0].mFields.mInterval_us = 100;

        // mBAR1->mFlowControlReceiveThresholdHigh.mFields.mReceiveThresholdHigh = 3000;

        // mBAR1->mFlowControlReceiveThresholdLow.mFields.mReceiveThresholdLow = 2000;
        // mBAR1->mFlowControlReceiveThresholdLow.mFields.mXOnEnable           = true;

        // mBAR1->mFlowControlRefreshThreshold.mFields.mValue = 0x8000;

        Rx_Config_Zone0();
        Tx_Config_Zone0();

    mZone0->UnlockFromThread( lFlags );
}

void Intel_82576::Interrupt_Enable()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mZone0);

    Intel::Interrupt_Enable();

    uint32_t lFlags = mZone0->LockFromThread();

        ASSERT(NULL != mBAR1_82576_MA);

        mBAR1_82576_MA->mInterruptMaskSet.mFields.mTx_DescriptorWritten = true;
        mBAR1_82576_MA->mInterruptMaskSet.mFields.mRx_DescriptorWritten = true;

    mZone0->UnlockFromThread( lFlags );
}

bool Intel_82576::Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing)
{
    // TRACE_DEBUG "%s( %u, 0x%p )" DEBUG_EOL, __FUNCTION__, aMessageId, aNeedMoreProcessing TRACE_END;

    ASSERT(NULL != aNeedMoreProcessing);

    ASSERT(NULL != mBAR1_82576_MA);

    uint32_t lValue = mBAR1_82576_MA->mInterruptCauseRead.mValue;
    (void)(lValue);

    return Intel::Interrupt_Process( aMessageId, aNeedMoreProcessing );
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== Intel ==============================================================

void Intel_82576::Interrupt_Disable_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_82576_MA);

    mBAR1_82576_MA->mInterruptMaskClear.mValue = 0xffffffff;
}

void Intel_82576::Reset_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_82576_MA);

    Intel::Reset_Zone0();

    uint32_t lValue = mBAR1_82576_MA->mInterruptCauseRead.mValue;
    (void)(lValue);
}

void Intel_82576::Statistics_Update()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_82576_MA);

    mStatistics[OpenNetK::HARDWARE_STATS_RX_BMC_MANAGEMENT_DROPPED_packet] += mBAR1_82576_MA->mRx_BmcManagementDropper_packet    ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_HOST_byte                    ] += mBAR1_82576_MA->mRx_HostGoodLow_byte               ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_HOST_packet                  ] += mBAR1_82576_MA->mRx_ToHost_packet                  ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_LENGTH_ERRORS_packet         ] += mBAR1_82576_MA->mRx_LengthErrors_packet            ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_NO_BUFFER_packet             ] += mBAR1_82576_MA->mRx_NoBuffer_packet                ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_QUEUE_DROPPED_packet         ] += mBAR1_82576_MA->mRx_QueueDropPacket0.mFields.mValue;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_byte                    ] += mBAR1_82576_MA->mTx_HostGoodLow_byte               ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_packet                  ] += mBAR1_82576_MA->mTx_HostGood_packet                ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_NO_CRS_packet                ] += mBAR1_82576_MA->mTx_NoCrs_packet                   ;
}

// ===== OpenNetK::Adapter ==================================================

void Intel_82576::Unlock_AfterReceive_Internal()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL              != mBAR1_82576_MA);
    ASSERT(RX_DESCRIPTOR_QTY >  mRx_In        );

    Intel_82576_Rx_DescriptorTail lReg;

    lReg.mValue = 0;
    lReg.mFields.mValue = mRx_In;

    mBAR1_82576_MA->mRx_DescriptorTail0.mValue = lReg.mValue;
}

void Intel_82576::Unlock_AfterSend_Internal()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__, TRACE_END;

    ASSERT(NULL              != mBAR1_82576_MA);
    ASSERT(RX_DESCRIPTOR_QTY >  mTx_In        );

    Intel_82576_Tx_DescriptorTail lReg;

    lReg.mValue = 0;
    lReg.mFields.mValue = mTx_In;

    mBAR1_82576_MA->mTx_DescriptorTail0.mValue = lReg.mValue;
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Level   Thread
// Thread  Init
void Intel_82576::Rx_Config_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_82576_MA          );
    ASSERT(   0 <  mConfig.mPacketSize_byte);

    MulticastArray_Clear_Zone0();

    // TODO  OpenNet.Adapter
    //       Low (Feature) - Add configuration field for:
    //       BroadcastAcceptMode, MulticastPromiscuousEnabled,
    //       PassMacControlFrames, StoreBadPackets, UnicastPromiscuousEnabled
    mBAR1_82576_MA->mRx_Control.mFields.mBroadcastAcceptMode         = true ;
    mBAR1_82576_MA->mRx_Control.mFields.mDiscardPauseFrames          = true ;
    mBAR1_82576_MA->mRx_Control.mFields.mLongPacketEnabled           = true ;
    mBAR1_82576_MA->mRx_Control.mFields.mMulticastPromiscuousEnabled = true ;
    mBAR1_82576_MA->mRx_Control.mFields.mPassMacControlFrames        = false;
    mBAR1_82576_MA->mRx_Control.mFields.mStoreBadPackets             = false;
    mBAR1_82576_MA->mRx_Control.mFields.mStripEthernetCRC            = true ;
    mBAR1_82576_MA->mRx_Control.mFields.mUnicastPromiscuousEnabled   = true ;

    mBAR1_82576_MA->mRx_DmaMaxOutstandingData.mFields.mValue_256_bytes = 0xfff;

    mBAR1_82576_MA->mRx_LongPacketMaximumLength.mFields.mValue_byte = mConfig.mPacketSize_byte;

    mBAR1_82576_MA->mRx_DescriptorBaseAddressHigh0 = (mRx_PA >> 32) & 0xffffffff;
    mBAR1_82576_MA->mRx_DescriptorBaseAddressLow0  =  mRx_PA        & 0xffffffff;

    mBAR1_82576_MA->mRx_DescriptorRingLength0.mFields.mValue_byte = sizeof(Intel_Rx_Descriptor) * RX_DESCRIPTOR_QTY;

    mBAR1_82576_MA->mRx_PacketBufferSize.mFields.mValue_KB = 84;

    mBAR1_82576_MA->mRx_SplitAndReplicationControl.mFields.mDescriptorType     =    0;
    mBAR1_82576_MA->mRx_SplitAndReplicationControl.mFields.mDropEnabled        = true;
    mBAR1_82576_MA->mRx_SplitAndReplicationControl.mFields.mHeaderSize_64bytes =    0;
    mBAR1_82576_MA->mRx_SplitAndReplicationControl.mFields.mPacketSize_KB      = mConfig.mPacketSize_byte / 1024;

    mBAR1_82576_MA->mRx_DescriptorControl0.mFields.mHostThreshold      =   16;
    mBAR1_82576_MA->mRx_DescriptorControl0.mFields.mPrefetchThreshold  =   16;
    mBAR1_82576_MA->mRx_DescriptorControl0.mFields.mWriteBackThreshold =   16;
    mBAR1_82576_MA->mRx_DescriptorControl0.mFields.mQueueEnable        = true;

    mBAR1_82576_MA->mRx_Control.mFields.mEnable = true;
}

// Level   Thread
// Thread  Init
void Intel_82576::Tx_Config_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_82576_MA);

    mBAR1_82576_MA->mTx_DescriptorBaseAddressHigh0 = (mTx_PA >> 32) & 0xffffffff;
    mBAR1_82576_MA->mTx_DescriptorBaseAddressLow0  =  mTx_PA        & 0xffffffff;

    mBAR1_82576_MA->mTx_DescriptorRingLength0.mFields.mValue_bytes = sizeof(Intel_Tx_Descriptor) * TX_DESCRIPTOR_QTY;

    mBAR1_82576_MA->mTx_PacketBufferSize.mFields.mValue_KB = 20;

    Intel_82576_Tx_DescriptorControl lTXDCTL;

    lTXDCTL.mValue = mBAR1_82576_MA->mTx_DescriptorControl0.mValue;

    lTXDCTL.mFields.mHostThreshold      =   16;
    lTXDCTL.mFields.mPrefetchThreshold  =   16;
    lTXDCTL.mFields.mWriteBackThreshold =   16;
    lTXDCTL.mFields.mQueueEnable        = true;

    mBAR1_82576_MA->mTx_DescriptorControl0.mValue = lTXDCTL.mValue;

    mBAR1_82576_MA->mTx_Control.mFields.mEnable = true;
}
