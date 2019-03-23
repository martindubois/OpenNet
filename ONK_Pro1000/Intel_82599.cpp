
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Intel_82599.cpp

#define __CLASS__     "Intel_82599::"
#define __NAMESPACE__ ""

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Hardware_Statistics.h>
#include <OpenNetK/SpinLock.h>

// ===== ONL_Pro1000 ========================================================
#include "Intel_82599.h"

namespace Intel_82599
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Intel_82599::Intel_82599() : Intel(OpenNetK::ADAPTER_TYPE_HARDWARE_10G), mBAR1_82599_MA(NULL)
    {
        ASSERT(INTEL_82599_BAR1_SIZE == sizeof(BAR1));

        mInfo.mRx_Descriptors = 8192;
        mInfo.mTx_Descriptors = 8192;

        mInfo.mCommonBufferSize_byte += (sizeof(Intel_Rx_Descriptor) * 8192); // Rx packet descriptors
        mInfo.mCommonBufferSize_byte += (sizeof(Intel_Tx_Descriptor) * 8192); // Tx packet descriptors

        strcpy(mInfo.mVersion_Hardware.mComment, "Intel 10 Gb Ethernet Adapter (82599)");
    }

    // ===== OpenNetK::Adapter ==============================================

    // NOT TESTED  ONK_Hardware.Intel.ErrorHandling
    //             Memory 0 too small
    bool Intel_82599::SetMemory(unsigned int aIndex, void * aMemory_MA, unsigned int aSize_byte)
    {
        // TRACE_DEBUG "%s( %u, 0x%p, %u bytes )" DEBUG_EOL, __FUNCTION__, aIndex, aMemory_MA, aSize_byte TRACE_END;

        ASSERT(NULL != aMemory_MA);
        ASSERT(0 < aSize_byte);

        ASSERT(NULL != mZone0);

        switch (aIndex)
        {
        case 0:
            if (sizeof(BAR1) > aSize_byte)
            {
                return false;
            }

            uint32_t lFlags = mZone0->LockFromThread();

            ASSERT(NULL == mBAR1_82599_MA);

            mBAR1_82599_MA = reinterpret_cast<volatile BAR1 *>(aMemory_MA);

            uint32_t lTmp = mBAR1_82599_MA->mRxAddress[0].mRAL;

            mInfo.mEthernetAddress.mAddress[0] = static_cast<uint8_t>( lTmp & 0x000000ff       );
            mInfo.mEthernetAddress.mAddress[1] = static_cast<uint8_t>((lTmp & 0x0000ff00) >>  8);
            mInfo.mEthernetAddress.mAddress[2] = static_cast<uint8_t>((lTmp & 0x00ff0000) >> 16);
            mInfo.mEthernetAddress.mAddress[3] = static_cast<uint8_t>((lTmp & 0xff000000) >> 24);

            lTmp = mBAR1_82599_MA->mRxAddress[0].mRAH;

            mInfo.mEthernetAddress.mAddress[4] = static_cast<uint8_t>(lTmp & 0x000000ff       );
            mInfo.mEthernetAddress.mAddress[5] = static_cast<uint8_t>((lTmp & 0x0000ff00) >> 8);

            mZone0->UnlockFromThread(lFlags);
            break;
        }

        return Intel::SetMemory(aIndex, aMemory_MA, aSize_byte);
    }

    void Intel_82599::D0_Entry()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mZone0);

        Intel::D0_Entry();

        uint32_t lFlags = mZone0->LockFromThread();

        ASSERT(NULL != mBAR1_82599_MA);

        mBAR1_82599_MA->mGPIE.mFields.mMultipleMSIX = true;
        mBAR1_82599_MA->mGPIE.mFields.mPBA_Support  = true;

        mBAR1_82599_MA->mIVAR[0].mFields.mIntAllocVal0 = true;
        mBAR1_82599_MA->mIVAR[0].mFields.mIntAllocVal1 = true;

        mBAR1_82599_MA->mEITR_00[0].mFields.mInterval_2us = 50;

        /* TODO  Dev
        mBAR1_82599_MA->mInterruptVectorAllocationMisc.mFields.mVector32Valid = true;
        mBAR1_82599_MA->mInterruptVectorAllocationMisc.mFields.mVector33Valid = true;
        */

        // mBAR1->mFlowControlReceiveThresholdHigh.mFields.mReceiveThresholdHigh = 3000;

        // mBAR1->mFlowControlReceiveThresholdLow.mFields.mReceiveThresholdLow = 2000;
        // mBAR1->mFlowControlReceiveThresholdLow.mFields.mXOnEnable           = true;

        // mBAR1->mFlowControlRefreshThreshold.mFields.mValue = 0x8000;

        Rx_Config_Zone0();
        Tx_Config_Zone0();

        mZone0->UnlockFromThread(lFlags);
    }

    void Intel_82599::Interrupt_Enable()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mZone0);

        Intel::Interrupt_Enable();

        uint32_t lFlags = mZone0->LockFromThread();

        ASSERT(NULL != mBAR1_82599_MA);

        EI lReg;

        lReg.mValue = 0;

        lReg.mFields.mRTxQ = 0x0001;

        mBAR1_82599_MA->mEIMS_0.mValue = lReg.mValue;

        mZone0->UnlockFromThread(lFlags);
    }

    bool Intel_82599::Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing)
    {
        // TRACE_DEBUG "%s( %u, 0x%p )" DEBUG_EOL, __FUNCTION__, aMessageId, aNeedMoreProcessing TRACE_END;

        ASSERT(NULL != aNeedMoreProcessing);

        ASSERT(NULL != mBAR1_82599_MA);

        uint32_t lValue = mBAR1_82599_MA->mEICR_0.mValue;
        (void)(lValue);

        return Intel::Interrupt_Process(aMessageId, aNeedMoreProcessing);
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    // ===== Intel ==========================================================

    void Intel_82599::Interrupt_Disable_Zone0()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mBAR1_82599_MA);

        mBAR1_82599_MA->mEIMC_0.mValue = 0xffffffff;
    }

    void Intel_82599::Reset_Zone0()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mBAR1_82599_MA);

        Intel::Reset_Zone0();

        uint32_t lValue = mBAR1_82599_MA->mEICR_0.mValue;
        (void)(lValue);
    }

    void Intel_82599::Statistics_Update()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mBAR1_82599_MA);

        /* TODO  Dev
        mStatistics[OpenNetK::HARDWARE_STATS_RX_BMC_MANAGEMENT_DROPPED_packet] += mBAR1_82599_MA->mRx_BmcManagementDropper_packet;
        */
        mStatistics[OpenNetK::HARDWARE_STATS_RX_HOST_byte  ] += mBAR1_82599_MA->mDmaGoodRxLow_byte;
        mStatistics[OpenNetK::HARDWARE_STATS_RX_HOST_packet] += mBAR1_82599_MA->mDmaGoodRx_packet ;

        /* TODO  Dev
        mStatistics[OpenNetK::HARDWARE_STATS_RX_LENGTH_ERRORS_packet] += mBAR1_82599_MA->mRx_LengthErrors_packet;
        mStatistics[OpenNetK::HARDWARE_STATS_RX_NO_BUFFER_packet] += mBAR1_82599_MA->mRx_NoBuffer_packet;
        mStatistics[OpenNetK::HARDWARE_STATS_RX_QUEUE_DROPPED_packet] += mBAR1_82599_MA->mRx_QueueDropPacket0.mFields.mValue;
        */

        mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_byte  ] += mBAR1_82599_MA->mDmaGoodTxLow_byte;
        mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_packet] += mBAR1_82599_MA->mDmaGoodTx_packet ;

        /* TODO  Dev
        mStatistics[OpenNetK::HARDWARE_STATS_TX_NO_CRS_packet] += mBAR1_82599_MA->mTx_NoCrs_packet;
        */
    }

    // ===== OpenNetK::Adapter ==============================================

    void Intel_82599::Unlock_AfterReceive_Internal()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mBAR1_82599_MA);

        mBAR1_82599_MA->mRxQueue_00[0].mRDT = mRx_In;
    }

    void Intel_82599::Unlock_AfterSend_Internal()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__, TRACE_END;

        ASSERT(NULL != mBAR1_82599_MA);

        mBAR1_82599_MA->mTxQueue[0].mTDT = mTx_In;
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    // Level   Thread
    // Thread  Init
    void Intel_82599::Rx_Config_Zone0()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mBAR1_82599_MA);
        ASSERT(0 < mConfig.mPacketSize_byte);

        MulticastArray_Clear_Zone0();

        // TODO  OpenNet.Adapter
        //       Low (Feature) - Add configuration field for:
        //       BroadcastAcceptMode, MulticastPromiscuousEnabled,
        //       PassMacControlFrames, StoreBadPackets, UnicastPromiscuousEnabled

        mBAR1_82599_MA->mFCtrl.mFields.mBAM = true ;
        mBAR1_82599_MA->mFCtrl.mFields.mMPE = true ;
        mBAR1_82599_MA->mFCtrl.mFields.mSBP = false;
        mBAR1_82599_MA->mFCtrl.mFields.mUPE = true ;

        mBAR1_82599_MA->mRDRxCtl.mFields.mCRCStrip = true;

        mBAR1_82599_MA->mMFlCn.mFields.mDPF  = true ;
        mBAR1_82599_MA->mMFlCn.mFields.mPMCF = false;

        mBAR1_82599_MA->mHLReg0.mFields.mJumboEn      = true;
        mBAR1_82599_MA->mHLReg0.mFields.mRxCrcStrip   = true;
        mBAR1_82599_MA->mHLReg0.mFields.mRxPadStripEn = true;

        MaxFrS lMaxFrS;

        lMaxFrS.mValue = 0;
        lMaxFrS.mFields.mMFS_byte = mConfig.mPacketSize_byte;

        mBAR1_82599_MA->mMaxFrS.mValue = lMaxFrS.mValue;;

        /* TODO  Dev
        mBAR1_82599_MA->mRx_DmaMaxOutstandingData.mFields.mValue_256_bytes = 0xfff;

        mBAR1_82599_MA->mRx_LongPacketMaximumLength.mFields.mValue_byte = mConfig.mPacketSize_byte;
        */

        mBAR1_82599_MA->mRxQueue_00[0].mRDBAH      = (mRx_PA >> 32) & 0xffffffff;
        mBAR1_82599_MA->mRxQueue_00[0].mRDBAL      = mRx_PA & 0xffffffff;
        mBAR1_82599_MA->mRxQueue_00[0].mRDLen_byte = sizeof(Intel_Rx_Descriptor) * mInfo.mRx_Descriptors;

        mBAR1_82599_MA->mRxQueue_00[0].mSRRCtl.mFields.mBSizePacket_KiB     = mConfig.mPacketSize_byte / 1024;
        mBAR1_82599_MA->mRxQueue_00[0].mSRRCtl.mFields.mDescType            = 0;
        mBAR1_82599_MA->mRxQueue_00[0].mSRRCtl.mFields.mDropEn              = true;
        mBAR1_82599_MA->mRxQueue_00[0].mSRRCtl.mFields.mBSizeHeader_64bytes = 0;

        mBAR1_82599_MA->mRxQueue_00[0].mRxDCtl.mFields.mEnable = true;

        mBAR1_82599_MA->mRxCtrl.mFields.mRxEn = true;
    }

    // Level   Thread
    // Thread  Init
    void Intel_82599::Tx_Config_Zone0()
    {
        // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mBAR1_82599_MA);

        mBAR1_82599_MA->mTxQueue[0].mTDBAH      = (mTx_PA >> 32) & 0xffffffff;
        mBAR1_82599_MA->mTxQueue[0].mTDBAL      = mTx_PA & 0xffffffff;
        mBAR1_82599_MA->mTxQueue[0].mTDLen_byte = sizeof(Intel_Rx_Descriptor) * mInfo.mRx_Descriptors;

        /* TODO  Dev
        mBAR1_82599_MA->mTx_PacketBufferSize.mFields.mValue_KB = 20;
        */

        mBAR1_82599_MA->mDmaTxCtl.mFields.mTE = true;

        TxDCtl lTxDCtl;

        lTxDCtl.mValue = mBAR1_82599_MA->mTxQueue[0].mTxDCtl.mValue;

        lTxDCtl.mFields.mHThresh = 16;
        lTxDCtl.mFields.mPThresh = 16;
        lTxDCtl.mFields.mWThresh = 16;
        lTxDCtl.mFields.mEnable  = true;

        mBAR1_82599_MA->mTxQueue[0].mTxDCtl.mValue = lTxDCtl.mValue;
    }

}
