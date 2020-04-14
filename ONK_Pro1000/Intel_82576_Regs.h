
// Author     KMS - Martin Dubois, P.Eng.
// Copyright  (C) 2018-2020 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Intel_82576_Regs.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== ONK_Pro1000 ========================================================
#include "Regs.h"

namespace Intel_82576
{

    // Data types
    /////////////////////////////////////////////////////////////////////////

    // ===== Registers ======================================================

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 4;

            unsigned mReceiveThresholdHigh : 12;

            unsigned mReserved1 : 16;
        }
        mFields;
    }
    FlowControlReceiveThresholdHigh;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 4;

            unsigned mReceiveThresholdLow : 12;

            unsigned mReserved1 : 15;

            unsigned mXOnEnable : 1;
        }
        mFields;
    }
    FlowControlReceiveThresholdLow;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue : 16;

            unsigned mReserved0 : 16;
        }
        mFields;
    }
    FlowControlRefreshThreshold;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mNonSelectiveInterruptClearOnRead : 1;

            unsigned mReserved0 : 3;

            unsigned mMultipleMSIX : 1;

            unsigned mReserved1 : 2;

            unsigned mLowLatencyCredits : 5;

            unsigned mReserved2 : 18;

            unsigned mExtendedInterruptAutoMaskEnable : 1;
            unsigned mPbaSupport                      : 1;
        }
        mFields;
    }
    GeneralPurposeInterruptEnable;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mTx_DescriptorWritten : 1;

            unsigned mReserved0 : 1;

            unsigned mLinkStatusChange : 1;

            unsigned mReserved1 : 1;

            unsigned mRx_DescriptorMinimumThreshold : 1;

            unsigned mReserved2 : 1;

            unsigned mRx_Overrun           : 1;
            unsigned mRx_DescriptorWritten : 1;

            unsigned mReserved3 : 24;
        }
        mFields;
    }
    InterruptMask;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 2;

            unsigned mInterval_us : 13;

            unsigned mLliModerationEnable :  1;
            unsigned mLlCounter           :  5;
            unsigned mModerationCounter   : 10;
            unsigned mCounterIgnore       :  1;
        }
        mFields;
    }
    InterruptThrottle;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mVector0 : 5;

            unsigned mReserved0 : 2;

            unsigned mVector0Valid : 1;
            unsigned mVector1      : 5;

            unsigned mReserved1 : 2;

            unsigned mVector1Valid : 1;
            unsigned mVector2      : 5;

            unsigned mReserved2 : 2;

            unsigned mVector2Valid : 1;
            unsigned mVector3      : 5;

            unsigned mReserved3 : 2;

            unsigned mVector3Valid : 1;
        }
        mFields;
    }
    InterruptVectorAllocation;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mVector32 : 5;

            unsigned mReserved0 : 2;

            unsigned mVector32Valid : 1;
            unsigned mVector33      : 5;

            unsigned mReserved1 : 2;

            unsigned mVector33Valid : 1;

            unsigned mReserved2 : 16;
        }
        mFields;
    }
    InterruptVectorAllocationMisc;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mData            : 16;
            unsigned mRegisterAddress :  5;
            unsigned mPhyAddress      :  5;
            unsigned mOpCode          :  2;
            unsigned mReady           :  1;
            unsigned mInterruptEnable :  1;
            unsigned mError           :  1;
            unsigned mDestination     :  1;
        }
        mFields;
    }
    MdiControl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mE : 8;
            unsigned mF : 8;

            unsigned mReserved0 : 15;

            unsigned mAddressValid : 1;
        }
        mFields;
    }
    Rx_AddressHigh;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mA : 8;
            unsigned mB : 8;
            unsigned mC : 8;
            unsigned mD : 8;
        }
        mFields;
    }
    Rx_AddressLow;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 1;

            unsigned mEnable                      : 1;
            unsigned mStoreBadPackets             : 1;
            unsigned mUnicastPromiscuousEnabled   : 1;
            unsigned mMulticastPromiscuousEnabled : 1;
            unsigned mLongPacketEnabled           : 1;
            unsigned mLoopbackMode                : 2;

            unsigned mReserved1 : 4;

            unsigned mMulticastOffset : 2;

            unsigned mReserved2 : 1;

            unsigned mBroadcastAcceptMode : 1;

            unsigned mBufferSize                  : 2;
            unsigned mVLanFilerEnable             : 1;
            unsigned mCanonialFormIndicatorEnable : 1;
            unsigned mCanonialFormIndicator       : 1;
            unsigned mPadSmallPackets             : 1;
            unsigned mDiscardPauseFrames          : 1;
            unsigned mPassMacControlFrames        : 1;

            unsigned mReserved3 : 2;

            unsigned mStripEthernetCRC : 1;

            unsigned mReserved4 : 5;
        }
        mFields;
    }
    Rx_Control;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mPrefetchThreshold : 5;

            unsigned mReserved0 : 3;

            unsigned mHostThreshold : 5;

            unsigned mReserved1 : 3;

            unsigned mWriteBackThreshold : 5;

            unsigned mReserved2 : 4;

            unsigned mQueueEnable   : 1;
            unsigned mSoftwareFlush : 1;

            unsigned mReserved : 5;
        }
        mFields;
    }
    Rx_DescriptorControl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue : 16;

            unsigned mReserved0 : 16;
        }
        mFields;
    }
    Rx_DescriptorHead;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue_byte : 20;

            unsigned mReserved0 : 12;
        }
        mFields;
    }
    Rx_DescriptorRingLength;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue : 16;

            unsigned mReserved0 : 16;
        }
        mFields;
    }
    Rx_DescriptorTail;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue_256_bytes : 12;

            unsigned mReserved0 : 20;
        }
        mFields;
    }
    Rx_DmaMaxOutstandingData;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue_byte : 14;

            unsigned mReserved0 : 18;
        }
        mFields;
    }
    Rx_LongPacketMaximumLength;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue_KB : 7;

            unsigned mReserved0 : 25;
        }
        mFields;
    }
    Rx_PacketBufferSize;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue : 12;

            unsigned mReserved0 : 20;
        }
        mFields;
    }
    Rx_QueueDropPacket;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mPacketSize_KB : 7;

            unsigned mReserved0 : 1;

            unsigned mHeaderSize_64bytes : 4;

            unsigned mReserved1 : 8;

            unsigned mDescriptorMinimumThresholdSize : 5;
            unsigned mDescriptorType                 : 3;

            unsigned mReserved2 : 3;

            unsigned mDropEnabled : 1;
        }
        mFields;
    }
    Rx_SplitAndReplicationControl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue_KB : 5;

            unsigned mReserved0 : 27;
        }
        mFields;
    }
    SwitchPacketBufferSize;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue        : 10;
            unsigned mPart1        : 10;
            unsigned mAterDeferral : 10;

            unsigned mReserved0 : 2;
        }
        mFields;
    }
    Tx_InterPacketGap;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 1;

            unsigned mEnable : 1;

            unsigned mReserved1 : 1;

            unsigned mPadShortPacket     :  1;
            unsigned mCollisionThreshold :  8;
            unsigned mBackOffSlotTime    : 10;
            unsigned mSoftwareOff        :  1;

            unsigned mReserved2 : 1;

            unsigned mRetryOnLateCollision : 1;

            unsigned mReserved3 : 7;
        }
        mFields;
    }
    Tx_Control;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mPrefetchThreshold : 5;

            unsigned mReserved0 : 3;

            unsigned mHostThreshold : 5;

            unsigned mReserved1 : 3;

            unsigned mWriteBackThreshold : 5;

            unsigned mReserved2 : 4;

            unsigned mQueueEnable   : 1;
            unsigned mSoftwareFlush : 1;

            unsigned mReserved3 : 5;
        }
        mFields;
    }
    Tx_DescriptorControl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue : 16;

            unsigned mReserved0 : 16;
        }
        mFields;
    }
    Tx_DescriptorHead;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue_bytes : 20;

            unsigned mReserved0 : 12;
        }
        mFields;
    }
    Tx_DescriptorRingLength;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue : 16;

            unsigned mReserved0 : 16;
        }
        mFields;
    }
    Tx_DescriptorTail;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 1;

            unsigned mNoSnoopHeaderBufferOsTSO        :  1;
            unsigned m8023LengthLocation              :  1;
            unsigned mAddVLanLocation                 :  1;
            unsigned mOutOfSyncEnable                 :  1;
            unsigned mMaliciousDriverProtectionEnable :  1;
            unsigned mSpoofInt                        :  1;
            unsigned mDefaultCtsTag                   : 16;

            unsigned mReserved1 : 8;
        }
        mFields;
    }
    Tx_DmaControl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mValue_KB : 6;

            unsigned mReserved0 : 26;
        }
        mFields;
    }
    Tx_PacketBufferSize;

    // ===== BARs ===========================================================

    typedef struct
    {

        REG_RESERVED(00000, 00020);

        MdiControl mMdiControl; // 0x00020

        REG_RESERVED(00024, 00100);

        Rx_Control mRx_Control; // 0x00100

        REG_RESERVED(00104, 00400);

        Tx_Control mTx_Control; // 0x00400

        REG_RESERVED(00404, 00410);

        Tx_InterPacketGap mTx_InterPacketGap; // 0x00410

        REG_RESERVED(00414, 01500);

        InterruptMask mInterruptCauseRead; // 0x01500

        REG_RESERVED(01504, 01508);

        InterruptMask                 mInterruptMaskSet             ; // 0x01508
        InterruptMask                 mInterruptMaskClear           ; // 0x0150c
        InterruptMask                 mInterruptAcknowledgeAutoMask ; // 0x01510
        GeneralPurposeInterruptEnable mGeneralPurposeInterruptEnable; // 0x01514

        REG_RESERVED(01518, 01680);

        InterruptThrottle mInterruptTrottle[25]; // 0x01680

        REG_RESERVED(016e4, 01700);

        InterruptVectorAllocation mInterruptVectorAllocation[8]; // 0x01700

        REG_RESERVED(01720, 01740);

        InterruptVectorAllocationMisc mInterruptVectorAllocationMisc; // 0x01740

        REG_RESERVED(01744, 02160);

        FlowControlReceiveThresholdLow mFlowControlReceiveThresholdLow; // 0x02160 - page 488

        REG_RESERVED(02164, 02168);

        FlowControlReceiveThresholdHigh mFlowControlReceiveThresholdHigh; // 0x02168 - page 489

        REG_RESERVED(0216c, 02404);

        Rx_PacketBufferSize mRx_PacketBufferSize; // 0x02404

        REG_RESERVED(02408, 02460);

        FlowControlRefreshThreshold mFlowControlRefreshThreshold; // 0x02460

        REG_RESERVED(02464, 02540);

        Rx_DmaMaxOutstandingData mRx_DmaMaxOutstandingData; // 0x2540 - Page 524

        REG_RESERVED(02544, 03004);

        SwitchPacketBufferSize mSwitchPacketBufferSize; // 0x03004

        REG_RESERVED(03008, 03404);

        Tx_PacketBufferSize mTx_PacketBufferSize; // 0x03404

        REG_RESERVED(03408, 03590);

        Tx_DmaControl mTx_DmaControl;// 0x03590

        REG_RESERVED(03594, 04004);

        uint32_t mRx_AlignmentError_packet    ; // 0x04004 - Page 582
        uint32_t mRx_SymbolError_packet       ; // 0x04008
        uint32_t mRx_Error_packet             ; // 0x0400c
        uint32_t mRx_Missed_packet            ; // 0x04010
        uint32_t mTx_SingleCollision_packet   ; // 0x04014
        uint32_t mTx_ExcessiveCollision_packet; // 0x04018 - Page 583
        uint32_t mTx_MultipleCollision_packet ; // 0x0401c
        uint32_t mTx_LateCollision_packet     ; // 0x04020

        REG_RESERVED(04024, 04028);

        uint32_t mTx_Collisions; // 0x04028

        REG_RESERVED(0402c, 04030);
    
        uint32_t mTx_DeferEvents ; // 0x04030
        uint32_t mTx_NoCrs_packet; // 0x04034 - Page 584

        REG_RESERVED(04038, 0403c);

        uint32_t mTx_Discarded_packet; // 0x0403c

        REG_RESERVED(04040, 04044);

        uint32_t mRx_CircuitBreakerDropped_packet ; // 0x04044 - Page 585
        uint32_t mRx_XOn_packet                   ; // 0x04048
        uint32_t mTx_XOn_packet                   ; // 0x0404c
        uint32_t mRx_XOff_packet                  ; // 0x04050
        uint32_t mTx_XOff_packet                  ; // 0x04054
        uint32_t mRx_FlowControlUnsupported_packet; // 0x04058 - Page 586

        REG_RESERVED(0405c, 040a0);

        uint32_t mRx_NoBuffer_packet; // 0x040a0 - Page 590

        REG_RESERVED(040a4, 040bc);

        uint32_t mTx_Management_packet; // 0x040bc

        REG_RESERVED(040c0, 040c8);

        uint32_t mTx_TotalLow_byte; // 0x040c8

        REG_RESERVED(040cc, 040f8);

        uint32_t mTx_TcpSegmentationContexts           ; // 0x040f8
        uint32_t mRx_CircuitBreakerManageability_packet; // 0x040fc
        uint32_t mInterruptionAssertions               ; // 0x04100 - Page  597
        uint32_t mRx_ToHost_packet                     ; // 0x04104

        REG_RESERVED(04108, 04118);

        uint32_t mTx_HostGood_packet; // 0x04118 - Page 599

        REG_RESERVED(0411c, 04120);

        uint32_t mRx_DescriptorMinimumThreshold      ; // 0x04120
        uint32_t mTx_HostCircuitBreakerDropped_packet; // 0x04124 - Page 600
        uint32_t mRx_HostGoodLow_byte                ; // 0x04128
        uint32_t mRx_HostGoodHigh_byte               ; // 0x0412c
        uint32_t mTx_HostGoodLow_byte                ; // 0x04130
        uint32_t mTx_HostGoodHigh_byte               ; // 0x04134 - Page 601
        uint32_t mRx_LengthErrors_packet             ; // 0x04138
        uint32_t mRx_BmcManagement_packet            ; // 0x0413c - Page 591
        uint32_t mRx_BmcManagementDropper_packet     ; // 0x04140 - Page 592
        uint32_t mTx_DmcManagementPacket             ; // 0x04144

        REG_RESERVED(04148, 05004);

        Rx_LongPacketMaximumLength mRx_LongPacketMaximumLength; // 0x05004

        REG_RESERVED(05008, 05400);

        Intel_Rx_Address  mRxAddress[ 16 ]; // 0x05400

        REG_RESERVED(05480, 0c000);

        uint32_t                      mRx_DescriptorBaseAddressLow0 ; // 0x0c000
        uint32_t                      mRx_DescriptorBaseAddressHigh0; // 0x0c004
        Rx_DescriptorRingLength       mRx_DescriptorRingLength0     ; // 0x0c008
        Rx_SplitAndReplicationControl mRx_SplitAndReplicationControl; // 0x0c00c
        Rx_DescriptorHead             mRx_DescriptorHead0           ; // 0x0c010

        REG_RESERVED(0c014, 0c018);

        Rx_DescriptorTail mRx_DescriptorTail0; // 0xc018

        REG_RESERVED(0c01c, 0c028);

        Rx_DescriptorControl mRx_DescriptorControl0; // 0x0c028

        REG_RESERVED(0c02c, 0c030);

        Rx_QueueDropPacket mRx_QueueDropPacket0; // 0x0c030

        REG_RESERVED(0c034, 0e000);

        uint32_t                            mTx_DescriptorBaseAddressLow0 ; // 0x0e000
        uint32_t                            mTx_DescriptorBaseAddressHigh0; // 0x0e004
        Tx_DescriptorRingLength mTx_DescriptorRingLength0     ; // 0x0e008

        REG_RESERVED(0e00c, 0e010);

        Tx_DescriptorHead mTx_DescriptorHead0; // 0x0e010

        REG_RESERVED(0e014, 0e018);

        Tx_DescriptorTail mTx_DescriptorTail0; // 0x0e018

        REG_RESERVED(0e01c, 0e028);

        Tx_DescriptorControl mTx_DescriptorControl0; // 0x0e028

    }
    BAR1;

}