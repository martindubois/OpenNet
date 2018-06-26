
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Pro1000_Regs.h

#pragma once

// Data types
/////////////////////////////////////////////////////////////////////////////

// ===== Descriptors ========================================================

typedef struct
{
    uint64_t       mLogicalAddress;
    unsigned short mSize_byte     ;
    unsigned short mCheckSum      ;
    struct
    {
        unsigned mDescriptorDone : 1;
        unsigned mEndOfPacket    : 1;

        unsigned mReserved0 : 1;

        unsigned mVLanPacket        : 1;
        unsigned mUdpCheckSum       : 1;
        unsigned mTcpUdpCheckSum    : 1;
        unsigned mIPv4CheckSum      : 1;
        unsigned mPassedExactFilter : 1;

        unsigned mReserved1 : 5;

        unsigned mTcpUdpCheckSumError : 1;
        unsigned mIPv4CheckSumError   : 1;
        unsigned mDataError           : 1;

        unsigned mVLan : 12;
        unsigned mCFI  :  1;
        unsigned mPRI  :  3;
    }
    mFields;
}
Pro1000_Rx_Descriptor;

typedef struct {
    uint64_t       mLogicalAddress;
    struct
    {
        unsigned mSize_byte      : 16;
        unsigned mCheckSumOffset :  8;
        unsigned mEndOfPacket    :  1;
        unsigned mInsertCRC      :  1;
        unsigned mInsertCheckSum :  1;
        unsigned mReportStatus   :  1;

        unsigned mReserved0 : 1;

        unsigned mDescriptorExtension : 1;
        unsigned mVLanPacketEnable    : 1;

        unsigned mReserved1 : 1;

        unsigned mDescriptorDone : 1;

        unsigned mReserved2 : 7;

        unsigned mCheckSumStart :  8;
        unsigned mVLanId        : 12;
        unsigned mCFI           :  1;
        unsigned mPRI           :  3;
    }
    mFields;
 }
Pro1000_Tx_Descriptor;

// ===== Registers ==========================================================

typedef union
{
    uint32_t mValue;

    struct
    {
        unsigned mFullDuplex : 1;

        unsigned mReserved0 : 1;

        unsigned mGioMasterDisable : 1;
        unsigned mLinkReset        : 1;

        unsigned mReserved1 : 2;

        unsigned mSetLinkUp          : 1;
        unsigned mInvertLossOfSignal : 1;
        unsigned mSpeed              : 2;

        unsigned mReserved2 : 1;

        unsigned mForceSpeed  : 1;
        unsigned mForceDuplex : 1;

        unsigned mReserved3 : 13;

        unsigned mReset                : 1;
        unsigned mRx_FlowControlEnable : 1;
        unsigned mTx_FlowControlEnable : 1;

        unsigned mReserved4 : 1;

        unsigned mVLanModeEnable : 1;
        unsigned mPhyReset       : 1;
    }
    mFields;
}
Pro1000_DeviceControl;

typedef union
{
    uint32_t mValue;

    struct
    {
        unsigned mFullDuplex : 1;
        unsigned mLinkUp     : 1;

        unsigned mReserved0 : 1;

        unsigned mTx_Off : 1;

        unsigned mReserved1 : 1;

        unsigned mSpeed                   : 2;
        unsigned mAutoSpeedDetectionValue : 2;

        unsigned mReserved2 : 23;
    }
    mFields;
}
Pro1000_DeviceStatus;

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
Pro1000_GeneralPurposeInterruptEnable;

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
Pro1000_InterruptMask;

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
Pro1000_InterruptThrottle;

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
Pro1000_InterruptVectorAllocation;

typedef union
{
    uint32_t mValue;

    struct
    {
        unsigned mVector32 : 5;

        unsigned mReserved0 : 2;

        unsigned mVector32Valid : 1;
        unsigned mVector33 : 5;

        unsigned mReserved1 : 2;

        unsigned mVector33Valid : 1;

        unsigned mReserved2 : 16;
    }
    mFields;
}
Pro1000_InterruptVectorAllocationMisc;

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
Pro1000_MdiControl;

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
Pro1000_Rx_AddressHigh;

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
Pro1000_Rx_AddressLow;

typedef union
{
    uint32_t mValue;

    struct
    {
        unsigned mPacketStar          : 8;
        unsigned mIpOffLoadEnable     : 1;
        unsigned mTcpUdpOffLoadEnable : 1;

        unsigned mReserved0 : 1;

        unsigned mCrcEnable       : 1;
        unsigned mIpPayloadEnable : 1;
        unsigned mPacketDisable   : 1;

        unsigned mReserved1 : 18;
    }
    mFields;
}
Pro1000_Rx_CheckSumControl;

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
Pro1000_Rx_Control;

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
Pro1000_Rx_DescriptorControl;

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
Pro1000_Rx_DescriptorHead;

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
Pro1000_Rx_DescriptorRingLength;

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
Pro1000_Rx_DescriptorTail;

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
Pro1000_Rx_LongPacketMaximumLength;

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
Pro1000_Rx_PacketBufferSize;

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
Pro1000_Rx_QueueDropPacket;

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
Pro1000_Rx_SplitAndReplicationControl;

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
Pro1000_SwitchPacketBufferSize;

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
Pro1000_Tx_Control;

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
Pro1000_Tx_DescriptorControl;

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
Pro1000_Tx_DescriptorHead;

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
Pro1000_Tx_DescriptorRingLength;

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
Pro1000_Tx_DescriptorTail;

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
Pro1000_Tx_DmaControl;

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
Pro1000_Tx_InterPacketGap;

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
Pro1000_Tx_PacketBufferSize;

// ===== BARs ===============================================================

typedef struct
{

    Pro1000_DeviceControl mDeviceControl;// 0x00000

    uint32_t mReserved_00004[(0x00008 - 0x00004) / 4];

    Pro1000_DeviceStatus mDeviceStatus; // 0x00008

    uint32_t mReserved_0000c[(0x00020 - 0x0000c) / 4];

    Pro1000_MdiControl mMdiControl; // 0x00020

    uint32_t mReserved_00024[(0x00100 - 0x00024) / 4];

    Pro1000_Rx_Control mRx_Control; // 0x00100

    uint32_t mReserved_00104[(0x00400 - 0x00104) / 4];

    Pro1000_Tx_Control mTx_Control; // 0x00400

    uint32_t mReserved_00404[(0x00410 - 0x00404) / 4];

    Pro1000_Tx_InterPacketGap mTx_InterPacketGap; // 0x00410

    uint32_t mReserved_00414[(0x01500 - 0x00414) / 4];

    Pro1000_InterruptMask mInterruptCauseRead; // 0x01500

    uint32_t mReserved_01504[(0x01508 - 0x01504) / 4];

    Pro1000_InterruptMask                 mInterruptMaskSet             ; // 0x01508
    Pro1000_InterruptMask                 mInterruptMaskClear           ; // 0x0150c
    Pro1000_InterruptMask                 mInterruptAcknowledgeAutoMask ; // 0x01510
    Pro1000_GeneralPurposeInterruptEnable mGeneralPurposeInterruptEnable; // 0x01514

	uint32_t mReserved_01518[(0x01680 - 0x01518) / 4];

    Pro1000_InterruptThrottle mInterruptTrottle[25]; // 0x01680

    uint32_t mReserved_016e4[(0x01700 - 0x016e4) / 4];

    Pro1000_InterruptVectorAllocation mInterruptVectorAllocation[8]; // 0x01700

    uint32_t mReserved_01720[(0x01740 - 0x01720) / 4];

    Pro1000_InterruptVectorAllocationMisc mInterruptVectorAllocationMisc; // 0x01740

    uint32_t mReserved_01744[(0x02404 - 0x01744) / 4];

    Pro1000_Rx_PacketBufferSize mRx_PacketBufferSize; // 0x02404

    uint32_t mReserved_02408[(0x03004 - 0x02408) / 4];

    Pro1000_SwitchPacketBufferSize mSwitchPacketBufferSize; // 0x03004

    uint32_t mReserved_03008[(0x03404 - 0x03008) / 4];

    Pro1000_Tx_PacketBufferSize mTx_PacketBufferSize; // 0x03404

    uint32_t mReserved_03408[(0x03590 - 0x03408) / 4];

    Pro1000_Tx_DmaControl mTx_DmaControl;// 0x03590

    uint32_t mReserved_03594[(0x04000 - 0x03594) / 4];

    uint32_t mCrcErrorCount          ; // 0x04000
    uint32_t mAlignmentErrorCount    ; // 0x04004
    uint32_t mSymbolErrorCount       ; // 0x04008
    uint32_t mRx_ErrorCount          ; // 0x0400c
    uint32_t mMissedPacketsCount     ; // 0x04010
    uint32_t mSingleCollisionCount   ; // 0x04014
    uint32_t mExcessiveCollisionCount; // 0x04018
    uint32_t mMultipleCollisionCount ; // 0x0401c
    uint32_t mLateCollisionsCount    ; // 0x04020

    uint32_t mReserved_04024[(0x04028 - 0x04024) / 4];

    uint32_t mCollisionCount; // 0x04028

    uint32_t mReserved_0402c[(0x04030 - 0x0402c) / 4];

    uint32_t mDeferCount; // 0x04030
    uint32_t mTx_NoCrs  ; // 0x04034

    uint32_t mReserved_04038[(0x0403c - 0x04038) / 4];

    uint32_t mTx_DiscardedPacket            ; // 0x0403c
    uint32_t mRx_LengthError                ; // 0x04040
    uint32_t mRx_CircuitBreakerDroppedPacket; // 0x04044
    uint32_t mRx_XOn                        ; // 0x04048
    uint32_t mTx_XOn                        ; // 0x0404c
    uint32_t mRx_XOff                       ; // 0x04050
    uint32_t mTx_XOff                       ; // 0x04054
    uint32_t mRx_FlowControlUnsupported     ; // 0x04058
    uint32_t mRx_Packets_64_bytes           ; // 0x0405c
    uint32_t mRx_Packets_64_127_byte        ; // 0x04060
    uint32_t mRx_Packets_128_255_byte       ; // 0x04064
    uint32_t mRx_Packets_256_511_byte       ; // 0x04068
    uint32_t mRx_Packets_512_1023_byte      ; // 0x0406c
    uint32_t mRx_Packets_1024_Max_byte      ; // 0x04070
    uint32_t mRx_GoodPacket                 ; // 0x04074
    uint32_t mRx_BroadcastPacket            ; // 0x04078
    uint32_t mRx_MulticastPacket            ; // 0x0407c
    uint32_t mTx_GoodPacket                 ; // 0x04080

    uint32_t mReserved_04084[(0x04088 - 0x04084) / 4];

    uint32_t mRx_GoodOctetLow ; // 0x04088
    uint32_t mRx_GoodOctetHigh; // 0x0408c
    uint32_t mTx_GoodOctetLow ; // 0x04090
    uint32_t mTx_GoodOctetHigh; // 0x04094

    uint32_t mReserved_04098[(0x040a0 - 0x04098) / 4];

    uint32_t mRx_NoBuffer                         ; // 0x040a0
    uint32_t mRx_Undersize                        ; // 0x040a4
    uint32_t mRx_Fragment                         ; // 0x040a8
    uint32_t mRx_Oversize                         ; // 0x040ac
    uint32_t mRx_Jabber                           ; // 0x040b0
    uint32_t mRx_ManagementPacket                 ; // 0x040b4
    uint32_t mManagementPacketDropped             ; // 0x040b8
    uint32_t mTx_ManagementPacket                 ; // 0x040bc
    uint32_t mRx_TotalOctetLow                    ; // 0x040c0
    uint32_t mRx_TotalOctetHigh                   ; // 0x040c4
    uint32_t mTx_TotalOctetLow                    ; // 0x040c8
    uint32_t mTx_TotalOctetHigh                   ; // 0x040cc
    uint32_t mRx_TotalPacket                      ; // 0x040d0
    uint32_t mTx_TotalPacket                      ; // 0x040d4
    uint32_t mTx_Packet_64_byte                   ; // 0x040d8
    uint32_t mTx_Packet_65_127_byte               ; // 0x040dc
    uint32_t mTx_Packet_128_255_byte              ; // 0x040e0
    uint32_t mTx_Packet_256_511_byte              ; // 0x040e4
    uint32_t mTx_Packet_512_1023_byte             ; // 0x040e8
    uint32_t mTx_Packet_1024_Max_byte             ; // 0x040ec
    uint32_t mTx_MulticastPacket                  ; // 0x040f0
    uint32_t mTx_BroadcastPacket                  ; // 0x040f4
    uint32_t mTx_TcpSegmentationContext           ; // 0x040f8
    uint32_t mRx_CircuitBreakerManageabilityPacket; // 0x040fc
    uint32_t mInterruptionAssertion               ; // 0x04100
    uint32_t mRx_PacketToHost                     ; // 0x04104

    uint32_t mReserved_04108[(0x04118 - 0x04108) / 4];

    uint32_t mTx_HostGoodPacket; // 0x04118

    uint32_t mReserved_0411c[(0x04120 - 0x0411c) / 4];

    uint32_t mRx_DescriptorMinimumThreshold     ; // 0x04120
    uint32_t mTx_HostCircuitBreakerDroppedPacket; // 0x04124
    uint32_t mRx_HostGoodOctetLow               ; // 0x04128
    uint32_t mRx_HostGoodOctetHigh              ; // 0x0412c
    uint32_t mTx_HostGoodOctetLow               ; // 0x04130
    uint32_t mTx_HostGoodOctetHigh              ; // 0x04134
    uint32_t mLengthError                       ; // 0x04138
    uint32_t mRx_BmcManagementPacket            ; // 0x0413c
    uint32_t mRx_BmcManagementPacketDropper     ; // 0x04140
    uint32_t mTx_DmcManagementPacket            ; // 0x04144

    uint32_t mReserved_04148[(0x05000 - 0x04148) / 4];

    Pro1000_Rx_CheckSumControl         mRx_CheckSumControl        ; // 0x05000
    Pro1000_Rx_LongPacketMaximumLength mRx_LongPacketMaximumLength; // 0x05004

    uint32_t mReserved_05008[(0x05200 - 0x05008) / 4];

    uint32_t               mMulticastTableArray[128]; // 0x05200
    Pro1000_Rx_AddressLow  mRx_AddressLow0          ; // 0x05400
    Pro1000_Rx_AddressHigh mRx_AddressHigh0         ; // 0x05404

    uint32_t mReserved_05408[(0x05840 - 0x05408) / 4];

    uint32_t mIPv4AddressTable0; // 0x05840

    uint32_t mReserved_05844[(0x05880 - 0x05844) / 4];

    uint32_t mIPv6AddressTable[4]; // 0x05880

    uint32_t mReserved_05890[(0x09000 - 0x05890) / 4];

    uint32_t mFlexibleFilterTable[256]; // 0x09000

    uint32_t mReserved_09400[(0x0c000 - 0x09400) / 4];

    uint32_t                              mRx_DescriptorBaseAddressLow0 ; // 0x0c000
    uint32_t                              mRx_DescriptorBaseAddressHigh0; // 0x0c004
    Pro1000_Rx_DescriptorRingLength       mRx_DescriptorRingLength0     ; // 0x0c008
    Pro1000_Rx_SplitAndReplicationControl mRx_SplitAndReplicationControl; // 0x0c00c
    Pro1000_Rx_DescriptorHead             mRx_DescriptorHead0           ; // 0x0c010

    uint32_t mReserved_0c014[(0x0c018 - 0x0c014) / 4];

    Pro1000_Rx_DescriptorTail mRx_DescriptorTail0; // 0xc018

    uint32_t mReserved_0c01c[(0x0c028 - 0x0c01c) / 4];

    Pro1000_Rx_DescriptorControl mRx_DescriptorControl0; // 0x0c028
    
    uint32_t mReserved_0c02c[(0x0c030 - 0x0c02c) / 4];

    Pro1000_Rx_QueueDropPacket mRx_QueueDropPacket0; // 0x0c030

    uint32_t mReserved_0c034[(0x0e000 - 0x0c034) / 4];

    uint32_t                        mTx_DescriptorBaseAddressLow0 ; // 0x0e000
    uint32_t                        mTx_DescriptorBaseAddressHigh0; // 0x0e004
    Pro1000_Tx_DescriptorRingLength mTx_DescriptorRingLength0     ; // 0x0e008

    uint32_t mReserved_0e00c[(0x0e010 - 0x0e00c) / 4];

    Pro1000_Tx_DescriptorHead mTx_DescriptorHead0; // 0x0e010

    uint32_t mReserved_0e014[(0x0e018 - 0x0e014) / 4];

    Pro1000_Tx_DescriptorTail mTx_DescriptorTail0; // 0x0e018

    uint32_t mReserved_0e01c[(0x0e028 - 0x0e01c) / 4];

    Pro1000_Tx_DescriptorControl mTx_DescriptorControl0; // 0x0e028
}
Pro1000_BAR1;
