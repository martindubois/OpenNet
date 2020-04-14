
// Author     KMS - Martin Dubois, P.Eng.
// Copyright  (C) 2018-2020 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Intel_Regs.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== ONK_Pro1000 ========================================================
#include "Regs.h"

// Data types
/////////////////////////////////////////////////////////////////////////////

// ===== Descriptors ========================================================

typedef struct
{
    uint64_t                mLogicalAddress;
    volatile unsigned short mSize_byte     ;
    volatile unsigned short mCheckSum      ;
    volatile struct
    {
        unsigned mDescriptorDone : 1;
        unsigned mEndOfPacket    : 1;

        unsigned mReserved0 : 1;

        unsigned mVLanPacket        : 1;
        unsigned mUdpCheckSum       : 1;
        unsigned mTcpUdpCheckSum    : 1;
        unsigned mIPv4CheckSum      : 1;
        unsigned mPassedExactFilter : 1;

        unsigned mRxError : 1; // 82599 only

        unsigned mReserved1 : 4;

        unsigned mTcpUdpCheckSumError : 1; // 82576 only
        unsigned mIPv4CheckSumError   : 1; // 82599 - TcpUdpCheckSumError
        unsigned mDataError           : 1; // 82599 - IPv4CheckSumError

        unsigned mVLan : 12;
        unsigned mCFI  :  1;
        unsigned mPRI  :  3;
    }
    mFields;
}
Intel_Rx_Descriptor;

typedef struct {
    uint64_t mLogicalAddress;
    volatile struct
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
Intel_Tx_Descriptor;

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
Intel_DeviceControl;

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
Intel_DeviceStatus;

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
Intel_Rx_CheckSumControl;

// ===== Groups =============================================================

typedef struct
{
    uint32_t mRAL; // 0x00

    struct
    {
        unsigned mRAH           : 16;
        unsigned mAddressSelect :  2;
        unsigned mPoolSelect    :  8;

        unsigned mReserved : 5;

        unsigned mAddressValid : 1;
    }
    mFields;
}
Intel_Rx_Address;

// ===== BARs ===============================================================

typedef struct
{

    Intel_DeviceControl mDeviceControl;// 0x00000

    REG_RESERVED(00004, 00008);

    Intel_DeviceStatus mDeviceStatus; // 0x00008

    REG_RESERVED(0000c, 04000);

                                  //             82576      82599
    uint32_t mRx_CrcError_packet; // 0x04000 - Page 581 - Page 687

    REG_RESERVED(04004, 04040);

                                     //             82576      82599
    uint32_t mRx_LengthError_packet; // 0x04040            - Page 688

    REG_RESERVED(04044, 0405c);

                                        //             82576      82599
    uint32_t mRx_64_Bytes_packet      ; // 0x0405c            - Page 690
    uint32_t mRx_64_127_Bytes_packet  ; // 0x04060            - Page 691
    uint32_t mRx_128_255_Bytes_packet ; // 0x04064
    uint32_t mRx_256_511_Bytes_packet ; // 0x04068 - Page 587
    uint32_t mRx_512_1023_Bytes_packet; // 0x0406c
    uint32_t mRx_1024_Max_Bytes_packet; // 0x04070 - Page 588 - Page 692
    uint32_t mRx_Good_packet          ; // 0x04074
    uint32_t mRx_Broadcast_packet     ; // 0x04078            - Page 691
    uint32_t mRx_Multicast_packet     ; // 0x0407c
    uint32_t mTx_Good_packet          ; // 0x04080            - Page 696

    REG_RESERVED(04084, 04088);

                                //             82576      82599
    uint32_t mRx_GoodLow_byte ; // 0x04088 - Page 589 - Page 693
    uint32_t mRx_GoodHigh_byte; // 0x0408c
    uint32_t mTx_GoodLow_byte ; // 0x04090
    uint32_t mTx_GoodHigh_byte; // 0x04094            - Page 694

    REG_RESERVED(04098, 040a4);

                                           //             82576      82599
    uint32_t mRx_Undersize_packet        ; // 0x040a4            - Page 697
    uint32_t mRx_Fragment_packet         ; // 0x040a8            - Page 698
    uint32_t mRx_Oversize_packet         ; // 0x040ac
    uint32_t mRx_Jabber_packet           ; // 0x040b0 - Page 591
    uint32_t mRx_Management_packet       ; // 0x040b4
    uint32_t mRx_ManagementDropped_packet; // 0x040b8 - Page 592

    REG_RESERVED(040bc, 040c0);

                                 //             82576      82599
    uint32_t mRx_TotatLow_byte ; // 0x040c0            - Page 699
    uint32_t mRx_TotalHigh_byte; // 0x040c4 - Page 593

    REG_RESERVED(040c8, 040cc);

                                        //             82576      82599
    uint32_t mTx_TotalHigh_byte       ; // 0x040cc
    uint32_t mRx_Total_packet         ; // 0x040d0
    uint32_t mTx_Total_packet         ; // 0x040d4 - Page 594
    uint32_t mTx_64_Bytes_packet      ; // 0x040d8            - Page 700
    uint32_t mTx_65_127_Bytes_packet  ; // 0x040dc
    uint32_t mTx_128_255_Bytes_packet ; // 0x040e0 - Page 595
    uint32_t mTx_256_511_Bytes_packet ; // 0x040e4
    uint32_t mTx_512_1023_Bytes_packet; // 0x040e8            - Page 701
    uint32_t mTx_1024_Max_Bytes_packet; // 0x040ec
    uint32_t mTx_Multicast_packet     ; // 0x040f0 - Page 596
    uint32_t mTx_Broadcast_packet     ; // 0x040f4

    REG_RESERVED(040f8, 05000);
    
                                                  //             82576      82599
    Intel_Rx_CheckSumControl mRx_CheckSumControl; // 0x05000 - Page 585

    REG_RESERVED(05004, 05200);

    uint32_t mMulticastTableArray[128]; // 0x05200

    REG_RESERVED(05400, 05840);

    uint32_t mIPv4AddressTable0; // 0x05840

    REG_RESERVED(05844, 05880);

    uint32_t mIPv6AddressTable[4]; // 0x05880

    REG_RESERVED(05890, 09000);

    uint32_t mFlexibleFilterTable[256]; // 0x09000

}
Intel_BAR1;
