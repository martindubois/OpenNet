
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

    ASSERT(NULL != mBAR1 );
    ASSERT(NULL != mZone0);

    Hardware::GetState(aState);

    mZone0->Lock();

        ASSERT(NULL != mBAR1);

        Pro1000_DeviceStatus lDeviceStatus;

        lDeviceStatus.mValue = mBAR1->mDeviceStatus.mValue;
        // TRACE_DEBUG "%s - STATUS = 0x%08x" DEBUG_EOL, __FUNCTION__, lDeviceStatus.mValue TRACE_END;

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

        // TRACE_DEBUG "%s - EICR = 0x%08x" DEBUG_EOL, __FUNCTION__, reinterpret_cast< volatile uint32_t * >( mBAR1 )[ 0x01580 / 4 ] TRACE_END;
        // TRACE_DEBUG "%s - RWS  = 0x%08x" DEBUG_EOL, __FUNCTION__, reinterpret_cast< volatile uint32_t * >( mBAR1 )[ 0x01048 / 4 ] TRACE_END;
        // TRACE_DEBUG "%s - ICR  = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mInterruptCauseRead.mValue                               TRACE_END;

    mZone0->Unlock();
}

void Pro1000::ResetMemory()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    Hardware::ResetMemory();

    mZone0->Lock();

        mBAR1 = NULL;

    mZone0->Unlock();
}

void Pro1000::SetCommonBuffer(uint64_t aLogical, void * aVirtual)
{
    // TRACE_DEBUG "%s( 0x%llx, 0x%px )" DEBUG_EOL, __FUNCTION__, aLogical, aVirtual TRACE_END;

    ASSERT(NULL != aVirtual);

    ASSERT(NULL != mZone0);

    mZone0->Lock();

        uint64_t  lLogical = aLogical;
        uint8_t * lVirtual = reinterpret_cast<uint8_t *>(aVirtual);

        mRx_Logical = lLogical;
        mRx_Virtual = reinterpret_cast<Pro1000_Rx_Descriptor *>(lVirtual);

        lLogical += sizeof(Pro1000_Rx_Descriptor) * RX_DESCRIPTOR_QTY;
        lVirtual += sizeof(Pro1000_Rx_Descriptor) * RX_DESCRIPTOR_QTY;

        mTx_Logical = lLogical;
        mTx_Virtual = reinterpret_cast<Pro1000_Tx_Descriptor *>(lVirtual);

        lLogical += sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY;
        lVirtual += sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY;

        unsigned int i;

        for (i = 0; i < PACKET_BUFFER_QTY; i++)
        {
            SkipDangerousBoundary(&lLogical, &lVirtual, mConfig.mPacketSize_byte, mPacketBuffer_Logical + i, reinterpret_cast<uint8_t **>(mPacketBuffer_Virtual + i));

            ASSERT(NULL != mPacketBuffer_Virtual[i]);
        }

        for (i = 0; i < TX_DESCRIPTOR_QTY; i++)
        {
            mTx_Virtual[i].mFields.mEndOfPacket  = true;
            mTx_Virtual[i].mFields.mInsertCRC    = true;
            mTx_Virtual[i].mFields.mReportStatus = true;
        }

    mZone0->Unlock();
}

// NOT TESTED  ONK_Pro1000.Pro1000.ErrorHandling
//             Memory 0 too small
bool Pro1000::SetMemory(unsigned int aIndex, void * aVirtual, unsigned int aSize_byte)
{
    // TRACE_DEBUG "%s( %u, 0x%px, %u bytes )" DEBUG_EOL, __FUNCTION__, aIndex, aVirtual, aSize_byte TRACE_END;

    ASSERT(NULL != aVirtual  );
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

            mBAR1 = reinterpret_cast< volatile Pro1000_BAR1 * >( aVirtual );

            Interrupt_Disable_Zone0();

            mInfo.mEthernetAddress.mAddress[0] = mBAR1->mRx_AddressLow0 .mFields.mA;
            mInfo.mEthernetAddress.mAddress[1] = mBAR1->mRx_AddressLow0 .mFields.mB;
            mInfo.mEthernetAddress.mAddress[2] = mBAR1->mRx_AddressLow0 .mFields.mC;
            mInfo.mEthernetAddress.mAddress[3] = mBAR1->mRx_AddressLow0 .mFields.mD;
            mInfo.mEthernetAddress.mAddress[4] = mBAR1->mRx_AddressHigh0.mFields.mE;
            mInfo.mEthernetAddress.mAddress[5] = mBAR1->mRx_AddressHigh0.mFields.mF;

        mZone0->Unlock();
        break;
    }

    return Hardware::SetMemory(aIndex, aVirtual, aSize_byte);
}

void Pro1000::D0_Entry()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1 );
    ASSERT(NULL != mZone0);

    mZone0->Lock();

        mRx_In  = 0;
        mRx_Out = 0;

        mTx_In  = 0;
        mTx_Out = 0;

        memset(&mTx_Counter, 0, sizeof(mTx_Counter));

        mPacketBuffer_In = 0;

        Reset_Zone0();

        Pro1000_DeviceControl lCTRL;

        lCTRL.mValue = mBAR1->mDeviceControl.mValue;
        // TRACE_DEBUG "%s - CTRL (Initial)  = 0x%08x" DEBUG_EOL, __FUNCTION__, lCTRL.mValue TRACE_END;

        lCTRL.mFields.mInvertLossOfSignal   = false;
        // lCTRL.mFields.mRx_FlowControlEnable = true ;
        lCTRL.mFields.mSetLinkUp            = true ;
        // lCTRL.mFields.mTx_FlowControlEnable = true ;

        // TRACE_DEBUG "%s - CTRL (Modified) = 0x%08x" DEBUG_EOL, __FUNCTION__, lCTRL.mValue TRACE_END;
        mBAR1->mDeviceControl.mValue = lCTRL.mValue;

        mBAR1->mGeneralPurposeInterruptEnable.mFields.mExtendedInterruptAutoMaskEnable = true;

        mBAR1->mInterruptVectorAllocation[0].mFields.mVector0Valid = true;
        mBAR1->mInterruptVectorAllocation[0].mFields.mVector1Valid = true;
        mBAR1->mInterruptVectorAllocation[0].mFields.mVector2Valid = true;
        mBAR1->mInterruptVectorAllocation[0].mFields.mVector3Valid = true;

        mBAR1->mInterruptVectorAllocationMisc.mFields.mVector32Valid = true;
        mBAR1->mInterruptVectorAllocationMisc.mFields.mVector33Valid = true;

        mBAR1->mInterruptTrottle[0].mFields.mInterval_us = 100;

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

    ASSERT(NULL != mBAR1 );
    ASSERT(NULL != mZone0);

    Hardware::Interrupt_Enable();

    mZone0->Lock();

        mBAR1->mInterruptMaskSet.mFields.mTx_DescriptorWritten = true;
        mBAR1->mInterruptMaskSet.mFields.mRx_DescriptorWritten = true;

    mZone0->Unlock();
}

// CRITICAL PATH
bool Pro1000::Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing)
{
    // TRACE_DEBUG "%s( %u, 0x%px )" DEBUG_EOL, __FUNCTION__, aMessageId, aNeedMoreProcessing TRACE_END;

    ASSERT(NULL != aNeedMoreProcessing);

    ASSERT(NULL != mBAR1);

    (void)(aMessageId);

    uint32_t lValue = mBAR1->mInterruptCauseRead.mValue; // Reading hardware !!!
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

// CRITICAL PATH - Buffer
void Pro1000::Unlock_AfterReceive(volatile long * aCounter, unsigned int aPacketQty)
{
    // TRACE_DEBUG "%s( 0x%px, %u packets )" DEBUG_EOL, __FUNCTION__, aCounter, aPacketQty TRACE_END;

    ASSERT(NULL != aCounter  );
    ASSERT(0    <  aPacketQty);

    ASSERT(NULL              != mBAR1 );
    ASSERT(RX_DESCRIPTOR_QTY >  mRx_In);

    Hardware::Unlock_AfterReceive(aCounter, aPacketQty);

    Pro1000_Rx_DescriptorTail lReg;

    lReg.mValue         =      0;
    lReg.mFields.mValue = mRx_In;

    mBAR1->mRx_DescriptorTail0.mValue = lReg.mValue; // Writing hardware !

    mStatistics[OpenNetK::HARDWARE_STATS_PACKET_RECEIVE] += aPacketQty;
}

// CRITICAL PATH - Buffer
void Pro1000::Unlock_AfterSend(volatile long * aCounter, unsigned int aPacketQty)
{
    // TRACE_DEBUG "%s( 0x%px, %u packets )" DEBUG_EOL, __FUNCTION__, aCounter, aPacketQty TRACE_END;

    ASSERT(NULL              != mBAR1 );
    ASSERT(RX_DESCRIPTOR_QTY >  mTx_In);

    Hardware::Unlock_AfterSend(aCounter, aPacketQty);

    if (0 < aPacketQty)
    {
        Pro1000_Tx_DescriptorTail lReg;

        lReg.mValue         =      0;
        lReg.mFields.mValue = mTx_In;

        // TRACE_DEBUG "%s - TDT (Write) = 0x%08x" DEBUG_EOL, __FUNCTION__, lReg.mValue TRACE_END;
        mBAR1->mTx_DescriptorTail0.mValue = lReg.mValue; // Writing hardware !

        // TRACE_DEBUG "%s - TDT (Read)  = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mTx_DescriptorTail0.mFields.mValue TRACE_END;

        mStatistics[OpenNetK::HARDWARE_STATS_PACKET_SEND] += aPacketQty;
    }
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

        Packet_Receive_NoLock( mPacketBuffer_Logical[ mPacketBuffer_In ], & mPacketData, & mPacketInfo, lCounter );

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
void Pro1000::Packet_Receive_NoLock(uint64_t aData, OpenNetK::Packet * aPacketData, OpenNet_PacketInfo * aPacketInfo, volatile long * aCounter)
{
    // TRACE_DEBUG "%s( 0x%llx, 0x%px, 0x%px, 0x%px )" DEBUG_EOL, __FUNCTION__, aData, aPacketData, aPacketInfo, aCounter TRACE_END;

    ASSERT(NULL != aPacketData);
    ASSERT(NULL != aPacketInfo);
    ASSERT(NULL != aCounter   );

    ASSERT(RX_DESCRIPTOR_QTY >  mRx_In     );
    ASSERT(NULL              != mRx_Virtual);

    mRx_Counter   [mRx_In] = aCounter   ;
    mRx_PacketData[mRx_In] = aPacketData;
    mRx_PacketInfo[mRx_In] = aPacketInfo;

    mRx_PacketData[mRx_In]->IndicateRxRunning();

    memset((Pro1000_Rx_Descriptor *)(mRx_Virtual) + mRx_In, 0, sizeof(Pro1000_Rx_Descriptor)); // volatile_cast

    mRx_Virtual[mRx_In].mLogicalAddress = aData;

    mRx_In = (mRx_In + 1) % RX_DESCRIPTOR_QTY;
}

// CRITICAL PATH - Packet
void Pro1000::Packet_Send_NoLock(uint64_t aLogicalAddress, const void *, unsigned int aSize_byte, volatile long * aCounter)
{
    // TRACE_DEBUG "%s( 0x%llx, , %u bytes, 0x%px )" DEBUG_EOL, __FUNCTION__, aLogicalAddress, aSize_byte, aCounter TRACE_END;

    ASSERT(0    != aLogicalAddress);
    ASSERT(0    <  aSize_byte     );

    ASSERT(TX_DESCRIPTOR_QTY >  mTx_In     );
    ASSERT(NULL              != mTx_Virtual);

    mTx_Counter[mTx_In] = aCounter;

    mTx_Virtual[mTx_In].mFields.mDescriptorDone = false     ;
    mTx_Virtual[mTx_In].mFields.mSize_byte      = aSize_byte;
    mTx_Virtual[mTx_In].mLogicalAddress         = aLogicalAddress;

    mTx_In = (mTx_In + 1) % TX_DESCRIPTOR_QTY;
}

// CRITICAL PATH
bool Pro1000::Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aRepeatCount)
{
    // TRACE_DEBUG "%s( 0x%px, %u bytes, %u )" DEBUG_EOL, __FUNCTION__, aPacket, aSize_byte, aRepeatCount TRACE_END;

    ASSERT( NULL != aPacket      );
    ASSERT(    0 <  aSize_byte   );
    ASSERT(    0 <  aRepeatCount );

    ASSERT(NULL != mBAR1      );
    ASSERT(NULL != mTx_Virtual);
    ASSERT(NULL != mZone0     );

    bool lResult;

    Lock();

    ASSERT(PACKET_BUFFER_QTY >  mPacketBuffer_In                       );
    ASSERT(NULL              != mPacketBuffer_Virtual[mPacketBuffer_In]);

    lResult = ((0 >= mPacketBuffer_Counter[mPacketBuffer_In]) && (Tx_GetAvailableDescriptor_Zone0() >= aRepeatCount));
    if (lResult)
    {
        memcpy(mPacketBuffer_Virtual[mPacketBuffer_In], aPacket, aSize_byte);

        volatile long * lCounter = mPacketBuffer_Counter + mPacketBuffer_In;
        uint64_t lPacket_PA = mPacketBuffer_Logical[mPacketBuffer_In];

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

// Private
/////////////////////////////////////////////////////////////////////////////

void Pro1000::Interrupt_Disable_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1);

    mBAR1->mInterruptMaskClear.mValue = 0xffffffff;
}

void Pro1000::Reset_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1);

    Interrupt_Disable_Zone0();

    mBAR1->mDeviceControl.mFields.mReset = true;

    while (mBAR1->mDeviceControl.mFields.mReset)
    {
        // TRACE_DEBUG "%s - Waiting ..." DEBUG_EOL, __FUNCTION__ TRACE_END;
    }

    Interrupt_Disable_Zone0();

    uint32_t lValue = mBAR1->mInterruptCauseRead.mValue;
    (void)(lValue);

    // TRACE_DEBUG "%s - ICR = 0x%08x" DEBUG_EOL, __FUNCTION__, lValue TRACE_END;
}

// Level   Thread
// Thread  Init
void Pro1000::Rx_Config_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1                   );
    ASSERT(   0 <  mConfig.mPacketSize_byte);

    for (unsigned int i = 0; i < (sizeof(mBAR1->mMulticastTableArray) / sizeof(mBAR1->mMulticastTableArray[0])); i++)
    {
        mBAR1->mMulticastTableArray[i] = 0;
    }

    // TODO  OpenNet.Adapter
    //       Low (Feature) - Add configuration field for:
    //       BroadcastAcceptMode, MulticastPromiscuousEnabled,
    //       PassMacControlFrames, StoreBadPackets, UnicastPromiscuousEnabled
    mBAR1->mRx_Control.mFields.mBroadcastAcceptMode         = true ;
    mBAR1->mRx_Control.mFields.mDiscardPauseFrames          = true ;
    mBAR1->mRx_Control.mFields.mLongPacketEnabled           = true ;
    mBAR1->mRx_Control.mFields.mMulticastPromiscuousEnabled = true ;
    mBAR1->mRx_Control.mFields.mPassMacControlFrames        = false;
    mBAR1->mRx_Control.mFields.mStoreBadPackets             = false;
    mBAR1->mRx_Control.mFields.mStripEthernetCRC            = true ;
    mBAR1->mRx_Control.mFields.mUnicastPromiscuousEnabled   = true ;

    mBAR1->mRx_DmaMaxOutstandingData.mFields.mValue_256_bytes = 0xfff;

    mBAR1->mRx_LongPacketMaximumLength.mFields.mValue_byte = mConfig.mPacketSize_byte;

    mBAR1->mRx_DescriptorBaseAddressHigh0 = (mRx_Logical >> 32) & 0xffffffff;
    mBAR1->mRx_DescriptorBaseAddressLow0  =  mRx_Logical        & 0xffffffff;

    mBAR1->mRx_DescriptorRingLength0.mFields.mValue_byte = sizeof(Pro1000_Rx_Descriptor) * RX_DESCRIPTOR_QTY;

    mBAR1->mRx_PacketBufferSize.mFields.mValue_KB = 84;

    mBAR1->mRx_SplitAndReplicationControl.mFields.mDescriptorType     =    0;
    mBAR1->mRx_SplitAndReplicationControl.mFields.mDropEnabled        = true;
    mBAR1->mRx_SplitAndReplicationControl.mFields.mHeaderSize_64bytes =    0;
    mBAR1->mRx_SplitAndReplicationControl.mFields.mPacketSize_KB      = mConfig.mPacketSize_byte / 1024;

    mBAR1->mRx_DescriptorControl0.mFields.mHostThreshold      =   16;
    mBAR1->mRx_DescriptorControl0.mFields.mPrefetchThreshold  =   16;
    mBAR1->mRx_DescriptorControl0.mFields.mWriteBackThreshold =   16;
    mBAR1->mRx_DescriptorControl0.mFields.mQueueEnable        = true;

    mBAR1->mRx_Control.mFields.mEnable = true;

    // TRACE_DEBUG "%s - RCTL              = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mRx_Control.mValue               TRACE_END;
    // TRACE_DEBUG "%s - RDBAH             = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mRx_DescriptorBaseAddressHigh0   TRACE_END;
    // TRACE_DEBUG "%s - RDBAL             = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mRx_DescriptorBaseAddressLow0    TRACE_END;
    // TRACE_DEBUG "%s - RDLEN             = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mRx_DescriptorRingLength0.mValue TRACE_END;
    // TRACE_DEBUG "%s - RXDCTL            = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mRx_DescriptorControl0.mValue    TRACE_END;
}

// CRITICAL PATH
//
// Level   SoftInt
// Thread  SoftInt
void Pro1000::Rx_Process_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(RX_DESCRIPTOR_QTY >  mRx_In     );
    ASSERT(RX_DESCRIPTOR_QTY >  mRx_Out    );
    ASSERT(NULL              != mRx_Virtual);

    while (mRx_In != mRx_Out)
    {
        if (!mRx_Virtual[mRx_Out].mFields.mDescriptorDone)
        {
            break;
        }

        // TRACE_DEBUG "%s - Rx Packet !" DEBUG_EOL, __FUNCTION__ TRACE_END;

        ASSERT(NULL != mRx_Counter   [mRx_Out]);
        ASSERT(NULL != mRx_PacketData[mRx_Out]);
        ASSERT(NULL != mRx_PacketInfo[mRx_Out]);

        mRx_PacketData[mRx_Out]->IndicateRxCompleted();
        mRx_PacketInfo[mRx_Out]->mSize_byte = mRx_Virtual[mRx_Out].mSize_byte; // Writing DirectGMA buffer !
        mRx_PacketInfo[mRx_Out]->mSendTo    =                               0; // Writing DirectGMA buffer !

        #ifdef _KMS_LINUX_
            ( * mRx_Counter[ mRx_Out ] ) --;
        #endif

        #ifdef _KMS_WINDOWS_
            InterlockedDecrement(mRx_Counter[mRx_Out]);
        #endif

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

    ASSERT(NULL != mBAR1);

    mStatistics[OpenNetK::HARDWARE_STATS_RX_BMC_MANAGEMENT_DROPPED_packet      ] += mBAR1->mRx_BmcManagementDropper_packet     ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_CIRCUIT_BREAKER_DROPPED_packet     ] += mBAR1->mRx_CircuitBreakerDropped_packet    ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_HOST_byte                          ] += mBAR1->mRx_HostGoodLow_byte                ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_HOST_packet                        ] += mBAR1->mRx_ToHost_packet                   ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_LENGTH_ERRORS_packet               ] += mBAR1->mRx_LengthErrors_packet             ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_MANAGEMENT_DROPPED_packet          ] += mBAR1->mRx_ManagementDropped_packet        ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_MISSED_packet                      ] += mBAR1->mRx_Missed_packet                   ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_NO_BUFFER_packet                   ] += mBAR1->mRx_NoBuffer_packet                 ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_OVERSIZE_packet                    ] += mBAR1->mRx_Oversize_packet                 ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_QUEUE_DROPPED_packet               ] += mBAR1->mRx_QueueDropPacket0.mFields.mValue ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_UNDERSIZE_packet                   ] += mBAR1->mRx_Undersize_packet                ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_XOFF_packet                        ] += mBAR1->mRx_XOff_packet                     ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_XON_packet                         ] += mBAR1->mRx_XOn_packet                      ;

    mStatistics[OpenNetK::HARDWARE_STATS_TX_DEFER_EVENTS                       ] += mBAR1->mTx_DeferEvents                     ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_DISCARDED_packet                   ] += mBAR1->mTx_Discarded_packet                ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_byte                          ] += mBAR1->mTx_HostGoodLow_byte                ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_packet                        ] += mBAR1->mTx_HostGood_packet                 ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_HOST_CIRCUIT_BREAKER_DROPPED_packet] += mBAR1->mTx_HostCircuitBreakerDropped_packet;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_NO_CRS_packet                      ] += mBAR1->mTx_NoCrs_packet                    ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_XOFF_packet                        ] += mBAR1->mTx_XOff_packet                     ;
    mStatistics[OpenNetK::HARDWARE_STATS_TX_XON_packet                         ] += mBAR1->mTx_XOn_packet                      ;
}

// Level   Thread
// Thread  Init
void Pro1000::Tx_Config_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1);

    mBAR1->mTx_DescriptorBaseAddressHigh0 = (mTx_Logical >> 32) & 0xffffffff;
    mBAR1->mTx_DescriptorBaseAddressLow0  =  mTx_Logical        & 0xffffffff;

    mBAR1->mTx_DescriptorRingLength0.mFields.mValue_bytes = sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY;

    mBAR1->mTx_PacketBufferSize.mFields.mValue_KB = 20;

    Pro1000_Tx_DescriptorControl lTXDCTL;

    lTXDCTL.mValue = mBAR1->mTx_DescriptorControl0.mValue;
    // TRACE_DEBUG "%s - TXDCTL (Initial)  = 0x%08x" DEBUG_EOL, __FUNCTION__, lTXDCTL.mValue TRACE_END;

    lTXDCTL.mFields.mHostThreshold      =   16;
    lTXDCTL.mFields.mPrefetchThreshold  =   16;
    lTXDCTL.mFields.mWriteBackThreshold =   16;
    lTXDCTL.mFields.mQueueEnable        = true;

    mBAR1->mTx_DescriptorControl0.mValue = lTXDCTL.mValue;
    // TRACE_DEBUG "%s - TXDCTL (Modified) = 0x%08x" DEBUG_EOL, __FUNCTION__, lTXDCTL.mValue TRACE_END;

    mBAR1->mTx_Control.mFields.mEnable = true;

    // TRACE_DEBUG "%s - TCTL              = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mTx_Control.mValue               TRACE_END;
    // TRACE_DEBUG "%s - TDBAH             = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mTx_DescriptorBaseAddressHigh0   TRACE_END;
    // TRACE_DEBUG "%s - TDBAL             = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mTx_DescriptorBaseAddressLow0    TRACE_END;
    // TRACE_DEBUG "%s - TDLEN             = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mTx_DescriptorRingLength0.mValue TRACE_END;
    // TRACE_DEBUG "%s - TXDCTL            = 0x%08x" DEBUG_EOL, __FUNCTION__, mBAR1->mTx_DescriptorControl0.mValue    TRACE_END;
}

// CRITICAL PATH
//
// Level   SoftInt
// Thread  SoftInt
void Pro1000::Tx_Process_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(TX_DESCRIPTOR_QTY >  mTx_In     );
    ASSERT(TX_DESCRIPTOR_QTY >  mTx_Out    );
    ASSERT(NULL              != mTx_Virtual);

    while (mTx_In != mTx_Out)
    {
        if (!mTx_Virtual[mTx_Out].mFields.mDescriptorDone)
        {
            break;
        }

        if (NULL != mTx_Counter[mTx_Out])
        {
            #ifdef _KMS_LINUX_
                ( * mTx_Counter[ mTx_Out ] ) --;
            #endif

            #ifdef _KMS_WINDOWS_
                InterlockedDecrement(mTx_Counter[mTx_Out]);
            #endif
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
