
// Author     KMS - Martin Dubois, P.Eng.
// Copyright  (C) 2018-2020 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Intel.cpp

// REQUIREMENT  ONK_X.InterruptRateLimitation
//              The adapter driver limit the interruption rate.

// REQUIREMENT  ONK_X.Tx.OverflowDetection
//              The adapter driver fail transmit request when the tx queue or
//              or the descriptor ring is full.

#define __CLASS__     "Intel::"
#define __NAMESPACE__ ""

// TODO  ONK_Hardware
//       Normal (Feature) - Create a library for all network adapter code
//       with a factory.

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
#include "Intel.h"

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define PACKET_SIZE_byte  (9 * 1024)

// Public
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNetK::Adapter ==================================================

void Intel::GetState(OpenNetK::Adapter_State * aState)
{
    // TRACE_DEBUG "%s( 0x%p )" DEBUG_EOL, __FUNCTION__, aState TRACE_END;

    ASSERT(NULL != aState);

    ASSERT(NULL != mZone0);

    uint32_t lFlags = mZone0->LockFromThread();

        ASSERT(NULL != mBAR1_MA);

        Intel_DeviceStatus lDeviceStatus;

        lDeviceStatus.mValue = mBAR1_MA->mDeviceStatus.mValue;

        aState->mFlags.mFullDuplex = lDeviceStatus.mFields.mFullDuplex;
        aState->mFlags.mLinkUp     = lDeviceStatus.mFields.mLinkUp    ;
        aState->mFlags.mTx_Off     = lDeviceStatus.mFields.mTx_Off    ;

        // TODO  ONK_Hardware.Intel
        //       Normal (Issue) - Comprendre pourquoi la vitesse n'est pas
        //       indique correctement.

        aState->mSpeed_Mb_s = mInfo.mMaxSpeed_Mb_s;

    mZone0->UnlockFromThread( lFlags );
}

void Intel::ResetConfig()
{
    Hardware::ResetConfig();

    uint32_t lFlags = mZone0->LockFromThread();

        Config_Apply_Zone0();

    mZone0->UnlockFromThread(lFlags);
}

void Intel::ResetMemory()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    Hardware::ResetMemory();

    uint32_t lFlags = mZone0->LockFromThread();

        mBAR1_MA = NULL;

    mZone0->UnlockFromThread( lFlags );
}

void Intel::SetCommonBuffer(uint64_t aCommon_PA, void * aCommon_CA)
{
    // TRACE_DEBUG "%s( 0x%llx, 0x%p )" DEBUG_EOL, __FUNCTION__, aCommon_PA, aCommon_CA TRACE_END;

    ASSERT(NULL != aCommon_CA);

    ASSERT(NULL != mZone0);

    uint32_t lFlags = mZone0->LockFromThread();

        uint64_t  lCommon_PA = aCommon_PA;
        uint8_t * lCommon_CA = reinterpret_cast<uint8_t *>(aCommon_CA);

        mRx_CA = reinterpret_cast<Intel_Rx_Descriptor *>(lCommon_CA);
        mRx_PA = lCommon_PA;

        lCommon_CA += sizeof(Intel_Rx_Descriptor) * mInfo.mRx_Descriptors;
        lCommon_PA += sizeof(Intel_Rx_Descriptor) * mInfo.mRx_Descriptors;

        mTx_CA = reinterpret_cast<Intel_Tx_Descriptor *>(lCommon_CA);
        mTx_PA = lCommon_PA;

        lCommon_CA += sizeof(Intel_Tx_Descriptor) * mInfo.mTx_Descriptors;
        lCommon_PA += sizeof(Intel_Tx_Descriptor) * mInfo.mTx_Descriptors;

        unsigned int i;

        for (i = 0; i < PACKET_BUFFER_QTY; i++)
        {
            SkipDangerousBoundary(&lCommon_PA, &lCommon_CA, mConfig.mPacketSize_byte, mPacketBuffer_PA + i, reinterpret_cast<uint8_t **>(mPacketBuffer_CA + i));

            ASSERT(NULL != mPacketBuffer_CA[i]);
        }

        for (i = 0; i < mInfo.mTx_Descriptors; i++)
        {
            mTx_CA[i].mFields.mEndOfPacket  = true;
            mTx_CA[i].mFields.mInsertCRC    = true;
            mTx_CA[i].mFields.mReportStatus = true;
        }

    mZone0->UnlockFromThread( lFlags );
}

void Intel::SetConfig(const OpenNetK::Adapter_Config & aConfig)
{
    ASSERT(NULL != (&aConfig));

    Hardware::SetConfig(aConfig);

    uint32_t lFlags = mZone0->LockFromThread();

        Config_Apply_Zone0();

    mZone0->UnlockFromThread(lFlags);
}

// NOT TESTED  ONK_Hardware.Intel.ErrorHandling
//             Memory 0 too small
bool Intel::SetMemory(unsigned int aIndex, void * aMemory_MA, unsigned int aSize_byte)
{
    // TRACE_DEBUG "%s( %u, 0x%p, %u bytes )" DEBUG_EOL, __FUNCTION__, aIndex, aMemory_MA, aSize_byte TRACE_END;

    ASSERT(NULL != aMemory_MA);
    ASSERT(   0 <  aSize_byte);

    ASSERT(NULL != mZone0);

    switch (aIndex)
    {
    case 0:
        if (sizeof(Intel_BAR1) > aSize_byte)
        {
            return false;
        }

        uint32_t lFlags = mZone0->LockFromThread();

            ASSERT(NULL == mBAR1_MA);

            mBAR1_MA = reinterpret_cast< volatile Intel_BAR1 * >( aMemory_MA );

            Interrupt_Disable_Zone0();

        mZone0->UnlockFromThread( lFlags );
        break;
    }

    return Hardware::SetMemory(aIndex, aMemory_MA, aSize_byte);
}

void Intel::D0_Entry()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mZone0);

    uint32_t lFlags = mZone0->LockFromThread();

        ASSERT(NULL != mBAR1_MA);

        mRx_In  = 0;
        mRx_Out = 0;

        mTx_In  = 0;
        mTx_Out = 0;

        memset(&mTx_Counter, 0, sizeof(mTx_Counter));

        mPacketBuffer_In = 0;

        Reset_Zone0();

        Intel_DeviceControl lCTRL;

        lCTRL.mValue = mBAR1_MA->mDeviceControl.mValue;

        lCTRL.mFields.mInvertLossOfSignal   = false;
        // lCTRL.mFields.mRx_FlowControlEnable = true ;
        lCTRL.mFields.mSetLinkUp            = true ;
        // lCTRL.mFields.mTx_FlowControlEnable = true ;

        mBAR1_MA->mDeviceControl.mValue = lCTRL.mValue;

    mZone0->UnlockFromThread( lFlags );

    Hardware::D0_Entry();
}

bool Intel::D0_Exit()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    Interrupt_Disable();

    return Hardware::D0_Exit();
}

void Intel::Interrupt_Disable()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mZone0);

    Hardware::Interrupt_Disable();

    uint32_t lFlags = mZone0->LockFromThread();

        Interrupt_Disable_Zone0();

    mZone0->UnlockFromThread( lFlags );
}

bool Intel::Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing)
{
    // TRACE_DEBUG "%s( %u, 0x%p )" DEBUG_EOL, __FUNCTION__, aMessageId, aNeedMoreProcessing TRACE_END;

    ASSERT(NULL != aNeedMoreProcessing);

    (*aNeedMoreProcessing) = true;

    mStatistics[OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS] ++;

    mStatistics[OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS_LAST_MESSAGE_ID] = aMessageId;

    return true;
}

void Intel::Interrupt_Process2(bool * aNeedMoreProcessing)
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

bool Intel::Packet_Drop()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT( PACKET_BUFFER_QTY > mPacketBuffer_In );

    bool lResult;

    uint32_t lFlags = mZone0->LockFromThread();

    lResult = ( ( 0 >= mPacketBuffer_Counter[ mPacketBuffer_In ] ) && ( Rx_GetAvailableDescriptor_Zone0() > 0 ) );
    if ( lResult )
    {
        volatile long * lCounter = mPacketBuffer_Counter + mPacketBuffer_In;

        mPacketData.Init(mPacketBuffer_PA[mPacketBuffer_In], mPacketBuffer_CA[mPacketBuffer_In], &mPacketInfo);

        Packet_Receive_NoLock( & mPacketData, lCounter );

        mPacketBuffer_In = (mPacketBuffer_In + 1) % PACKET_BUFFER_QTY;

        Unlock_AfterReceive_FromThread( lCounter, 1, lFlags );
    }
    else
    {
        mZone0->UnlockFromThread( lFlags );
    }

    return lResult;
}

void Intel::Packet_Receive_NoLock(OpenNetK::Packet * aPacket, volatile long * aCounter)
{
    // TRACE_DEBUG "%s( 0x%p, 0x%p )" DEBUG_EOL, __FUNCTION__, aPacketData, aCounter TRACE_END;

    ASSERT(NULL != aPacket );
    ASSERT(NULL != aCounter);

    ASSERT(NULL                  != mRx_CA);
    ASSERT(mInfo.mRx_Descriptors >  mRx_In);

    mRx_Counter   [mRx_In] = aCounter;
    mRx_PacketData[mRx_In] = aPacket ;

    mRx_PacketData[mRx_In]->IndicateRxRunning();

    memset((Intel_Rx_Descriptor *)(mRx_CA) + mRx_In, 0, sizeof(Intel_Rx_Descriptor)); // volatile_cast

    mRx_CA[mRx_In].mLogicalAddress = aPacket->GetData_PA();

    mRx_In = (mRx_In + 1) % mInfo.mRx_Descriptors;
}

void Intel::Packet_Send_NoLock(uint64_t aData_PA, const void *, unsigned int aSize_byte, volatile long * aCounter)
{
    // TRACE_DEBUG "%s( 0x%llx, , %u bytes, 0x%p )" DEBUG_EOL, __FUNCTION__, aData_PA, aSize_byte, aCounter TRACE_END;

    ASSERT(0 != aData_PA  );
    ASSERT(0 <  aSize_byte);

    ASSERT(NULL                  != mTx_CA);
    ASSERT(mInfo.mTx_Descriptors >  mTx_In);

    mTx_Counter[mTx_In] = aCounter;

    mTx_CA[mTx_In].mFields.mDescriptorDone = false     ;
    mTx_CA[mTx_In].mFields.mSize_byte      = aSize_byte;
    mTx_CA[mTx_In].mLogicalAddress         = aData_PA  ;

    mTx_In = (mTx_In + 1) % mInfo.mTx_Descriptors;
}

bool Intel::Packet_Send(const void * aPacket, unsigned int aSize_byte, bool aPriority, unsigned int aRepeatCount)
{
    // TRACE_DEBUG "%s( 0x%p, %u bytes, %s, %u )" DEBUG_EOL, __FUNCTION__, aPacket, aSize_byte, aPriority ? "true" : "false", aRepeatCount TRACE_END;

    ASSERT( NULL != aPacket      );
    ASSERT(    0 <  aSize_byte   );
    ASSERT(    0 <  aRepeatCount );

    ASSERT(NULL != mTx_CA);

    bool lResult;

    uint32_t lFlags = mZone0->LockFromThread();

    ASSERT(NULL != mBAR1_MA);

    ASSERT(PACKET_BUFFER_QTY >  mPacketBuffer_In                  );
    ASSERT(NULL              != mPacketBuffer_CA[mPacketBuffer_In]);

    unsigned int lNext = (mPacketBuffer_In + 1) % PACKET_BUFFER_QTY;

    lResult = ((0 >= mPacketBuffer_Counter[mPacketBuffer_In]) && (aPriority || (0 >= mPacketBuffer_Counter[lNext])) && (Tx_GetAvailableDescriptor_Zone0() >= aRepeatCount));
    if (lResult)
    {
        memcpy(mPacketBuffer_CA[mPacketBuffer_In], aPacket, aSize_byte);

        volatile long * lCounter   = mPacketBuffer_Counter + mPacketBuffer_In;
        uint64_t        lPacket_PA = mPacketBuffer_PA[mPacketBuffer_In];

        for (unsigned int i = 0; i < aRepeatCount; i++)
        {
            Packet_Send_NoLock(lPacket_PA, NULL, aSize_byte, lCounter);
        }

        mPacketBuffer_In = lNext;

        Unlock_AfterSend_FromThread(lCounter, aRepeatCount, lFlags);
    }
    else
    {
        mZone0->UnlockFromThread( lFlags );
    }

    return lResult;
}

unsigned int Intel::Statistics_Get(uint32_t * aOut, unsigned int aOutSize_byte, bool aReset)
{
    // TRACE_DEBUG "%s( 0x%p, %u bytes, %s )" DEBUG_EOL, __FUNCTION__, aOut, aOutSize_byte, aReset ? "true" : "false" TRACE_END;

    Statistics_Update();

    return Hardware::Statistics_Get(aOut, aOutSize_byte, aReset);
}

void Intel::Statistics_Reset()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    Statistics_Update();

    Hardware::Statistics_Reset();
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// aOut   [---;-W-]
// aIn_MA [---;R--]
void Intel::RxAddress_Read(uint8_t * aOut, volatile Intel_Rx_Address * aIn_MA)
{
    ASSERT(NULL != aOut  );
    ASSERT(NULL != aIn_MA);

    uint32_t lTmp = aIn_MA->mRAL;

    aOut[0] = static_cast<uint8_t>( lTmp & 0x000000ff       );
    aOut[1] = static_cast<uint8_t>((lTmp & 0x0000ff00) >>  8);
    aOut[2] = static_cast<uint8_t>((lTmp & 0x00ff0000) >> 16);
    aOut[3] = static_cast<uint8_t>((lTmp & 0xff000000) >> 24);

    lTmp = aIn_MA->mFields.mRAH;

    aOut[4] = static_cast<uint8_t>( lTmp & 0x000000ff      );
    aOut[5] = static_cast<uint8_t>((lTmp & 0x0000ff00) >> 8);
}

// aOut_MA [---;RW-]
// aIn     [---;R--]
void Intel::RxAddress_Write(volatile Intel_Rx_Address * aOut_MA, const uint8_t * aIn)
{
    ASSERT(NULL != aOut_MA);
    ASSERT(NULL != aIn    );

    bool lAddressValid = false;

    uint32_t lTmp32;

    lTmp32  = aIn[0]      ;
    lTmp32 |= aIn[1] <<  8;
    lTmp32 |= aIn[2] << 16;
    lTmp32 |= aIn[3] << 24;

    if (0 != lTmp32) { lAddressValid = true; }

    aOut_MA->mRAL = lTmp32;

    uint16_t lTmp16;

    lTmp16  = aIn[4]     ;
    lTmp16 |= aIn[5] << 8;

    if (0 != lTmp16) { lAddressValid = true; }

    aOut_MA->mFields.mRAH           = lTmp16       ;
    aOut_MA->mFields.mAddressSelect =             0;
    aOut_MA->mFields.mAddressValid  = lAddressValid;
    aOut_MA->mFields.mPoolSelect    =             0;
}

Intel::Intel(OpenNetK::Adapter_Type aType) : Hardware(aType, PACKET_SIZE_byte)
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT( 0x09400 == sizeof( Intel_BAR1          ) );
    ASSERT(       4 == sizeof( Intel_DeviceControl ) );
    ASSERT(      16 == sizeof( Intel_Rx_Descriptor ) );
    ASSERT(      16 == sizeof( Intel_Tx_Descriptor ) );

    mInfo.mCommonBufferSize_byte += (PACKET_SIZE_byte * PACKET_BUFFER_QTY); // Packet buffers
    mInfo.mCommonBufferSize_byte += (mInfo.mCommonBufferSize_byte / OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte) * PACKET_SIZE_byte; // Skip 64 KB boundaries

    strcpy(mInfo.mComment                  , "ONK_Hardware");
    strcpy(mInfo.mVersion_Driver  .mComment, "ONK_Hardware");

    memset((void *)(&mPacketBuffer_Counter), 0, sizeof(mPacketBuffer_Counter)); // volatile_cast
}

void Intel::MulticastArray_Clear_Zone0()
{
    ASSERT(NULL != mBAR1_MA);

    for (unsigned int i = 0; i < (sizeof(mBAR1_MA->mMulticastTableArray) / sizeof(mBAR1_MA->mMulticastTableArray[0])); i++)
    {
        mBAR1_MA->mMulticastTableArray[i] = 0;
    }
}

void Intel::Reset_Zone0()
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
}

void Intel::Statistics_Update()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL != mBAR1_MA);

    mStatistics[OpenNetK::HARDWARE_STATS_RX_MANAGEMENT_DROPPED_packet] += mBAR1_MA->mRx_ManagementDropped_packet;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_OVERSIZE_packet          ] += mBAR1_MA->mRx_Oversize_packet         ;
    mStatistics[OpenNetK::HARDWARE_STATS_RX_UNDERSIZE_packet         ] += mBAR1_MA->mRx_Undersize_packet        ;
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Level   SoftInt

// CRITICAL PATH  Interrupt.Rx
//                1 / hardware interrupt + 1 / tick
void Intel::Rx_Process_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL                  != mRx_CA );
    ASSERT(mInfo.mRx_Descriptors >  mRx_In );
    ASSERT(mInfo.mRx_Descriptors >  mRx_Out);

    while (mRx_In != mRx_Out)
    {
        if (!mRx_CA[mRx_Out].mFields.mDescriptorDone)
        {
            break;
        }

        ASSERT(NULL != mRx_Counter   [mRx_Out]);
        ASSERT(NULL != mRx_PacketData[mRx_Out]);

        mRx_PacketData   [mRx_Out]->IndicateRxCompleted(mRx_CA[mRx_Out].mSize_byte);

        ( * mRx_Counter[ mRx_Out ] ) --;

        mRx_Out = (mRx_Out + 1) % mInfo.mRx_Descriptors;

        mStatistics[OpenNetK::HARDWARE_STATS_RX_packet] ++;
    }
}

unsigned int Intel::Rx_GetAvailableDescriptor_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(mInfo.mRx_Descriptors > mTx_In );
    ASSERT(mInfo.mRx_Descriptors > mTx_Out);

    return ((mRx_Out + mInfo.mRx_Descriptors - mRx_In - 1) % mInfo.mRx_Descriptors);
}

// Level  SoftInt

// CRITICAL PATH  Interrupt.Tx
//                1 / hardware interrupt + 1 / tick
void Intel::Tx_Process_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(NULL                  != mTx_CA );
    ASSERT(mInfo.mTx_Descriptors >  mTx_In );
    ASSERT(mInfo.mTx_Descriptors >  mTx_Out);

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

        mTx_Out = (mTx_Out + 1) % mInfo.mTx_Descriptors;

        mStatistics[OpenNetK::HARDWARE_STATS_TX_packet] ++;
    }
}

unsigned int Intel::Tx_GetAvailableDescriptor_Zone0()
{
    // TRACE_DEBUG "%s()" DEBUG_EOL, __FUNCTION__ TRACE_END;

    ASSERT(mInfo.mTx_Descriptors > mTx_In );
    ASSERT(mInfo.mTx_Descriptors > mTx_Out);

    return ((mTx_Out + mInfo.mTx_Descriptors - mTx_In - 1) % mInfo.mTx_Descriptors);
}
