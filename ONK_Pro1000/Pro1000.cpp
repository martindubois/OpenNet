
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Pro1000.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Constants.h>
#include <OpenNetK/Interface.h>

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
    DbgPrintEx(DEBUG_ID, DEBUG_CONSTRUCTOR, PREFIX __FUNCTION__ "()" DEBUG_EOL);

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
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNetK::Adapter ==================================================

void Pro1000::GetState(OpenNet_State * aState)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aState);

    Hardware::GetState(aState);

    aState->mFlags.mFullDuplex = mBAR1->mDeviceStatus.mFields.mFullDuplex;
    aState->mFlags.mLinkUp     = mBAR1->mDeviceStatus.mFields.mLinkUp    ;
    aState->mFlags.mTx_Off     = mBAR1->mDeviceStatus.mFields.mTx_Off    ;

    // TODO  ONK_Pro1000.Pro1000
    //       Comprendre pourquoi la vitesse n'est pas indique correctement.

    switch (mBAR1->mDeviceStatus.mFields.mSpeed)
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
}

void Pro1000::ResetMemory()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    Hardware::ResetMemory();

    mBAR1 = NULL;
}

void Pro1000::SetCommonBuffer(uint64_t aLogical, volatile void * aVirtual)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aVirtual);

    uint64_t           lLogical = aLogical;
    volatile uint8_t * lVirtual = reinterpret_cast<volatile uint8_t *>(aVirtual);

    SkipDangerousBoundary(&lLogical, &lVirtual, sizeof(Pro1000_Rx_Descriptor) * RX_DESCRIPTOR_QTY, &mRx_Logical, reinterpret_cast<volatile uint8_t **>(&mRx_Virtual));
    SkipDangerousBoundary(&lLogical, &lVirtual, sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY, &mTx_Logical, reinterpret_cast<volatile uint8_t **>(&mTx_Virtual));

    unsigned int i;

    for (i = 0; i < PACKET_BUFFER_QTY; i++)
    {
        SkipDangerousBoundary(&lLogical, &lVirtual, mConfig.mPacketSize_byte, mTx_PacketBuffer_Logical + i, reinterpret_cast<volatile uint8_t **>(mTx_PacketBuffer_Virtual + i));
    }

    for (i = 0; i < TX_DESCRIPTOR_QTY; i++)
    {
        mTx_Virtual[i].mFields.mEndOfPacket  = true;
        mTx_Virtual[i].mFields.mReportStatus = true;
    }
}

// NOT TESTED  ONK_Pro1000.Pro1000.ErrorHandling
//             Memory 0 too small
bool Pro1000::SetMemory(unsigned int aIndex, volatile void * aVirtual, unsigned int aSize_byte)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( %u, , %u bytes )" DEBUG_EOL, aIndex, aSize_byte);

    ASSERT(NULL != aVirtual);

    switch (aIndex)
    {
    case 0:
        if (sizeof(Pro1000_BAR1) > aSize_byte)
        {
            return false;
        }

        mBAR1 = reinterpret_cast< volatile Pro1000_BAR1 * >( aVirtual );

        Interrupt_Disable();

        mInfo.mEthernetAddress.mAddress[0] = mBAR1->mRx_AddressLow0 .mFields.mA;
        mInfo.mEthernetAddress.mAddress[1] = mBAR1->mRx_AddressLow0 .mFields.mB;
        mInfo.mEthernetAddress.mAddress[2] = mBAR1->mRx_AddressLow0 .mFields.mC;
        mInfo.mEthernetAddress.mAddress[3] = mBAR1->mRx_AddressLow0 .mFields.mD;
        mInfo.mEthernetAddress.mAddress[4] = mBAR1->mRx_AddressHigh0.mFields.mE;
        mInfo.mEthernetAddress.mAddress[5] = mBAR1->mRx_AddressHigh0.mFields.mF;
        break;
    }

    return Hardware::SetMemory(aIndex, aVirtual, aSize_byte);
}

bool Pro1000::D0_Entry()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    mRx_In  = 0;
    mRx_Out = 0;

    mTx_In  = 0;
    mTx_Out = 0;

    memset(&mTx_Counter, 0, sizeof(mTx_Counter));

    mTx_PacketBuffer_In = 0;

    Reset();

    mBAR1->mDeviceControl.mFields.mInvertLossOfSignal = false;
    mBAR1->mDeviceControl.mFields.mSetLinkUp          = true ;

    mBAR1->mGeneralPurposeInterruptEnable.mFields.mExtendedInterruptAutoMaskEnable = true;

    Rx_Config();
    Tx_Config();

    return Hardware::D0_Entry();
}

bool Pro1000::D0_Exit()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    Interrupt_Disable();

    return Hardware::D0_Exit();
}

void Pro1000::Interrupt_Disable()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    ASSERT(NULL != mBAR1);

    Hardware::Interrupt_Disable();

    mBAR1->mInterruptMaskClear.mValue = 0xffffffff;
}

void Pro1000::Interrupt_Enable()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    Hardware::Interrupt_Enable();

    mBAR1->mInterruptMaskSet.mFields.mTx_DescriptorWritten = true;
    mBAR1->mInterruptMaskSet.mFields.mRx_DescriptorWritten = true;
}

bool Pro1000::Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing)
{
    ASSERT(NULL != aNeedMoreProcessing);

    (void)(aMessageId);

    uint32_t lValue = mBAR1->mInterruptCauseRead.mValue;
    (void)(lValue);

    (*aNeedMoreProcessing) = true;

    mStats.mInterrupt_Process++;

    return true;
}

void Pro1000::Interrupt_Process2()
{
    // DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    Rx_Process();
    Tx_Process();

    Hardware::Interrupt_Process2();
}

void Pro1000::Packet_Receive(uint64_t aData, OpenNet_PacketInfo * aPacketInfo, volatile long * aCounter)
{
    // DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( , ,  )" DEBUG_EOL);

    ASSERT(NULL != aPacketInfo);
    ASSERT(NULL != aCounter   );

    ASSERT(NULL              != mBAR1      );
    ASSERT(RX_DESCRIPTOR_QTY >  mRx_In     );
    ASSERT(NULL              != mRx_Virtual);

    mRx_Counter   [mRx_In] = aCounter   ;
    mRx_PacketInfo[mRx_In] = aPacketInfo;

    mRx_PacketInfo[mRx_In]->mPacketState = OPEN_NET_PACKET_STATE_RECEIVING;

    memset((Pro1000_Rx_Descriptor *)(mRx_Virtual) + mRx_In, 0, sizeof(Pro1000_Rx_Descriptor)); // volatile_cast

    mRx_Virtual[mRx_In].mLogicalAddress = aData;

    InterlockedIncrement(mRx_Counter[mRx_In]);

    mRx_In = (mRx_In + 1) % RX_DESCRIPTOR_QTY;

    mBAR1->mRx_DescriptorTail0.mFields.mValue = mRx_In;

    mStats.mPacket_Receive++;
}

void Pro1000::Packet_Send(uint64_t aData, unsigned int aSize_byte, volatile long * aCounter)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( , %u,  )" DEBUG_EOL, aSize_byte);

    ASSERT(OPEN_NET_PACKET_SIZE_MAX_byte >= aSize_byte);

    mTx_Counter[mTx_In] = aCounter;

    mTx_Virtual[mTx_In].mFields.mDescriptorDone = false     ;
    mTx_Virtual[mTx_In].mFields.mSize_byte      = aSize_byte;
    mTx_Virtual[mTx_In].mLogicalAddress         = aData     ;

    if (NULL != mTx_Counter[mTx_In])
    {
        InterlockedIncrement(mTx_Counter[mTx_In]);
    }

    mTx_In = (mTx_In + 1) % TX_DESCRIPTOR_QTY;

    mBAR1->mTx_DescriptorTail0.mFields.mValue = mTx_In;

    mStats.mPacket_Send++;
}

void Pro1000::Packet_Send(const void * aPacket, unsigned int aSize_byte)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( , %u bytes )" DEBUG_EOL, aSize_byte);

    UNREFERENCED_PARAMETER(aPacket);

    memcpy((void *)(mTx_PacketBuffer_Virtual[mTx_PacketBuffer_In]), aPacket, aSize_byte); // volatile_cast

    Packet_Send(mTx_PacketBuffer_Logical[mTx_PacketBuffer_In], aSize_byte, NULL);

    mTx_PacketBuffer_In++;
}

// Private
/////////////////////////////////////////////////////////////////////////////

void Pro1000::Reset()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    ASSERT(NULL != mBAR1);

    Interrupt_Disable();

    mBAR1->mDeviceControl.mFields.mReset = true;

    while (mBAR1->mDeviceControl.mFields.mReset);

    Interrupt_Disable();

    uint32_t lValue = mBAR1->mInterruptCauseRead.mValue;

    (void)(lValue);
}

// Level   Thread
// Thread  Init
void Pro1000::Rx_Config()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    ASSERT(NULL != mBAR1);

    for (unsigned int i = 0; i < (sizeof(mBAR1->mMulticastTableArray) / sizeof(mBAR1->mMulticastTableArray[0])); i++)
    {
        mBAR1->mMulticastTableArray[i] = 0;
    }

    mBAR1->mRx_Control.mFields.mBroadcastAcceptMode         = true ; // TODO Config
    mBAR1->mRx_Control.mFields.mLongPacketEnabled           = true ;
    mBAR1->mRx_Control.mFields.mMulticastPromiscuousEnabled = true ; // TODO Config
    mBAR1->mRx_Control.mFields.mPassMacControlFrames        = true ; // TODO Config
    mBAR1->mRx_Control.mFields.mStoreBadPackets             = true ; // TODO Config
    mBAR1->mRx_Control.mFields.mUnicastPromiscuousEnabled   = true ; // TODO Config

    mBAR1->mRx_LongPacketMaximumLength.mFields.mValue_byte = mConfig.mPacketSize_byte;

    mBAR1->mRx_DescriptorBaseAddressHigh0 = (mRx_Logical >> 32) & 0xffffffff;
    mBAR1->mRx_DescriptorBaseAddressLow0  =  mRx_Logical        & 0xffffffff;

    mBAR1->mRx_DescriptorRingLength0.mFields.mValue_byte = sizeof(Pro1000_Rx_Descriptor) * RX_DESCRIPTOR_QTY;

    mBAR1->mRx_SplitAndReplicationControl.mFields.mHeaderSize_64bytes = 0;
    mBAR1->mRx_SplitAndReplicationControl.mFields.mPacketSize_KB      = mConfig.mPacketSize_byte / 1024;

    mBAR1->mRx_DescriptorControl0.mFields.mQueueEnable = true;

    mBAR1->mRx_Control.mFields.mEnable = true;
}

// Level   SoftInt
// Thread  SoftInt
void Pro1000::Rx_Process()
{
    ASSERT(RX_DESCRIPTOR_QTY >  mRx_In     );
    ASSERT(RX_DESCRIPTOR_QTY >  mRx_Out    );
    ASSERT(NULL              != mRx_Virtual);

    while (mRx_In != mRx_Out)
    {
        if (!mRx_Virtual[mRx_Out].mFields.mDescriptorDone)
        {
            break;
        }

        ASSERT(NULL                            != mRx_Counter   [mRx_Out]              );
        ASSERT(NULL                            != mRx_PacketInfo[mRx_Out]              );
        ASSERT(OPEN_NET_PACKET_STATE_RECEIVING == mRx_PacketInfo[mRx_Out]->mPacketState);
        ASSERT(OPEN_NET_PACKET_SIZE_MAX_byte   >= mRx_Virtual   [mRx_Out].mSize_byte   );

        mRx_PacketInfo[mRx_Out]->mPacketSize_byte = mRx_Virtual[mRx_Out].mSize_byte;
        mRx_PacketInfo[mRx_Out]->mPacketState     = OPEN_NET_PACKET_STATE_RECEIVED ;

        InterlockedDecrement(mRx_Counter[mRx_Out]);

        mRx_Out = (mRx_Out + 1) % RX_DESCRIPTOR_QTY;

        mStats.mRx_Packet++;
    }
}

// Level   Thread
// Thread  Init
void Pro1000::Tx_Config()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    ASSERT(NULL != mBAR1);

    mBAR1->mTx_DescriptorBaseAddressHigh0 = (mTx_Logical >> 32) & 0xffffffff;
    mBAR1->mTx_DescriptorBaseAddressLow0  =  mTx_Logical        & 0xffffffff;

    mBAR1->mTx_DescriptorRingLength0.mFields.mValue_bytes = sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY;

    mBAR1->mTx_DescriptorControl0.mFields.mQueueEnable = true;

    mBAR1->mTx_Control.mFields.mEnable = true;
}

// Level   SoftInt
// Thread  SoftInt
void Pro1000::Tx_Process()
{
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
            InterlockedDecrement(mTx_Counter[mTx_Out]);
        }

        mTx_Out = (mTx_Out + 1) % TX_DESCRIPTOR_QTY;

        mStats.mTx_Packet++;
    }
}
