
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Pro1000.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Interface.h>

// ===== Common =============================================================
#include "../Common/Version.h"

// ===== ONL_Pro1000 ========================================================
#include "Pro1000.h"

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define PACKET_SIZE_byte  (9728)

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
    mInfo.mCommonBufferSize_byte += (mInfo.mCommonBufferSize_byte / (64 * 1024)) * PACKET_SIZE_byte; // Skip 64 KB boundaries

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

    // TODO  Dev
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

    Skip64KByteBoundary(&lLogical, &lVirtual, sizeof(Pro1000_Rx_Descriptor) * RX_DESCRIPTOR_QTY, &mRx_Logical, reinterpret_cast<volatile uint8_t **>(&mRx_Virtual));
    Skip64KByteBoundary(&lLogical, &lVirtual, sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY, &mTx_Logical, reinterpret_cast<volatile uint8_t **>(&mTx_Virtual));

    unsigned int i;

    for (i = 0; i < PACKET_BUFFER_QTY; i++)
    {
        Skip64KByteBoundary(&lLogical, &lVirtual, mConfig.mPacketSize_byte, mTx_PacketBuffer_Logical + i, reinterpret_cast<volatile uint8_t **>(mTx_PacketBuffer_Virtual + i));
    }

    for (i = 0; i < TX_DESCRIPTOR_QTY; i++)
    {
        mTx_Virtual[i].mFields.mEndOfPacket  = true;
        mTx_Virtual[i].mFields.mReportStatus = true;
    }
}

// NOT TESTED  ONK_Pro1000.Pro1000
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

    mBAR1->mInterruptAcknowledgeAutoMask.mFields.mTx_DescriptorWritten = true;

    for (unsigned int i = 0; i < (sizeof(mBAR1->mMulticastTableArray) / sizeof(mBAR1->mMulticastTableArray[0])); i++)
    {
        mBAR1->mMulticastTableArray[i] = 0;
    }

    mBAR1->mRx_DescriptorBaseAddressHigh0 = (mRx_Logical >> 32) & 0xffffffff;
    mBAR1->mRx_DescriptorBaseAddressLow0  =  mRx_Logical        & 0xffffffff;

    mBAR1->mRx_DescriptorRingLength0.mFields.mValue_byte = sizeof(Pro1000_Rx_Descriptor) * TX_DESCRIPTOR_QTY;

    mBAR1->mRx_SplitAndReplicationControl.mFields.mHeaderSize_64bytes = 0;
    mBAR1->mRx_SplitAndReplicationControl.mFields.mPacketSize_KB      = mConfig.mPacketSize_byte / 1024;

    mBAR1->mRx_DescriptorControl0.mFields.mQueueEnable = true;

    // while (!mBAR1->mRx_DescriptorControl0.mFields.mQueueEnable);

    mBAR1->mRx_Control.mFields.mLongPacketEnabled = true;
    mBAR1->mRx_Control.mFields.mEnable            = true;

    mBAR1->mTx_DescriptorBaseAddressHigh0 = (mTx_Logical >> 32) & 0xffffffff;
    mBAR1->mTx_DescriptorBaseAddressLow0  =  mTx_Logical        & 0xffffffff;

    mBAR1->mTx_DescriptorRingLength0.mFields.mValue_bytes = sizeof(Pro1000_Tx_Descriptor) * TX_DESCRIPTOR_QTY;

    mBAR1->mTx_DescriptorControl0.mFields.mWriteBackThreshold = 1;
    mBAR1->mTx_DescriptorControl0.mFields.mQueueEnable        = true;

    // while (!mBAR1->mTx_DescriptorControl0.mFields.mQueueEnable);

    mBAR1->mTx_Control.mFields.mEnable = true;

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
}

bool Pro1000::Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing)
{
    ASSERT(NULL != aNeedMoreProcessing);

    (void)(aMessageId);

    uint32_t lValue = mBAR1->mInterruptCauseRead.mValue;
    (void)(lValue);

    (*aNeedMoreProcessing) = true;

    return true;
}

void Pro1000::Interrupt_Process2()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    Hardware::Interrupt_Process2();

    Tx_Process();

    // TODO  Dev
}

// TODO  Test
bool Pro1000::Packet_Receive(OpenNet_BufferInfo * aBuffer, unsigned int aIndex)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aBuffer);

    (void)(aIndex);
    // TODO Dev

    return true;
}

bool Pro1000::Packet_Send(OpenNet_BufferInfo * aBuffer, unsigned int aIndex)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( , %u )" DEBUG_EOL, aIndex);

    ASSERT(NULL != aBuffer);

    // TODO Dev

    return true;
}

bool Pro1000::Packet_Send(const void * aPacket, unsigned int aSize_byte)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( , %u bytes )" DEBUG_EOL, aSize_byte);

    UNREFERENCED_PARAMETER(aPacket);

    memcpy((void *)(mTx_PacketBuffer_Virtual[mTx_PacketBuffer_In]), aPacket, aSize_byte); // volatile_cast

    Packet_Send(mTx_PacketBuffer_Logical[mTx_PacketBuffer_In], aSize_byte, NULL);

    mTx_PacketBuffer_In++;

    return true;
}

// Private
/////////////////////////////////////////////////////////////////////////////

void Pro1000::FlushWrite()
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    ASSERT(NULL != mBAR1);

    uint32_t lValue = mBAR1->mDeviceStatus.mValue;

    (void)(lValue);
}

void Pro1000::Packet_Send(uint64_t aLogical, unsigned int aSize_byte, volatile long * aCounter)
{
    DbgPrintEx(DEBUG_ID, DEBUG_METHOD, PREFIX __FUNCTION__ "( , %u bytes,  )" DEBUG_EOL, aSize_byte);

    mTx_Counter[mTx_In] = aCounter;

    mTx_Virtual[mTx_In].mFields.mDescriptorDone = false     ;
    mTx_Virtual[mTx_In].mFields.mEndOfPacket    = true      ;
    mTx_Virtual[mTx_In].mFields.mReportStatus   = true      ;
    mTx_Virtual[mTx_In].mFields.mSize_byte      = aSize_byte;
    mTx_Virtual[mTx_In].mLogicalAddress         = aLogical  ;

    if (NULL != mTx_Counter[mTx_In])
    {
        InterlockedIncrement(mTx_Counter[mTx_In]);
    }

    mTx_In = (mTx_In + 1) % TX_DESCRIPTOR_QTY;

    mBAR1->mTx_DescriptorTail0.mFields.mValue = mTx_In;
}

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

// Level   DISPATCH
// Thread  DpcForIsr
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

        // TODO  Dev
        //       Comprendre pourquoi nous ne nous rendons pas ici.

        if (NULL != mTx_Counter[mTx_Out])
        {
            InterlockedDecrement(mTx_Counter[mTx_Out]);
        }

        mTx_Out = (mTx_Out + 1) % TX_DESCRIPTOR_QTY;
    }
}
