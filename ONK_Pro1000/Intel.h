
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Intel.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/Hardware.h>
#include <OpenNetK/Packet.h>
#include <OpenNetK/Types.h>

// ===== ONK_Pro1000 ========================================================
#include "Intel_Regs.h"
// Class
/////////////////////////////////////////////////////////////////////////////

class Intel : public OpenNetK::Hardware
{

public:

    // ===== OpenNetK::Hardware =============================================
    virtual void         GetState          (OpenNetK::Adapter_State * aStats);
    virtual void         ResetMemory       ();
    virtual void         SetCommonBuffer   (uint64_t aCommon_PA, void * aCommon_CA);
    virtual bool         SetMemory         (unsigned int aIndex, void * aMemory_MA, unsigned int aSize_byte);
    virtual void         D0_Entry          ();
    virtual bool         D0_Exit           ();
    virtual void         Interrupt_Disable ();
    virtual bool         Interrupt_Process (unsigned int aMessageId, bool * aNeedMoreProcessing);
    virtual void         Interrupt_Process2(bool * aNeedMoreProcessing);
    virtual bool         Packet_Drop          ();
    virtual void         Packet_Receive_NoLock(uint64_t aPacket_PA, OpenNetK::Packet * aPacketData, OpenNet_PacketInfo * aPacketInfo_XA, volatile long * aCounter);
    virtual void         Packet_Send_NoLock   (uint64_t aPacket_PA, const void * aPacket_XA, unsigned int aSize_byte, volatile long * aCounter = NULL);
    virtual bool         Packet_Send       (const void * aPacket, unsigned int aSize_byte, unsigned int aRepeatCount = 1);
    virtual unsigned int Statistics_Get    (uint32_t * aOut, unsigned int aOutSize_byte, bool aReset);
    virtual void         Statistics_Reset  ();

protected:

    enum
    {
        RX_DESCRIPTOR_QTY = 32 * 1024,
        TX_DESCRIPTOR_QTY = 32 * 1024,
    };

    Intel();

    void MulticastArray_Clear_Zone0();

    virtual void Interrupt_Disable_Zone0() = 0;

    virtual void Reset_Zone0();

    virtual void Statistics_Update();

    // ===== Zone 0 =========================================================

    unsigned int mRx_In;
    uint64_t     mRx_PA;

    unsigned int mTx_In;
    uint64_t     mTx_PA;

private:

    enum
    {
        PACKET_BUFFER_QTY = 64,
    };

    void Rx_Process_Zone0();

    unsigned int Rx_GetAvailableDescriptor_Zone0();

    void Tx_Process_Zone0();

    unsigned int Tx_GetAvailableDescriptor_Zone0();

    OpenNetK::Packet   mPacketData;
    OpenNet_PacketInfo mPacketInfo;

    // ===== Zone 0 =========================================================

    volatile Intel_BAR1 * mBAR1_MA;

    Intel_Rx_Descriptor   * mRx_CA ;
    volatile long         * mRx_Counter      [RX_DESCRIPTOR_QTY];
    unsigned int            mRx_Out;
    OpenNetK::Packet      * mRx_PacketData   [RX_DESCRIPTOR_QTY];
    OpenNet_PacketInfo    * mRx_PacketInfo_MA[RX_DESCRIPTOR_QTY];

    Intel_Tx_Descriptor   * mTx_CA ;
    volatile long         * mTx_Counter[TX_DESCRIPTOR_QTY];
    unsigned int            mTx_Out;

    void        * mPacketBuffer_CA     [PACKET_BUFFER_QTY];
    volatile long mPacketBuffer_Counter[PACKET_BUFFER_QTY];
    unsigned int  mPacketBuffer_In;
    uint64_t      mPacketBuffer_PA     [PACKET_BUFFER_QTY];

};
