
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Pro1000.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/Hardware.h>
#include <OpenNetK/Types.h>

// ===== ONK_Pro1000 ========================================================
#include "Pro1000_Regs.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Pro1000 : public OpenNetK::Hardware
{

public:

    Pro1000();

    // ===== OpenNetK::Hardware =============================================
    virtual void GetState          (OpenNet_State * aStatus);
    virtual void ResetMemory       ();
    virtual void SetCommonBuffer   (uint64_t aLogicalAddress, volatile void * aVirtualAddress);
    virtual bool SetMemory         (unsigned int aIndex, volatile void * aVirtual, unsigned int aSize_byte);
    virtual bool D0_Entry          ();
    virtual bool D0_Exit           ();
    virtual void Interrupt_Disable ();
    virtual void Interrupt_Enable  ();
    virtual bool Interrupt_Process (unsigned int aMessageId, bool * aNeedMoreProcessing);
    virtual void Interrupt_Process2();
    virtual void Packet_Receive    (uint64_t aLogicalAddres, volatile OpenNet_PacketInfo * aPacketInfo, volatile long * aCounter);
    virtual void Packet_Send       (uint64_t aData, unsigned int aSize_byte, volatile long * aCounter);
    virtual void Packet_Send       (const void * aPacket, unsigned int aSize_byte);
    virtual void Stats_Get         (OpenNet_Stats * aStats, bool aReset);
    virtual void Stats_Reset       ();

private:

    enum
    {
        PACKET_BUFFER_QTY =   32,
        RX_DESCRIPTOR_QTY = 1024,
        TX_DESCRIPTOR_QTY = 1024,
    };

    void Interrupt_Disable_Zone0();

    void Reset_Zone0();

    void Rx_Config_Zone0 ();
    void Rx_Process_Zone0();

    void Stats_Update();

    void Tx_Config_Zone0 ();
    void Tx_Process_Zone0();

    // ===== Zone 0 =========================================================

    volatile Pro1000_BAR1 * mBAR1;

    volatile long                  * mRx_Counter   [RX_DESCRIPTOR_QTY];
    unsigned int                     mRx_In     ;
    uint64_t                         mRx_Logical;
    unsigned int                     mRx_Out    ;
    volatile OpenNet_PacketInfo    * mRx_PacketInfo[RX_DESCRIPTOR_QTY];
    volatile Pro1000_Rx_Descriptor * mRx_Virtual;

    volatile long                  * mTx_Counter[TX_DESCRIPTOR_QTY];
    unsigned int                     mTx_In     ;
    uint64_t                         mTx_Logical;
    unsigned int                     mTx_Out    ;
    volatile Pro1000_Tx_Descriptor * mTx_Virtual;

    unsigned int    mTx_PacketBuffer_In;
    uint64_t        mTx_PacketBuffer_Logical[PACKET_BUFFER_QTY];
    volatile void * mTx_PacketBuffer_Virtual[PACKET_BUFFER_QTY];

};
