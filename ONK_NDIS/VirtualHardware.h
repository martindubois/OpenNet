
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/VirtualHardware.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/Hardware.h>
#include <OpenNetK/Packet.h>
#include <OpenNetK/Types.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class VirtualHardware : public OpenNetK::Hardware
{

public:

    typedef bool Tx_Callback(void * aContext, const void * aData, unsigned int aSize_byte);

    VirtualHardware();

    void Rx_IndicatePacket(const void * aData, unsigned int aSize_byte);

    void Tx_RegisterCallback(Tx_Callback aCallback, void * aContext);

    // ===== OpenNetK::Hardware =============================================
    virtual void         GetState             (OpenNetK::Adapter_State * aStats);
    virtual bool         D0_Entry             ();
    virtual void         Packet_Receive_NoLock(uint64_t aLogicalAddress, OpenNetK::Packet * aPacketData, OpenNet_PacketInfo * aPacketInfo, volatile long * aCounter);
    virtual void         Packet_Send_NoLock   (uint64_t aLogicalAddress, const void * aVirtualAddress, unsigned int aSize_byte, volatile long * aCounter = NULL);
    virtual void         Packet_Send          (const void * aPacket, unsigned int aSize_byte, unsigned int aRepeatCount = 1);

private:

    enum
    {
        RX_DESCRIPTOR_QTY = 32 * 1024,
    };

    Tx_Callback * mTx_Callback;
    void        * mTx_Context ;

    // ===== Zone 0 =========================================================

    volatile long      * mRx_Counter   [RX_DESCRIPTOR_QTY];
    uint64_t             mRx_Data      [RX_DESCRIPTOR_QTY];
    unsigned int         mRx_In ;
    unsigned int         mRx_Out;
    OpenNetK::Packet   * mRx_PacketData[RX_DESCRIPTOR_QTY];
    OpenNet_PacketInfo * mRx_PacketInfo[RX_DESCRIPTOR_QTY];

};
