
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/VirtualHardware.h

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

    VirtualHardware();

    unsigned int Read(void * aOut, unsigned int aOutSize_byte);

    // ===== OpenNetK::Hardware =============================================
    virtual void GetState                    (OpenNetK::Adapter_State * aStats);
    virtual bool Packet_Drop                 ();
    virtual void Packet_Receive_NoLock       (OpenNetK::Packet * aPacket, volatile long * aCounter);
    virtual bool Packet_Send                 (const void * aPacket, unsigned int aSize_byte, unsigned int aRepeatCount = 1);
    virtual void Packet_Send_NoLock          (uint64_t aPacket_PA, const void * aPacket_XA, unsigned int aSize_byte, volatile long * aCounter);
    virtual void Tx_Disable                  ();
    virtual void Unlock_AfterReceive_Internal();
    virtual void Unlock_AfterSend_Internal   ();

private:

    enum
    {
        TX_DESCRIPTOR_QTY = 32 * 1024,
    };

    unsigned int CopyPacket_Zone0(uint8_t * aOut);

    // ===== Zone 0 =========================================================

    volatile long * mTx_Counter  [TX_DESCRIPTOR_QTY];
    const void    * mTx_Data_XA  [TX_DESCRIPTOR_QTY]; // C or M
    unsigned int    mTx_In ;
    unsigned int    mTx_Out;
    uint16_t        mTx_Size_byte[TX_DESCRIPTOR_QTY];

};
