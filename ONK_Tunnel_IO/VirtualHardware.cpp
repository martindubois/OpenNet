
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/VirtualHardware.cpp

#define __CLASS__     "VirtualHardware::"
#define __NAMESPACE__ ""

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Tunnel.h>

// ===== Common =============================================================
#include "../Common/Version.h"

// ===== ONK_Tunnel_IO ======================================================
#include "VirtualHardware.h"

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define PACKET_SIZE_byte  (16 * 1024)

// Public
/////////////////////////////////////////////////////////////////////////////

VirtualHardware::VirtualHardware() : Hardware(OpenNetK::ADAPTER_TYPE_TUNNEL, PACKET_SIZE_byte)
{
    TRACE_DEBUG "VirtualHardware()" DEBUG_EOL TRACE_END;

    strcpy(mInfo.mComment                  , "ONK_Tunnel_IO"    );
    strcpy(mInfo.mVersion_Driver.mComment  , "ONK_Tunnel_IO"    );
    strcpy(mInfo.mVersion_Hardware.mComment, "OpenNet Tunnel IO");
}

unsigned int VirtualHardware::Read(void * aOut, unsigned int aOutSize_byte)
{
    TRACE_DEBUG "Read( , %u bytes )" DEBUG_EOL, aOutSize_byte TRACE_END;

    ASSERT(NULL != aOut         );
    ASSERT(NULL != aOutSize_byte);

    uint8_t    * lOut       = reinterpret_cast<uint8_t *>(aOut);
    unsigned int lResult    =             0;
    unsigned int lSize_byte = aOutSize_byte;

    uint32_t lFlags = mZone0->LockFromThread();

        ASSERT(TX_DESCRIPTOR_QTY > mTx_In );
        ASSERT(TX_DESCRIPTOR_QTY > mTx_Out);

        while (mTx_In != mTx_Out)
        {
            if (lSize_byte < (mTx_Size_byte[mTx_Out] + sizeof(OpenNet_Tunnel_PacketHeader)))
            {
                break;
            }

            unsigned int lPacketTotalSize_byte = CopyPacket_Zone0(lOut);

            lOut       += lPacketTotalSize_byte;
            lResult    += lPacketTotalSize_byte;
            lSize_byte -= lPacketTotalSize_byte;
        }

    mZone0->UnlockFromThread(lFlags);

    return lResult;
}

// ===== OpenNetK::Adapter ==================================================

void VirtualHardware::GetState(OpenNetK::Adapter_State * aState)
{
    TRACE_DEBUG "GetState(  )" DEBUG_EOL TRACE_END;

    ASSERT(NULL != aState);

    aState->mFlags.mFullDuplex = false;
    aState->mFlags.mLinkUp     = true ;
    aState->mFlags.mTx_Off     = false;
    aState->mSpeed_Mb_s        =  1000;
}

bool VirtualHardware::Packet_Drop()
{
    TRACE_DEBUG "Packet_Drop()" DEBUG_EOL TRACE_END;

    return false;
}

void VirtualHardware::Packet_Receive_NoLock(OpenNetK::Packet * aPacket, volatile long * aCounter)
{
    TRACE_DEBUG "Packet_Receive_NoLock( 0x%p, 0x%p )" DEBUG_EOL, aPacket, aCounter TRACE_END;

    ASSERT(NULL != aPacket );
    ASSERT(NULL != aCounter);

    ASSERT(false);
}

bool VirtualHardware::Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aRepeatCount)
{
    TRACE_DEBUG "Packet_Send( 0x%p, %u bytes, %u )" DEBUG_EOL, aPacket, aSize_byte, aRepeatCount TRACE_END;

    ASSERT(NULL != aPacket     );
    ASSERT(   0 <  aSize_byte  );
    ASSERT(   0 <  aRepeatCount);

    (void)(aPacket     );
    (void)(aSize_byte  );
    (void)(aRepeatCount);

    return false;
}

void VirtualHardware::Packet_Send_NoLock(uint64_t aData_PA, const void * aData_XA, unsigned int aSize_byte, volatile long * aCounter)
{
    TRACE_DEBUG "Packet_Send_NoLock( 0x%llx, , %u bytes, 0x%p )" DEBUG_EOL, aData_PA, aSize_byte, aCounter TRACE_END;

    ASSERT(               0 != aData_PA  );
    ASSERT(NULL             != aData_XA  );
    ASSERT(               0 <  aSize_byte);
    ASSERT(PACKET_SIZE_byte >  aSize_byte);
    ASSERT(NULL             != aCounter  );

    ASSERT(TX_DESCRIPTOR_QTY > mTx_In);

    mTx_Counter  [mTx_In] = aCounter  ;
    mTx_Data_XA  [mTx_In] = aData_XA  ;
    mTx_Size_byte[mTx_In] = static_cast<uint16_t>(aSize_byte);

    mTx_In = (mTx_In + 1) % TX_DESCRIPTOR_QTY;
}

void VirtualHardware::Tx_Disable()
{
    ASSERT(NULL != mZone0);

    Hardware::Tx_Disable();

    uint32_t lFlags = mZone0->LockFromThread();

        ASSERT(TX_DESCRIPTOR_QTY > mTx_In );
        ASSERT(TX_DESCRIPTOR_QTY > mTx_Out);

        while (mTx_In != mTx_Out)
        {
            (*mTx_Counter[mTx_Out])--;

            mTx_Out = (mTx_Out + 1) % TX_DESCRIPTOR_QTY;
        }

    mZone0->UnlockFromThread(lFlags);

}

void VirtualHardware::Unlock_AfterReceive_Internal()
{
    TRACE_DEBUG "Unlock_AfterReceive_Internal()" DEBUG_EOL TRACE_END;
}

void VirtualHardware::Unlock_AfterSend_Internal()
{
    TRACE_DEBUG "Unlock_AfterSend_Internal()" DEBUG_EOL TRACE_END;
}

// Private
/////////////////////////////////////////////////////////////////////////////

unsigned int VirtualHardware::CopyPacket_Zone0(uint8_t * aOut)
{
    TRACE_DEBUG "CopyPacket_Zone0(  )" DEBUG_EOL TRACE_END;

    ASSERT(NULL != aOut);

    ASSERT(TX_DESCRIPTOR_QTY > mTx_Out);

    ASSERT(NULL != mTx_Counter[mTx_Out]);
    ASSERT(NULL != mTx_Data_XA[mTx_Out]);

    uint8_t * lOut         = aOut;
    uint16_t  lResult_byte = mTx_Size_byte[mTx_Out];

    OpenNet_Tunnel_PacketHeader * lHeader = reinterpret_cast<OpenNet_Tunnel_PacketHeader *>(lOut);

    lHeader->mPacketSize_byte = lResult_byte;
    lHeader->mReserved0       = 0;
    lHeader->mSyncCheck       = OPEN_NET_SYNC_CHECK_VALUE;

    lOut += sizeof(OpenNet_Tunnel_PacketHeader);

    if (0 < lResult_byte)
    {
        memcpy(lOut, mTx_Data_XA[mTx_Out], lResult_byte);

        uint16_t lMod = lResult_byte % sizeof(uint32_t);
        if (0 != lMod)
        {
            lResult_byte += (sizeof(uint32_t) - lMod);
        }
    }

    ( * mTx_Counter[mTx_Out] ) --;

    mTx_Out = (mTx_Out + 1) % TX_DESCRIPTOR_QTY;

    return ( lResult_byte + sizeof(OpenNet_Tunnel_PacketHeader) );
}
