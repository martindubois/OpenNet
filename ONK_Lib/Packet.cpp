
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Lib/Packet.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================
#include <ntddk.h>

// ===== Includes ===========================================================
#include <OpenNetK/StdInt.h>
#include <OpenNetK/Types.h>

#include <OpenNetK/Packet.h>

namespace OpenNetK
{

    // Internal
    /////////////////////////////////////////////////////////////////////////

    // aOffset_byte              The offset from the begining of the buffer
    // aVirtualAddress [-K-;RW-] The data's virtual address into the kernel
    //                           address space
    void Packet::Init(uint32_t aOffset_byte, void * aVirtualAddress)
    {
        ASSERT(   0 <  aOffset_byte   );
        ASSERT(NULL != aVirtualAddress);

        mOffset_byte    = aOffset_byte             ;
        mSendTo         = OPEN_NET_PACKET_PROCESSED;
        mState          = STATE_TX_RUNNING         ;
        mVirtualAddress = aVirtualAddress          ;
    }

}
