
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Lib/Packet.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/OS.h>
#include <OpenNetK/StdInt.h>
#include <OpenNetK/Types.h>

#include <OpenNetK/Packet.h>

namespace OpenNetK
{

    // Internal
    /////////////////////////////////////////////////////////////////////////

    // aData_PA [-K-;RW-] The physical address of the data
    // aData_XA [-K-;RW-] The address of data into the kernel address space
    //                    (C or M)
    // aInfo_XA [-K-;RW-] The address of the OpenNet_PacketInfo structure
    //                    into the kernel address space (C or M)
    void Packet::Init(uint64_t aData_PA, void * aData_XA, OpenNet_PacketInfo * aInfo_XA)
    {
        ASSERT(NULL != aData_XA);
        ASSERT(NULL != aInfo_XA);

        mData_PA = aData_PA                 ;
        mData_XA = aData_XA                 ;
        mInfo_XA = aInfo_XA                 ;
        mSendTo  = OPEN_NET_PACKET_PROCESSED;
        mState   = STATE_TX_RUNNING         ;
    }

}
