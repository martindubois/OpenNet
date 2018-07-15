
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/Test.h

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet_Tool =======================================================
#include "Loop.h"
#include "PacketSender.h"

#include "Test.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

// aSystem [---;RW-]
//
// Exception  KmsLib::Exception *  CODE_ERROR
//                                 CODE_NOT_FOUND
//                                 See Loop::Display
//                                 See Loop::GetAndDisplayAdapterStatistics
//                                 See Loop::GetAndDisplayKernelStatistics
//                                 See Loop::ResetAdapterStatistics
//                                 See Loop::SendPacket
//                                 See Loop::Start
//                                 See Loop::Stop
void Test_Loop(unsigned int aBufferQty, unsigned int aPacketSize_byte, unsigned int aPacketQty)
{
    assert(0 < aBufferQty      );
    assert(0 < aPacketSize_byte);
    assert(0 < aPacketQty      );

    Loop lLoop(aBufferQty, aPacketSize_byte, aPacketQty, Loop::MODE_CIRCLE_HALF);

    lLoop.Start();

    Sleep(2000);

    lLoop.SendPackets();

    Sleep(2000);

    lLoop.ResetAdapterStatistics();

    Sleep(10000);

    lLoop.GetAndDisplayKernelStatistics();
    lLoop.GetAdapterStatistics         ();
    lLoop.DisplayAdapterStatistics     ();
    lLoop.DisplaySpeed                 (10.0);

    lLoop.Stop();
}

void Test_Ramp(unsigned int aBufferQty, unsigned int aPacketSize_byte, unsigned int aPacketQty)
{
    assert(0 < aBufferQty);
    assert(0 < aPacketSize_byte);
    assert(0 < aPacketQty);

    Loop         lLoop(aBufferQty, aPacketSize_byte, 1, Loop::MODE_CIRCLE_FULL);
    PacketSender lPacketSender0(lLoop.mAdapters[0], aPacketSize_byte, aPacketQty);
    PacketSender lPacketSender1(lLoop.mAdapters[1], aPacketSize_byte, aPacketQty);

    lLoop         .Start();
    lPacketSender0.Start();
    lPacketSender1.Start();

    lLoop.ResetAdapterStatistics();

    for (unsigned int i = 0; i < 15; i++)
    {
        Sleep(1000);

        lLoop.GetAdapterStatistics    ();
        lLoop.DisplaySpeed            (1.0);
    }

    printf("\nPacketSender Stopped!\n\n");
    lPacketSender0.Stop();
    lPacketSender1.Stop();

    for (unsigned int i = 0; i < 10; i++)
    {
        Sleep(1000);

        lLoop.GetAdapterStatistics    ();
        lLoop.DisplaySpeed            (1.0);
    }

    lLoop.Stop();
}
