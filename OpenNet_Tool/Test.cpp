
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

#include "Test.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

// aSystem [---;RW-]
//
// Exception  KmsLib::Exception *  CODE_ERROR
//                                 CODE_NOT_FOUND
//                                 See Loop::Display
//                                 See Loop::GetAndDisplayStatistics
//                                 See Loop::GetStatistics
//                                 See Loop::ResetStatistics
//                                 See Loop::SendPacket
//                                 See Loop::Start
//                                 See Loop::Stop
void Test_Loop(unsigned int aBufferQty, unsigned int aPacketSize_byte, unsigned int aPacketQty)
{
    assert(0 < aBufferQty      );
    assert(0 < aPacketSize_byte);
    assert(0 < aPacketQty      );

    Loop lLoop(aBufferQty, aPacketSize_byte, aPacketQty, Loop::MODE_DOUBLE_MIRROR);

    lLoop.Start();

    Sleep(2000);

    lLoop.SendPackets();

    Sleep(2000);

    lLoop.ResetStatistics();

    Sleep(10000);

    lLoop.GetAndDisplayStatistics();
    lLoop.GetStatistics          ();
    lLoop.Display                ();
    lLoop.Stop                   ();
}
