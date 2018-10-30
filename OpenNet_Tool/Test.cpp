
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

// ===== Includes ===========================================================
#include <OpenNet/PacketGenerator.h>

// ===== OpenNet_Tool =======================================================
#include "Test.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

void Test(char aTest, TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte)
{
    assert(0 < aBufferQty      );
    assert(0 < aPacketSize_byte);

    TestLib::Tester::Describe(aTest);

    TestLib::Tester lT(aMode, false);

    lT.SetPacketSize(aPacketSize_byte);

    lT.Search(aTest, aBufferQty);
    lT.Verify(aTest, aBufferQty);

    lT.DisplaySpeed();
}

void Test(char aTest, TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s)
{
    assert(0   < aBufferQty      );
    assert(0   < aPacketSize_byte);
    assert(0.0 < aBandwidth_MiB_s);

    TestLib::Tester::Describe(aTest);

    TestLib::Tester lT(aMode, false);

    lT.SetBandwidth (aBandwidth_MiB_s);
    lT.SetPacketSize(aPacketSize_byte);

    lT.Test(aTest, aBufferQty);

    lT.DisplaySpeed();
}
