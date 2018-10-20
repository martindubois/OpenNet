
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

void Test_A(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte)
{
    assert(0 < aBufferQty      );
    assert(0 < aPacketSize_byte);

    TestLib::Tester::A_Describe();

    TestLib::Tester lT(aMode, false);

    lT.SetPacketSize(aPacketSize_byte);

    lT.A_Search(aBufferQty);
    lT.A_Verify(aBufferQty);

    lT.DisplaySpeed();
}

void Test_A(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s)
{
    assert(0   < aBufferQty      );
    assert(0   < aPacketSize_byte);
    assert(0.0 < aBandwidth_MiB_s);

    TestLib::Tester::A_Describe();

    TestLib::Tester lT(aMode, false);

    lT.SetBandwidth (aBandwidth_MiB_s);
    lT.SetPacketSize(aPacketSize_byte);

    lT.A(aBufferQty);

    lT.DisplaySpeed();
}

void Test_B(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte)
{
    assert(0 < aBufferQty      );
    assert(0 < aPacketSize_byte);

    TestLib::Tester::B_Describe();

    TestLib::Tester lT(aMode, false);

    lT.SetPacketSize(aPacketSize_byte);

    lT.B_Search(aBufferQty);
    lT.B_Verify(aBufferQty);

    lT.DisplaySpeed();
}

void Test_B(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s)
{
    assert(0   < aBufferQty      );
    assert(0   < aPacketSize_byte);
    assert(0.0 < aBandwidth_MiB_s);

    TestLib::Tester::B_Describe();

    TestLib::Tester lT(aMode, false);

    lT.SetBandwidth (aBandwidth_MiB_s);
    lT.SetPacketSize(aPacketSize_byte);

    lT.B(aBufferQty);

    lT.DisplaySpeed();
}

void Test_C(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s)
{
    assert(0   < aBufferQty      );
    assert(0   < aPacketSize_byte);
    assert(0.0 < aBandwidth_MiB_s);

    TestLib::Tester::C_Describe();

    TestLib::Tester lT(aMode, false);

    lT.SetBandwidth (aBandwidth_MiB_s);
    lT.SetPacketSize(aPacketSize_byte);

    lT.C(aBufferQty);

    lT.DisplaySpeed();
}

void Test_D(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte)
{
    assert(0 < aBufferQty      );
    assert(0 < aPacketSize_byte);

    TestLib::Tester::D_Describe();

    TestLib::Tester lT(aMode, false);

    lT.SetPacketSize(aPacketSize_byte);

    lT.D_Search(aBufferQty);
    lT.D_Verify(aBufferQty);

    lT.DisplaySpeed();
}

void Test_D(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s)
{
    assert(0 < aBufferQty);
    assert(0 < aPacketSize_byte);
    assert(0.0 < aBandwidth_MiB_s);

    TestLib::Tester::D_Describe();

    TestLib::Tester lT(aMode, false);

    lT.SetBandwidth (aBandwidth_MiB_s);
    lT.SetPacketSize(aPacketSize_byte);

    lT.D(aBufferQty);

    lT.DisplaySpeed();
}
