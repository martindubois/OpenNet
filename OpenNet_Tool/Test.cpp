
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

// ===== Common =============================================================
#include "../Common/TestLib/TestDual.h"

// ===== OpenNet_Tool =======================================================
#include "Test.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

void Test_A(unsigned int aBufferQty, unsigned int aPacketSize_byte)
{
    assert(0 < aBufferQty      );
    assert(0 < aPacketSize_byte);

    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    lTD.A_Search(aBufferQty, aPacketSize_byte, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME);

    lTD.A_Verify(aBufferQty, aPacketSize_byte, lTD.mPacketGenerator_Config.mBandwidth_MiB_s, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME);

    lTD.DisplaySpeed();
}

void Test_A(unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s)
{
    assert(0   < aBufferQty      );
    assert(0   < aPacketSize_byte);
    assert(0.0 < aBandwidth_MiB_s);

    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    lTD.A(aBufferQty, aPacketSize_byte, aBandwidth_MiB_s, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME);

    lTD.DisplaySpeed();
}

void Test_B(unsigned int aBufferQty, unsigned int aPacketSize_byte)
{
    assert(0 < aBufferQty      );
    assert(0 < aPacketSize_byte);

    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    lTD.B_Search(aBufferQty, aPacketSize_byte, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME);

    lTD.B_Verify(aBufferQty, aPacketSize_byte, lTD.mPacketGenerator_Config.mBandwidth_MiB_s, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME);

    lTD.DisplaySpeed();
}

void Test_B(unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s)
{
    assert(0   < aBufferQty      );
    assert(0   < aPacketSize_byte);
    assert(0.0 < aBandwidth_MiB_s);

    TestLib::TestDual lTD(TestLib::TestDual::MODE_FUNCTION, false);

    lTD.B(aBufferQty, aPacketSize_byte, aBandwidth_MiB_s, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME);

    lTD.DisplaySpeed();
}

void Test_C(unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s)
{
    assert(0   < aBufferQty      );
    assert(0   < aPacketSize_byte);
    assert(0.0 < aBandwidth_MiB_s);

    TestLib::TestDual lTD(TestLib::TestDual::MODE_KERNEL, false);

    lTD.C(aBufferQty, aPacketSize_byte, aBandwidth_MiB_s, TestLib::TestDual::ADAPTER_SELECT_CARD_SAME);

    lTD.DisplaySpeed();
}
