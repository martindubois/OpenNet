
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/Test.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Common =============================================================
#include "../Common/TestLib/Tester.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

extern void Test_A(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte);
extern void Test_A(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s);
extern void Test_B(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte);
extern void Test_B(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s);
extern void Test_C(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s);
extern void Test_D(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte);
extern void Test_D(TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s);
