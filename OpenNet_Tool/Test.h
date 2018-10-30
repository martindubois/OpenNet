
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

extern void Test(char aTest, TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte);
extern void Test(char aTest, TestLib::Tester::Mode aMode, unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s);
