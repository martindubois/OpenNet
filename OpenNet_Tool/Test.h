
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/Test.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/System.h>

// Functions
/////////////////////////////////////////////////////////////////////////////

extern void Test_A(unsigned int aBufferQty, unsigned int aPacketSize_byte);
extern void Test_A(unsigned int aBufferQty, unsigned int aPacketSize_byte, double       aBandwidth_MiB_s);
extern void Test_B(unsigned int aBufferQty, unsigned int aPacketSize_byte, unsigned int aBandwidth_MiB_s);
