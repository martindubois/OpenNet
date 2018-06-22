
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Utilities.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>

// Constants
/////////////////////////////////////////////////////////////////////////////

#define UTL_MASK_EQUAL          (0)
#define UTL_MASK_IGNORE         (1)
#define UTL_MASK_ABOVE          (2)
#define UTL_MASK_ABOVE_OR_EQUAL (3)
#define UTL_MASK_BELOW          (4)
#define UTL_MASK_BELOW_OR_EQUAL (5)
#define UTL_MASK_DIFFERENT      (6)

#define UTL_MASK_QTY (7)

// Functions
/////////////////////////////////////////////////////////////////////////////

extern unsigned int Utl_Validate    (const OpenNet::Adapter::Stats & aIn, const OpenNet::Adapter::Stats & aExpected, const OpenNet::Adapter::Stats & aMask);
extern void         Utl_ValidateInit(OpenNet::Adapter::Stats * aExpected, OpenNet::Adapter::Stats * aMask);
