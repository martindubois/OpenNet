
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved
// Product    OpenNet
// File       TestLib/Code.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Common =============================================================
#include "../Common/TestLib/Code.h"

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    const char * mName;

    const char * mKernelCode;

    const char * mFunctionCodes[2];
    const char * mFunctionNames[2];
}
CodeInfo;

// Global constants
/////////////////////////////////////////////////////////////////////////////

extern const CodeInfo CODES[TestLib::CODE_QTY];

