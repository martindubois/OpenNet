
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Tool/Utils.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Status.h>

// Macros
/////////////////////////////////////////////////////////////////////////////

#define UTL_PARSE_ARGUMENT(F,A)                                                           \
    if (1 != scanf_s(aArg, (F), (A)))                                                     \
    {                                                                                     \
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Invalid command line"); \
        return;                                                                           \
    }

#define UTL_VERIFY_STATUS(M)             \
    if ( OpenNet::STATUS_OK != lStatus ) \
    {                                    \
        Utl_ReportError((M), lStatus);   \
        return;                          \
    }

// Functions
/////////////////////////////////////////////////////////////////////////////

extern void Utl_ReportError(const char * aMsg, OpenNet::Status aStatus);
