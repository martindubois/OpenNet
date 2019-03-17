
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Tool/Utils.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Import/Includes ====================================================
#include <KmsLib/ToolBase.h>

// ===== OpenNet_Tool =======================================================
#include "Utils.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

void Utl_ReportError(const char * aMsg, OpenNet::Status aStatus)
{
    assert(NULL != aMsg);

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, aMsg);
    OpenNet::Status_Display(aStatus, stdout);
    printf("\n");
}