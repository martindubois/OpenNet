
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Tool/Processor.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>

// ===== Import/Includes ====================================================
#include <KmsLib/ToolBase.h>

// ===== OpenNet_Tool =======================================================
#include "Globals.h"
#include "Utils.h"

#include "Processor.h"

// Commandes
/////////////////////////////////////////////////////////////////////////////

// ===== Processor ==========================================================

static void Display(KmsLib::ToolBase * aToolBase, const char * aArg);
static void List   (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Select (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo PROCESSOR_COMMANDS[] =
{
    { "Display", Display, "Display"       , NULL },
    { "List"   , List   , "List"          , NULL },
    { "Select" , Select , "Select {Index}", NULL },

    { NULL, NULL, NULL, NULL }
};

// Static variables
/////////////////////////////////////////////////////////////////////////////

static OpenNet::Processor * sProcessor = NULL;

// Macros
/////////////////////////////////////////////////////////////////////////////

#define VERIFY_SELECTED                                                                    \
    if (NULL == sProcessor)                                                                \
    {                                                                                      \
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "No Processor selected"); \
        return;                                                                            \
    }

// Commands
/////////////////////////////////////////////////////////////////////////////

// ===== Processor ==========================================================

void Display(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    VERIFY_SELECTED

    OpenNet::Status lStatus = sProcessor->Display(stdout);
    UTL_VERIFY_STATUS("OpenNet::Processor::Display(  ) failed");
}

void List(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    unsigned int lCount = gSystem->Processor_GetCount();

    for (unsigned int i = 0; i < lCount; i++)
    {
        printf("Processor %u : %s\n", i, gSystem->Processor_Get(i)->GetName());
    }

    printf("%u Processor\n", lCount);
}

void Select(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    unsigned int lIndex;

    UTL_PARSE_ARGUMENT("%u", &lIndex);

    unsigned int lCount = gSystem->Processor_GetCount();

    if (lCount <= lIndex)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid index");
        return;
    }

    sProcessor = gSystem->Processor_Get(lIndex);
    if (NULL == sProcessor)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "OpenNet::System::Processor_Get(  ) failed");
        return;
    }

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Processor selected");
}
