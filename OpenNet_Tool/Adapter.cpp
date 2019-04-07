
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Tool/Adapter.cpp

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

#include "Adapter.h"

// Commandes
/////////////////////////////////////////////////////////////////////////////

// ===== Adapter Get ========================================================

static void Get_Config(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Get_State (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Get_Stats (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo GET_COMMANDS[] =
{
    { "Config", Get_Config, "Config"             , NULL },
    { "State" , Get_State , "State"              , NULL },
    { "Stats" , Get_Stats , "Stats [false|true]" , NULL },

    { NULL, NULL, NULL, NULL }
};

// ===== Adapter ============================================================

static void Display(KmsLib::ToolBase * aToolBase, const char * aArg);
static void List   (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Select (KmsLib::ToolBase * aToolBase, const char * aArg);

const KmsLib::ToolBase::CommandInfo ADAPTER_COMMANDS[] =
{
    { "Display", Display, "Display"       , NULL         },
    { "Get"    , NULL   , "Get ..."       , GET_COMMANDS },
    { "List"   , List   , "List"          , NULL         },
    { "Select" , Select , "Select {Index}", NULL         },

    { NULL, NULL, NULL, NULL }
};

// Macros
/////////////////////////////////////////////////////////////////////////////

#define VERIFY_SELECTED                                                                  \
    if (NULL == gAdapter)                                                                \
    {                                                                                    \
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "No Adapter selected"); \
        return;                                                                          \
    }

// Commands
/////////////////////////////////////////////////////////////////////////////

// ===== Adapter Get ========================================================

void Get_Config(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    VERIFY_SELECTED

    OpenNet::Adapter::Config lConfig;

    OpenNet::Status lStatus = gAdapter->GetConfig(&lConfig);
    UTL_VERIFY_STATUS("OpenNet::Adapter::GetConfig(  ) failed");

    lStatus = OpenNet::Adapter::Display(lConfig, stdout);
    UTL_VERIFY_STATUS("OpenNet::Adapter::Display( ,  ) failed");
}

void Get_State(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    VERIFY_SELECTED

    OpenNet::Adapter::State lState;

    OpenNet::Status lStatus = gAdapter->GetState(&lState);
    UTL_VERIFY_STATUS("OpenNet::Adapter::GetState(  ) failed");

    lStatus = OpenNet::Adapter::Display(lState, stdout);
    UTL_VERIFY_STATUS("OpenNet::Adapter::Display( ,  ) failed");
}

void Get_Stats(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    VERIFY_SELECTED

    bool lReset = (0 == strcmp("true", aArg));

    // TODO  OpenNet_Tool.Adapter
    //       Low (Feature) - Ajouter un argument pour le MinLevel

    unsigned int lInfo_byte;
    unsigned int lStats[1024];

    OpenNet::Status lStatus = gAdapter->GetStatistics(lStats, sizeof(lStats), &lInfo_byte, lReset);
    UTL_VERIFY_STATUS("OpenNet::Adapter::GetStatistics( , , ,  ) failed");

    lStatus = gAdapter->DisplayStatistics(lStats, lInfo_byte, stdout);
    UTL_VERIFY_STATUS("OpenNet::Adapter::DisplayStatistics( , ,  ) failed");
}

// ===== Adapter ============================================================

void Display(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    VERIFY_SELECTED
    assert(NULL != aArg);

    OpenNet::Status lStatus = gAdapter->Display(stdout);
    UTL_VERIFY_STATUS("OpenNet::Adapter::Display(  ) failed");
}

void List(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    unsigned int lCount = gSystem->Adapter_GetCount();

    for (unsigned int i = 0; i < lCount; i++)
    {
        printf("Adapter %u : %s\n", i, gSystem->Adapter_Get(i)->GetName());
    }

    printf("%u Adapter\n", lCount);
}

void Select(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    unsigned int lIndex;

    UTL_PARSE_ARGUMENT("%u", &lIndex);

    unsigned int lCount = gSystem->Adapter_GetCount();
    if (lCount <= lIndex)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid index");
        return;
    }

    gAdapter = gSystem->Adapter_Get(lIndex);
    if (NULL == gAdapter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "OpenNet::System::Adapter_Get(  ) failed");
        return;
    }

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Adapter selected");
}
