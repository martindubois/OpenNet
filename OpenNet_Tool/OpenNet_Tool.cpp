
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Tool/OpenNet_Tool.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdio.h>

// ===== C++ ================================================================
#include <map>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>
#include <KmsLib/ToolBase.h>
#include <KmsTool.h>

// ===== Includes ===========================================================
#include <OpenNet/Kernel_Forward.h>
#include <OpenNetK/Constants.h>

// ===== Common =============================================================
#include "../Common/Version.h"

// ===== OpenNet_Tool =======================================================
#include "Adapter.h"
#include "Globals.h"
#include "PacketGenerator.h"
#include "Processor.h"
#include "Test.h"
#include "Utils.h"

// Commands
/////////////////////////////////////////////////////////////////////////////

static void Kernel_Forward_AddDestination   (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Forward_Create           (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Forward_List             (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Forward_RemoveDestination(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Forward_ResetDestinations(KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo KERNEL_FORWARD_COMMANDS[] =
{
    { "AddDestination"   , Kernel_Forward_AddDestination   , "AddDestination [Adapter]"   , NULL },
    { "Create"           , Kernel_Forward_Create           , "Create {Name}"              , NULL },
    { "List"             , Kernel_Forward_List             , "List"                       , NULL },
    { "RemoveDestination", Kernel_Forward_RemoveDestination, "RemoveDestination [Adapter]", NULL },
    { "ResetDestinations", Kernel_Forward_ResetDestinations, "RestedDestinations"         , NULL },

    { NULL, NULL, NULL, NULL }
};

static void Kernel_Create      (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Delete      (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Display     (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Edit_Remove (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Edit_Replace(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Edit_Search (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_GetCodeSize (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_List        (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_ResetCode   (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Select      (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_SetCode     (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo KERNEL_COMMANDS[] =
{
    { "Create"      , Kernel_Create      , "Create {Name}"              , NULL                    },
    { "Delete"      , Kernel_Delete      , "Delete"                     , NULL                    },
    { "Display"     , Kernel_Display     , "Display"                    , NULL                    },
    { "Edit_Remove" , Kernel_Edit_Remove , "Edit_Remote {Remove}"       , NULL                    },
    { "Edit_Replace", Kernel_Edit_Replace, "Edit_Replace {Search} [Rep]", NULL                    },
    { "Edit_Search" , Kernel_Edit_Search , "Edit_Search {Search}"       , NULL                    },
    { "Forward"     , NULL               , "Forward ..."                , KERNEL_FORWARD_COMMANDS },
    { "List"        , Kernel_List        , "List"                       , NULL                    },
    { "GetCodeSize" , Kernel_GetCodeSize , "GetCodeSize"                , NULL                    },
    { "ResetCode"   , Kernel_ResetCode   , "ResetCode"                  , NULL                    },
    { "Select"      , Kernel_Select      , "Select {Name}"              , NULL                    },
    { "SetCode"     , Kernel_SetCode     , "SetCode {FileName}"         , NULL                    },

    { NULL, NULL, NULL, NULL }
};

static void Display(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Exit   (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Start  (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Stop   (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo COMMANDS[] =
{
	{ "Adapter"        , NULL                           , "Adapter ..."          , ADAPTER_COMMANDS          },
    { "Display"        , Display                        , "Display"              , NULL                      },
	{ "ExecuteScript"  , KmsLib::ToolBase::ExecuteScript, "ExecuteSript {Script}", NULL                      },
	{ "Exit"           , Exit                           , "Exit"                 , NULL                      },
    { "Kernel"         , NULL                           , "Kernel ..."           , KERNEL_COMMANDS           },
    { "PacketGenerator", NULL                           , "PacketGenerator ..."  , PACKET_GENERATOR_COMMANDS },
	{ "Processor"      , NULL                           , "Processor ..."        , PROCESSOR_COMMANDS        },
    { "Start"          , Start                          , "Start"                , NULL                      },
    { "Stop"           , Stop                           , "Stop"                 , NULL                      },
    { "Test"           , NULL                           , "Test ..."             , TEST_COMMANDS             },

	{ NULL, NULL, NULL, NULL }
};

// Static functions declaration
/////////////////////////////////////////////////////////////////////////////

static void ReportStatus(OpenNet::Status aStatus, const char * aMsgOK);

static void System_Connect();

// Global variable
/////////////////////////////////////////////////////////////////////////////

typedef std::map<std::string, OpenNet::Kernel *> KernelMap;

static OpenNet::Kernel    * sKernel    = NULL;
static KernelMap            sKernels;
static OpenNet::Processor * sProcessor = NULL;

// Entry point
/////////////////////////////////////////////////////////////////////////////

int main(int aCount, const char ** aVector)
{
    KMS_TOOL_BANNER("OpenNet", "OpenNet_Tool", VERSION_STR, VERSION_TYPE);
    printf("Purchased by " VERSION_CLIENT "\n");

    try
    {
        System_Connect();

	    KmsLib::ToolBase lToolBase(COMMANDS);

        if (!lToolBase.ParseArguments(aCount, aVector))
	    {
		    lToolBase.ParseCommands();
	    }

        PacketGenerator_Exit();

	    gSystem->Delete();
    }
    catch ( KmsLib::Exception * eE )
    {
        printf( "FATAL ERROR  Exception\n" );
        eE->Write( stdout );
    }
    catch ( ... )
    {
        printf( "FATAL ERROR  Unknown exception\n" );
    }

    return 0;
}

// Commands
/////////////////////////////////////////////////////////////////////////////

void Display(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    OpenNet::Status lStatus = gSystem->Display(stdout);
    UTL_VERIFY_STATUS("OpenNet::System::Display(  ) failed");
}

void Exit(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    PacketGenerator_Exit();

    gSystem->Delete();

    exit( 0 );
}

void Kernel_Forward_AddDestination(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    OpenNet::Adapter * lAdapter;
    unsigned int       lAdapterIndex;

    switch (sscanf_s(aArg, "%u", &lAdapterIndex))
    {
    case EOF:
        printf("Kernel_Forward AddDestination\n");
        if (NULL == gAdapter)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
            return;
        }
        lAdapter = gAdapter;
        break;

    case 1:
        printf("Kernel_Forward AddDestination %u\n", lAdapterIndex);

        if (gSystem->Adapter_GetCount() >= lAdapterIndex)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid adapter index");
            return;
        }

        lAdapter = gSystem->Adapter_Get(lAdapterIndex);
        assert(NULL != lAdapter);
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
        return;
    }

    if (NULL == sKernel)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No kernel selected");
        return;
    }

    OpenNet::Kernel_Forward * lKF = dynamic_cast<OpenNet::Kernel_Forward *>(sKernel);
    if (NULL == lKF)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "The selected Kernel is not a Kernel_Forward");
        return;
    }
    
    ReportStatus(lKF->AddDestination(lAdapter), "Destination added");
}

void Kernel_Forward_Create(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    char lName[64];

    switch (sscanf_s(aArg, "%s", lName SIZE_INFO(static_cast<unsigned int>(sizeof(lName)))))
    {
    case 1:
        printf("Kernel_Forward Create %s\n", lName);

        if (sKernels.end() != sKernels.find(lName))
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "The name already exist");
            return;
        }

        sKernel = new OpenNet::Kernel_Forward();

        sKernel->SetName(lName);

        sKernels.insert(KernelMap::value_type(lName, sKernel));

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Kernel_Forward created");
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Kernel_Forward_List(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    for (KernelMap::iterator lIt = sKernels.begin(); lIt != sKernels.end(); lIt++)
    {
        assert(NULL != lIt->second);

        OpenNet::Kernel_Forward * lKF = dynamic_cast<OpenNet::Kernel_Forward *>(lIt->second);
        if (NULL != lKF)
        {
            printf("Kernel_Forward : %s\n", lIt->first.c_str());

			lKF->Display(stdout);
        }
    }

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Kernel_Forward listed");
}

void Kernel_Forward_RemoveDestination(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    OpenNet::Adapter * lAdapter;
    unsigned int       lAdapterIndex;

    switch (sscanf_s(aArg, "%u", &lAdapterIndex))
    {
    case EOF:
        printf("Kernel_Forward RemoveDestination\n");
        if (NULL == gAdapter)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
            return;
        }
        lAdapter = gAdapter;
        break;

    case 1:
        printf("Kernel_Forward RemoveDestination %u\n", lAdapterIndex);

        if (gSystem->Adapter_GetCount() >= lAdapterIndex)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid adapter index");
            return;
        }

        lAdapter = gSystem->Adapter_Get(lAdapterIndex);
        assert(NULL != lAdapter);
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
        return;
    }

    if (NULL == sKernel)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
        return;
    }

    OpenNet::Kernel_Forward * lKF = dynamic_cast<OpenNet::Kernel_Forward *>(sKernel);
    if (NULL == lKF)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "The selected filter is not a Kernel_Forward");
        return;
    }

    ReportStatus(lKF->RemoveDestination(lAdapter), "Destination removed");
}

void Kernel_Forward_ResetDestinations(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    if (NULL == sKernel)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No Kernel selected");
        return;
    }

    OpenNet::Kernel_Forward * lKF = dynamic_cast<OpenNet::Kernel_Forward *>(sKernel);
    if (NULL == lKF)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "The selected filter is not a Kernel_Forward");
        return;
    }

    ReportStatus(lKF->ResetDestinations(), "Destination reset");
}

void Kernel_Create(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    char lName[64];

    switch (sscanf_s(aArg, "%s", lName SIZE_INFO(static_cast<unsigned int>(sizeof(lName)))))
    {
    case 1:
        printf("Kernel Create %s\n", lName);

        if (sKernels.end() != sKernels.find(lName))
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "The name already exist");
            return;
        }

        sKernel = new OpenNet::Kernel();

        sKernel->SetName(lName);

        sKernels.insert(KernelMap::value_type(lName, sKernel));

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Kernel created");
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Kernel_Delete(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    if (NULL == sKernel)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
        return;
    }

    delete sKernel;

    for (KernelMap::iterator lIt = sKernels.begin(); lIt != sKernels.end(); lIt++)
    {
        assert(NULL != lIt->second);

        if (sKernel == lIt->second)
        {
            sKernels.erase(lIt);
            break;
        }
    }

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Kernel deleted");
}

void Kernel_Display(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    if (NULL == sKernel)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No Kernel selected");
        return;
    }

    OpenNet::Status lStatus = sKernel->Display(stdout);
    assert(OpenNet::STATUS_OK == lStatus);
}

void Kernel_Edit_Remove(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    char lRemove[64];

    switch (sscanf_s(aArg, "%s", lRemove SIZE_INFO(static_cast<unsigned int>(sizeof(lRemove)))))
    {
    case 1:
        if (NULL == sKernel)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No Kernel selected");
            return;
        }

        char lMsg[64];

        sprintf_s(lMsg, "%u occurence removed", sKernel->Edit_Remove(lRemove));

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, lMsg);
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Kernel_Edit_Search(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    char lSearch[64];

    switch (sscanf_s(aArg, "%s", lSearch SIZE_INFO(static_cast<unsigned int>(sizeof(lSearch)))))
    {
    case 1:
        if (NULL == sKernel)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No Kernel selected");
            return;
        }

        char lMsg[64];

        sprintf_s(lMsg, "%u occurence found", sKernel->Edit_Search(lSearch));

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, lMsg);
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Kernel_Edit_Replace(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    char lSearch [64];
    char lReplace[64];

    switch (sscanf_s(aArg, "%s %s", lSearch SIZE_INFO(static_cast<unsigned int>(sizeof(lSearch))), lReplace SIZE_INFO(static_cast<unsigned int>(sizeof(lReplace)))))
    {
    case 1 :
        strcpy_s(lReplace, "");
        // no break;

    case 2:
        printf("Kernel Edit_Search %s %s\n", lSearch, lReplace);

        if (NULL == sKernel)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No Kernel selected");
            return;
        }

        char lMsg[64];

        sprintf_s(lMsg, "%u occurence replaced", sKernel->Edit_Replace(lSearch, lReplace));

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, lMsg);
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Kernel_GetCodeSize(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    if (NULL == sKernel)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No Kernel selected");
        return;
    }
 
    unsigned int lCodeSize_byte = sKernel->GetCodeSize();

    char lMsg[64];

    sprintf_s(lMsg, "%u bytes", lCodeSize_byte);

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, lMsg);
}

void Kernel_List(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    for (KernelMap::iterator lIt = sKernels.begin(); lIt != sKernels.end(); lIt++)
    {
        assert(NULL != lIt->second);

        printf("Kernel : %s\n", lIt->first.c_str());

        lIt->second->Display(stdout);
    }

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Kernel listed");
}

void Kernel_ResetCode(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    if (NULL == sKernel)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No Kernel selected");
        return;
    }
 
    ReportStatus(sKernel->ResetCode(), "Code reset");
}

void Kernel_Select(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    KernelMap::iterator lIt;

    char lName[64];

    switch (sscanf_s(aArg, "%s", lName SIZE_INFO(static_cast<unsigned int>(sizeof(lName)))))
    {
    case 1:
        printf("Kernel Select %s\n", lName);

        lIt = sKernels.find(lName);

        if (sKernels.end() == lIt)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid name");
            return;
        }

        assert(NULL != lIt->second);

        sKernel = lIt->second;

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Kernel selected");
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Kernel_SetCode(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    KernelMap::iterator lIt;

    char lFileName[64];

    switch (sscanf_s(aArg, "%s", lFileName SIZE_INFO(static_cast<unsigned int>(sizeof(lFileName)))))
    {
    case 1:
        printf("Filter Select %s\n", lFileName);

        if (NULL == sKernel)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No Kernel selected");
            return;
        }

        ReportStatus(sKernel->SetCode(lFileName), "Code set");
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Stop(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    OpenNet::Status lStatus = gSystem->Stop();
    UTL_VERIFY_STATUS("OpenNet::System::Stop() failed");

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "System stopped");
}

void Start(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Start %s\n", aArg);
    printf("Start\n");

    OpenNet::Status lStatus = gSystem->Start(0);
    UTL_VERIFY_STATUS("OpenNet::System::Start(  ) failed");

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "System started");
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

void ReportStatus(OpenNet::Status aStatus, const char * aMsgOK)
{
    assert(NULL != aMsgOK);

    if (OpenNet::STATUS_OK != aStatus)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, OpenNet::Status_GetDescription(aStatus));
        return;
    }

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, aMsgOK);
}

void System_Connect()
{
    printf("Connecting...\n");

    gSystem = OpenNet::System::Create();
    if (NULL == gSystem)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_FATAL_ERROR, "OpenNet::System::Create() failed");
        exit(__LINE__);
    }
}
