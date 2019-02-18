
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
#include <KmsLib/ToolBase.h>
#include <KmsTool.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Kernel_Forward.h>
#include <OpenNetK/Constants.h>

// ===== Common =============================================================
#include "../Common/Version.h"
#include "../Common/TestLib/TestFactory.h"

// ===== OpenNet_Tool =======================================================
#include "Test.h"

// Commands
/////////////////////////////////////////////////////////////////////////////

static void Adapter_Display  (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Adapter_GetConfig(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Adapter_GetState (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Adapter_GetStats (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Adapter_List     (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Adapter_Select   (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo ADAPTER_COMMANDS[] =
{
    { "Display"  , Adapter_Display  , "Display                       Display the adapter"  , NULL },
    { "GetConfig", Adapter_GetConfig, "GetConfig                     Display configuration", NULL },
    { "GetState" , Adapter_GetState , "GetState                      Display state"        , NULL },
    { "GetStats" , Adapter_GetStats , "GetStats [false|true]         Display statistics"   , NULL },
	{ "List"     , Adapter_List     , "List                          List the adapters"    , NULL },
	{ "Select"   , Adapter_Select   , "Select {Index}                Select an adapter"    , NULL },

	{ NULL, NULL, NULL, NULL }
};

static void Kernel_Forward_AddDestination   (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Forward_Create           (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Forward_List             (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Forward_RemoveDestination(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Kernel_Forward_ResetDestinations(KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo KERNEL_FORWARD_COMMANDS[] =
{
    { "AddDestination"   , Kernel_Forward_AddDestination   , "AddDestination [Adapter]      Add a destination"               , NULL },
    { "Create"           , Kernel_Forward_Create           , "Create                        Create a Kernel_Forward instance", NULL },
    { "List"             , Kernel_Forward_List             , "List                          List the Kernel_Forward instance", NULL },
    { "RemoveDestination", Kernel_Forward_RemoveDestination, "RemoveDestination [Adapter]   Remove a destination"            , NULL },
    { "ResetDestinations", Kernel_Forward_ResetDestinations, "RestedDestinations            Remove all destiations"          , NULL },

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
    { "Create"      , Kernel_Create      , "Create {Name}                 Create a Kernel instance"  , NULL },
    { "Delete"      , Kernel_Delete      , "Delete                        Delete the Kernel instance", NULL },
    { "Display"     , Kernel_Display     , "Display                       Display the Kernel"        , NULL },
    { "Edit_Remove" , Kernel_Edit_Remove , "Edit_Remote {Remove}          Remove strings from code"  , NULL },
    { "Edit_Replace", Kernel_Edit_Replace, "Edit_Replace {Search} [Rep]   Search and replace strings", NULL },
    { "Edit_Search" , Kernel_Edit_Search , "Edit_Search {Search}          Search a string"           , NULL },
    { "Forward"     , NULL               , "Forward ..."                                             , KERNEL_FORWARD_COMMANDS },
    { "List"        , Kernel_List        , "List                          List the Kernel instances" , NULL },
    { "GetCodeSize" , Kernel_GetCodeSize , "GetCodeSize                   Display code size"         , NULL },
    { "ResetCode"   , Kernel_ResetCode   , "ResetCode                     Reset the code"            , NULL },
    { "Select"      , Kernel_Select      , "Select {Name}                 Select a Kernel"           , NULL },
    { "SetCode"     , Kernel_SetCode     , "SetCode {FileName}            Set the code"              , NULL },

    { NULL, NULL, NULL, NULL }
};

static void Processor_Display(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Processor_List   (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Processor_Select (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo PROCESSOR_COMMANDS[] =
{
	{ "Display", Processor_Display, "Display                       Display the processor", NULL },
	{ "List"   , Processor_List   , "List                          List the processors"  , NULL },
	{ "Select" , Processor_Select , "Select {Index}                Select an processor"  , NULL },

	{ NULL, NULL, NULL, NULL }
};

static void Test_Search_Bandwidth (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_Search_BufferQty (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_Search_PacketSize(KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo TEST_SEARCH_COMMANDS[] =
{
    { "Bandwidth ", Test_Search_Bandwidth , "Bandwidth  {TestName}", NULL },
    { "BufferQty ", Test_Search_BufferQty , "BufferQty  {TestName}", NULL },
    { "PacketSize", Test_Search_PacketSize, "PacketSize {TestName}", NULL },

    { NULL, NULL, NULL, NULL }
};

static void Test_Verify_Bandwidth (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_Verify_BufferQty (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_Verify_PacketSize(KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo TEST_VERIFY_COMMANDS[] =
{
    { "Bandwidth ", Test_Verify_Bandwidth , "Bandwidth {TestName}" , NULL },
    { "BufferQty ", Test_Verify_BufferQty , "BufferQty {TestName}" , NULL },
    { "PacketSize", Test_Verify_PacketSize, "PacketSize {TestName}", NULL },

    { NULL, NULL, NULL, NULL }
};

static void Test_DisplayConfig(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_Info         (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_ResetConfig  (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_Run          (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_SetBandwidth (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_SetBufferQty (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_SetCode      (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_SetMode      (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_SetPacketSize(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_SetProfiling (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Test_StartStop    (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo TEST_COMMANDS[] =
{
    { "DisplayConfig", Test_DisplayConfig, "DisplayConfig"                                            , NULL },
    { "Info"         , Test_Info         , "Info {TestName}"                                          , NULL },
    { "ResetCongif"  , Test_ResetConfig  , "ResetConfig"                                              , NULL },
	{ "Run"          , Test_Run          , "Run {TestName}"                                           , NULL },
    { "Search"       , NULL              , "Search ..."                                               , TEST_SEARCH_COMMANDS },
    { "SetBandwidth" , Test_SetBandwidth , "SetBandwidth {Bandwidth_MiB/s}"                           , NULL },
    { "SetBufferQty" , Test_SetBufferQty , "SetBufferQty {BufferQty}"                                 , NULL },
    { "SetCode"      , Test_SetCode      , "SetCode DEFAULT|FORWARD|NONE|NOTHING|REPLY|REPLY_ON_ERROR", NULL },
    { "SetMode"      , Test_SetMode      , "SetMode DEFAULT|FUNCTION|KERNEL"                          , NULL },
    { "SetPacketSize", Test_SetPacketSize, "SetPacketSize {PacketSize_byte}"                          , NULL },
    { "SetProfiling" , Test_SetProfiling , "SetProfiling false|true"                                  , NULL },
    { "Verify"       , NULL              , "Verify {TestName}"                                        , TEST_VERIFY_COMMANDS },
    { "StartStop"    , Test_StartStop    , "StatStop {TestName}"                                      , NULL },

	{ NULL, NULL, NULL, NULL }
};

static void Display(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Start  (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Stop   (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo COMMANDS[] =
{
	{ "Adapter"      , NULL                           , "Adapter ..."                                             , ADAPTER_COMMANDS   },
    { "Display"      , Display                        , "Display                       Display system information", NULL               },
	{ "ExecuteScript", KmsLib::ToolBase::ExecuteScript, "ExecuteSript {Script}         Execute script"            , NULL               },
	{ "Exit"         , KmsLib::ToolBase::Exit         , "Exit                          Exit"                      , NULL               },
    { "Kernel"       , NULL                           , "Kernel ..."                                              , KERNEL_COMMANDS    },
	{ "Processor"    , NULL                           , "Processor ..."                                           , PROCESSOR_COMMANDS },
    { "Start"        , Start                          , "Start                         Start the system"          , NULL               },
    { "Stop"         , Stop                           , "Stop                          Stop the system"           , NULL               },
    { "Test"         , NULL                           , "Test ..."                                                , TEST_COMMANDS      },

	{ NULL, NULL, NULL, NULL }
};

// Static functions declaration
/////////////////////////////////////////////////////////////////////////////

static void ReportStatus(OpenNet::Status aStatus, const char * aMsgOK);

static void System_Connect();

// Global variable
/////////////////////////////////////////////////////////////////////////////

typedef std::map<std::string, OpenNet::Kernel *> KernelMap;

static OpenNet::Adapter   * sAdapter   = NULL;
static OpenNet::Kernel    * sKernel    = NULL;
static KernelMap            sKernels;
static OpenNet::Processor * sProcessor = NULL;
static OpenNet::System    * sSystem;

static TestLib::TestFactory sTestFactory;

// Entry point
/////////////////////////////////////////////////////////////////////////////

int main(int aCount, const char ** aVector)
{
    KMS_TOOL_BANNER("OpenNet", "OpenNet_Tool", VERSION_STR, VERSION_TYPE);

    System_Connect();

	KmsLib::ToolBase lToolBase(COMMANDS);

	if (!lToolBase.ParseArguments(aCount, aVector))
	{
		lToolBase.ParseCommands();
	}

	sSystem->Delete();

    return 0;
}

// Commands
/////////////////////////////////////////////////////////////////////////////

void Adapter_Display(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Adapter Display %s\n", aArg);
    printf("Adapter Display\n");

    if (NULL == sAdapter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
        return;
    }

    OpenNet::Status lStatus = sAdapter->Display(stdout);
    assert(OpenNet::STATUS_OK == lStatus);
}

void Adapter_GetConfig(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Adapter GetConfig %s\n", aArg);
    printf("Adapter GetConfig\n");

    if (NULL == sAdapter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
        return;
    }

    OpenNet::Adapter::Config lConfig;

    OpenNet::Status lStatus = sAdapter->GetConfig(&lConfig);
    assert(OpenNet::STATUS_OK == lStatus);

    OpenNet::Adapter::Display(lConfig, stdout);
}

void Adapter_GetState(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Adapter GetState %s\n", aArg);
    printf("Adapter GetState\n");

    if (NULL == sAdapter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
        return;
    }

    OpenNet::Adapter::State lState;

    OpenNet::Status lStatus = sAdapter->GetState(&lState);
    assert(OpenNet::STATUS_OK == lStatus);

    OpenNet::Adapter::Display(lState, stdout);
}

void Adapter_GetStats(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Adapter GetStats %s\n", aArg);

    if (NULL == sAdapter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
        return;
    }

    bool lReset = (0 == strcmp("true", aArg));

    // TODO  OpenNet_Tool
    //       Low (Feature) - Ajouter un argument pour le MinLevel

    printf("Adapter GetStats %s\n", lReset ? "true" : "false");

    unsigned int lInfo_byte;
    unsigned int lStats[1024];

    OpenNet::Status lStatus = sAdapter->GetStatistics(lStats, sizeof(lStats), &lInfo_byte, lReset);
    assert(OpenNet::STATUS_OK == lStatus);

    lStatus = sAdapter->DisplayStatistics(lStats, lInfo_byte, stdout);
    assert(OpenNet::STATUS_OK == lStatus);
}

void Adapter_List(KmsLib::ToolBase * aToolBase, const char * aArg)
{
	assert(NULL != aArg);

	printf("Adapter List %s\n", aArg);
	printf("Adapter List\n");

	unsigned int lCount = sSystem->Adapter_GetCount();
	for (unsigned int i = 0; i < lCount; i++)
	{
		printf("Adapter %u of %u : %s\n", i, lCount, sSystem->Adapter_Get(i)->GetName());
	}
}

void Adapter_Select(KmsLib::ToolBase * aToolBase, const char * aArg)
{
	assert(NULL != aArg);

	printf("Adapter Select %s\n", aArg);

	unsigned int lIndex;

	switch (sscanf_s(aArg, "%u", &lIndex))
	{
	case 1 :
		printf("Adapter Select %u\n", lIndex);

		unsigned int lCount;
		
		lCount = sSystem->Adapter_GetCount();
		if (lCount <= lIndex)
		{
			KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid index");
            return;
		}

        sAdapter = sSystem->Adapter_Get(lIndex);
		assert(NULL != sAdapter);

		KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Adapter selected");
        break;

	default :
		KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
	}
}

void Display(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Display %s\n", aArg);
    printf("Display\n");

    OpenNet::Status lStatus = sSystem->Display(stdout);
    assert(OpenNet::STATUS_OK == lStatus);
}

void Kernel_Forward_AddDestination(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Kernel_Forward AddDestination %s\n", aArg);

    OpenNet::Adapter * lAdapter;
    unsigned int       lAdapterIndex;

    switch (sscanf_s(aArg, "%u", &lAdapterIndex))
    {
    case EOF:
        printf("Kernel_Forward AddDestination\n");
        if (NULL == sAdapter)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
            return;
        }
        lAdapter = sAdapter;
        break;

    case 1:
        printf("Kernel_Forward AddDestination %u\n", lAdapterIndex);

        if (sSystem->Adapter_GetCount() >= lAdapterIndex)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid adapter index");
            return;
        }

        lAdapter = sSystem->Adapter_Get(lAdapterIndex);
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

    printf("Kernel_Forward Create %s\n", aArg);

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

    printf("Kernel_Forward List %s\n", aArg);
    printf("Kernel_Forward List\n");

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

    printf("Kernel_Forward RemoveDestination %s\n", aArg);

    OpenNet::Adapter * lAdapter;
    unsigned int       lAdapterIndex;

    switch (sscanf_s(aArg, "%u", &lAdapterIndex))
    {
    case EOF:
        printf("Kernel_Forward RemoveDestination\n");
        if (NULL == sAdapter)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
            return;
        }
        lAdapter = sAdapter;
        break;

    case 1:
        printf("Kernel_Forward RemoveDestination %u\n", lAdapterIndex);

        if (sSystem->Adapter_GetCount() >= lAdapterIndex)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid adapter index");
            return;
        }

        lAdapter = sSystem->Adapter_Get(lAdapterIndex);
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

    printf("Kernel_Forward ResetDestination %s\n", aArg);
    printf("Kernel_Forward ResetDestination\n");

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

    printf("Kernel Create %s\n", aArg);

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

    printf("Kernel Delete %s\n", aArg);
    printf("Kernel Delete\n");

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

    printf("Kernel Display %s\n", aArg);
    printf("Kernel Display\n");

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

    printf("Kernel Edit_Remove %s\n", aArg);

    char lRemove[64];

    switch (sscanf_s(aArg, "%s", lRemove SIZE_INFO(static_cast<unsigned int>(sizeof(lRemove)))))
    {
    case 1:
        printf("Kernel Edit_Remove %s\n", lRemove);

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

    printf("Kernel Edit_Search %s\n", aArg);

    char lSearch[64];

    switch (sscanf_s(aArg, "%s", lSearch SIZE_INFO(static_cast<unsigned int>(sizeof(lSearch)))))
    {
    case 1:
        printf("Kernel Edit_Search %s\n", lSearch);

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

    printf("Kernel Edit_Search %s\n", aArg);

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

    printf("Kernel GetCodeSize %s\n", aArg);
    printf("Kernel GetCodeSize\n");

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

    printf("Kernel List %s\n", aArg);
    printf("Kernel List\n");

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

    printf("Kernel GetCodeSize %s\n", aArg);
    printf("Kernel GetCodeSize\n");

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

    printf("Kernel Select %s\n", aArg);

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

    printf("Kernel SetCode %s\n", aArg);

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

        ReportStatus(sKernel->SetCode(lFileName, 1 ), "Code set");
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Processor_Display(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Processor Display %s\n", aArg);
    printf("Processor Display\n");

    if (NULL == sProcessor)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No processor selected");
        return;
    }

    OpenNet::Status lStatus = sProcessor->Display(stdout);
    assert(OpenNet::STATUS_OK == lStatus);
}

void Processor_List(KmsLib::ToolBase * aToolBase, const char * aArg)
{
	assert(NULL != aArg);

	printf("Processor List %s\n", aArg);
	printf("Processor List\n");

	unsigned int lCount = sSystem->Processor_GetCount();
	for (unsigned int i = 0; i < lCount; i++)
	{
		printf("Processor %u of %u : %s\n", i, lCount, sSystem->Processor_Get(i)->GetName());
	}

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Processor listed");
}

void Processor_Select(KmsLib::ToolBase * aToolBase, const char * aArg)
{
	assert(NULL != aArg);

	printf("Processor Select %s", aArg);

	unsigned int lIndex;

	switch (sscanf_s(aArg, "%u", &lIndex))
	{
	case 1:
		printf("Processor Select %u\n", lIndex);

		unsigned int lCount;

		lCount = sSystem->Processor_GetCount();
		if (lCount <= lIndex)
		{
		    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid index");
            return;
		}

        sProcessor = sSystem->Processor_Get(lIndex);
		assert(NULL != sProcessor );

		KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Processor selected");
		break;

	default:
		KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
	}
}

void Test_DisplayConfig(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test DisplayConfig %s\n", aArg);
    printf("Test DisplayConfig\n");

    sTestFactory.DisplayConfig();
}

void Test_Info(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test Info %s\n", aArg);

    TestLib::Test * lTest = sTestFactory.CreateTest(aArg);
    if (NULL != lTest)
    {
        lTest->Info_Display();

        delete lTest;
    }
}

void Test_ResetConfig(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test ResetConfig %s\n", aArg);
    printf("Test ResetConfig\n");

    sTestFactory.ResetConfig();

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Config reset");
}

void Test_Run(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test Run %s\n", aArg);

    TestLib::Test * lTest = sTestFactory.CreateTest(aArg);
    if (NULL != lTest)
    {
        unsigned int lRet = lTest->Run();
        if (0 == lRet)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PASSED");
        }

        delete lTest;
    }
}

void Test_Search_Bandwidth(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test Search Bandwidth %s\n", aArg);

    TestLib::Test * lTest = sTestFactory.CreateTest(aArg);
    if (NULL != lTest)
    {
        unsigned int lRet = lTest->Search_Bandwidth();
        if (0 == lRet)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PASSED");
        }

        delete lTest;
    }
}

void Test_Search_BufferQty(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test Search BufferQty %s\n", aArg);

    TestLib::Test * lTest = sTestFactory.CreateTest(aArg);
    if (NULL != lTest)
    {
        unsigned int lRet = lTest->Search_BufferQty();
        if (0 == lRet)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PASSED");
        }

        delete lTest;
    }
}

void Test_Search_PacketSize(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test Search PacketSize %s\n", aArg);

    TestLib::Test * lTest = sTestFactory.CreateTest(aArg);
    if (NULL != lTest)
    {
        unsigned int lRet = lTest->Search_PacketSize();
        if (0 == lRet)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PASSED");
        }

        delete lTest;
    }
}

void Test_SetBandwidth(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test SetBandwidth %s\n", aArg);

    unsigned int lRet = sTestFactory.SetBandwidth(aArg);
    if (0 == lRet)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Bandwidth set");
    }
}

void Test_SetBufferQty(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test SetBufferQty %s\n", aArg);

    unsigned int lRet = sTestFactory.SetBufferQty(aArg);
    if (0 == lRet)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Buffer quantity set");
    }
}

void Test_SetCode(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test SetCode %s\n", aArg);

    unsigned int lRet = sTestFactory.SetCode(aArg);
    if (0 == lRet)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Test code set");
    }
}

void Test_SetMode(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test SetMode %s\n", aArg);

    unsigned int lRet = sTestFactory.SetMode(aArg);
    if (0 == lRet)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Test mode set");
    }
}

void Test_SetPacketSize(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test SetPacketSize %s\n", aArg);

    unsigned int lRet = sTestFactory.SetPacketSize(aArg);
    if (0 == lRet)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Packet size set");
    }
}

void Test_SetProfiling(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test SetProfiling %s\n", aArg);

    unsigned int lRet = sTestFactory.SetProfiling(aArg);
    if (0 == lRet)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Test profiling set");
    }
}

void Test_StartStop( KmsLib::ToolBase * aToolBase, const char * aArg )
{
    assert(NULL != aArg);

    printf("Test StartStup %s\n", aArg);

    TestLib::Test * lTest = sTestFactory.CreateTest(aArg);
    if (NULL != lTest)
    {
        unsigned int lRet = lTest->StartStop();
        if (0 == lRet)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PASSED");
        }

        delete lTest;
    }

}

void Test_Verify_Bandwidth(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test Verify Bandwidth %s\n", aArg);

    TestLib::Test * lTest = sTestFactory.CreateTest(aArg);
    if (NULL == lTest)
    {
        unsigned int lRet = lTest->Verify_Bandwidth();
        if (0 == lRet)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PASSED");
        }

        delete lTest;
    }
}

void Test_Verify_BufferQty(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test Verify BufferQty %s\n", aArg);

    TestLib::Test * lTest = sTestFactory.CreateTest(aArg);
    if (NULL == lTest)
    {
        unsigned int lRet = lTest->Verify_BufferQty();
        if (0 == lRet)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PASSED");
        }

        delete lTest;
    }
}

void Test_Verify_PacketSize(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test Verify PacketSize %s\n", aArg);

    TestLib::Test * lTest = sTestFactory.CreateTest(aArg);
    if (NULL == lTest)
    {
        unsigned int lRet = lTest->Verify_PacketSize();
        if (0 == lRet)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PASSED");
        }

        delete lTest;
    }
}

void Stop(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Stop %s\n", aArg);
    printf("Stop\n");

    OpenNet::Status lStatus = sSystem->Stop();
    if (OpenNet::STATUS_OK != lStatus)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "System::Stop failed");
        OpenNet::Status_Display(lStatus, stdout);
        return;
    }

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "System stopped");
}

void Start(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Start %s\n", aArg);
    printf("Start\n");

    OpenNet::Status lStatus = sSystem->Start(0);
    if (OpenNet::STATUS_OK != lStatus)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "System::Start failed");
        OpenNet::Status_Display(lStatus, stdout);
        return;
    }

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

    sSystem = OpenNet::System::Create();
    if (NULL == sSystem)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_FATAL_ERROR, "OpenNet::System::Create() failed");
        exit(__LINE__);
    }
}

/*
void Test(char aTest, KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test %c %s\n", aTest, aArg);

    double       lBandwidth_MiB_s;
    unsigned int lBufferQty;
    unsigned int lPacketSize_byte;

    switch (sscanf_s(aArg, "%u %u %lf", &lBufferQty, &lPacketSize_byte, &lBandwidth_MiB_s))
    {
    case EOF:
        lBufferQty = 1;
        // no break;

    case 1:
        lPacketSize_byte = 1024;
        // no break;

    case 2:
        printf("Test %c %u %u\n", aTest, lBufferQty, lPacketSize_byte);

        if ((0 >= lBufferQty) || (OPEN_NET_BUFFER_QTY < lBufferQty))
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid buffer quantity");
            return;
        }

        try
        {
            Test(aTest, lBufferQty, lPacketSize_byte);
        }
        catch (KmsLib::Exception * eE)
        {
            eE->Write(stdout);
        }
        break;

    case 3:
        printf("Test %c %u %u %f\n", aTest, lBufferQty, lPacketSize_byte, lBandwidth_MiB_s);

        if ((0 >= lBufferQty) || (OPEN_NET_BUFFER_QTY < lBufferQty))
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid buffer quantity");
            return;
        }

        if ((0.0 >= lBandwidth_MiB_s) || (120.0 < lBandwidth_MiB_s))
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid bandwidth");
            return;
        }

        try
        {
            Test(aTest, lBufferQty, lPacketSize_byte, lBandwidth_MiB_s);
        }
        catch (KmsLib::Exception * eE)
        {
            eE->Write(stdout);
        }
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}
*/
