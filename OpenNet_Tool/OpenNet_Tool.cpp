
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/OpenNet_Tool.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdio.h>

// ===== C++ ================================================================
#include <map>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Import/Includes ====================================================
#include <KmsLib/ToolBase.h>
#include <KmsTool.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Filter_Forward.h>

// ===== Common =============================================================
#include "../Common/Version.h"

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

static void Filter_Forward_AddDestination   (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_Forward_Create           (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_Forward_List             (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_Forward_RemoveDestination(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_Forward_ResetDestinations(KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo FILTER_FORWARD_COMMANDS[] =
{
    { "AddDestination"   , Filter_Forward_AddDestination   , "AddDestination [Adapter]      Add a destination"               , NULL },
    { "Create"           , Filter_Forward_Create           , "Create                        Create a Filter_Forward instance", NULL },
    { "List"             , Filter_Forward_List             , "List                          List the Filter_Forward instance", NULL },
    { "RemoveDestination", Filter_Forward_RemoveDestination, "RemoveDestination [Adapter]   Remove a destination"            , NULL },
    { "ResetDestinations", Filter_Forward_ResetDestinations, "RestedDestinations            Remove all destiations"          , NULL },

    { NULL, NULL, NULL, NULL }
};

static void Filter_Create      (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_Delete      (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_Display     (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_Edit_Remove (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_Edit_Replace(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_Edit_Search (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_GetCodeSize (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_List        (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_ResetCode   (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_Select      (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Filter_SetCode     (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo FILTER_COMMANDS[] =
{
    { "Create"      , Filter_Create      , "Create {Name}                 Create a Filter instance"  , NULL },
    { "Delete"      , Filter_Delete      , "Delete                        Delete the filter instance", NULL },
    { "Display"     , Filter_Display     , "Display                       Display the filter"        , NULL },
    { "Edit_Remove" , Filter_Edit_Remove , "Edit_Remote {Remove}          Remove strings from code"  , NULL },
    { "Edit_Replace", Filter_Edit_Replace, "Edit_Replace {Search} [Rep]   Search and replace strings", NULL },
    { "Edit_Search" , Filter_Edit_Search , "Edit_Search {Search}          Search a string"           , NULL },
    { "Forward"     , NULL               , "Forward ..."                                             , FILTER_FORWARD_COMMANDS },
    { "List"        , Filter_List        , "List                          List the Filter instances" , NULL },
    { "GetCodeSize" , Filter_GetCodeSize , "GetCodeSize                   Display code size"         , NULL },
    { "ResetCode"   , Filter_ResetCode   , "ResetCode                     Reset the code"            , NULL },
    { "Select"      , Filter_Select      , "Select {Name}                 Select a filter"           , NULL },
    { "SetCode"     , Filter_SetCode     , "SetCode {FileName}            Set the code"              , NULL },

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

static void Test_Loop(KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo TEST_COMMANDS[] =
{
	{ "Loop", Test_Loop, "Loop {BQ} {PS} {PQ}           Display the processor", NULL },

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
    { "Filter"       , NULL                           , "Filter ..."                                              , FILTER_COMMANDS    },
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
static void System_Reset  ();

// Global variable
/////////////////////////////////////////////////////////////////////////////

typedef std::map<std::string, OpenNet::Filter *> FilterMap;

static OpenNet::Adapter   * sAdapter   = NULL;
static OpenNet::Filter    * sFilter    = NULL;
static FilterMap            sFilters;
static OpenNet::Processor * sProcessor = NULL;
static OpenNet::System    * sSystem;

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

    printf("Adapter GetStats %s\n", lReset ? "true" : "false");

    OpenNet::Adapter::Stats lStats;

    OpenNet::Status lStatus = sAdapter->GetStats(&lStats, lReset);
    assert(OpenNet::STATUS_OK == lStatus);

    OpenNet::Adapter::Display(lStats, stdout);
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

void Filter_Forward_AddDestination(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter_Forward AddDestination %s\n", aArg);

    OpenNet::Adapter * lAdapter;
    unsigned int       lAdapterIndex;

    switch (sscanf_s(aArg, "%u", &lAdapterIndex))
    {
    case EOF:
        printf("Filter_Forward AddDestination\n");
        if (NULL == sAdapter)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
            return;
        }
        lAdapter = sAdapter;
        break;

    case 1:
        printf("Filter_Forward AddDestination %u\n", lAdapterIndex);

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

    if (NULL == sFilter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
        return;
    }

    OpenNet::Filter_Forward * lFF = dynamic_cast<OpenNet::Filter_Forward *>(sFilter);
    if (NULL == lFF)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "The selected filter is not a Filter_Forward");
        return;
    }
    
    ReportStatus(lFF->AddDestination(lAdapter), "Destination added");
}

void Filter_Forward_Create(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter_Forward Create %s\n", aArg);

    char lName[64];

    switch (sscanf_s(aArg, "%s", lName, static_cast<unsigned int>(sizeof(lName))))
    {
    case 1:
        printf("Filter_Forward Create %s\n", lName);

        if (sFilters.end() != sFilters.find(lName))
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "The name already exist");
            return;
        }

        sFilter = new OpenNet::Filter_Forward();

        sFilter->SetName(lName);

        sFilters.insert(FilterMap::value_type(lName, sFilter));

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Filter_Forward created");
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Filter_Forward_List(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter_Forward List %s\n", aArg);
    printf("Filter_Forward List\n");

    for (FilterMap::iterator lIt = sFilters.begin(); lIt != sFilters.end(); lIt++)
    {
        assert(NULL != lIt->second);

        OpenNet::Filter_Forward * lFF = dynamic_cast<OpenNet::Filter_Forward *>(lIt->second);
        if (NULL != lFF)
        {
            printf("Filter_Forward : %s\n", lIt->first.c_str());

			lFF->Display(stdout);
        }
    }

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Filter_Forward listed");
}

void Filter_Forward_RemoveDestination(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter_Forward RemoveDestination %s\n", aArg);

    OpenNet::Adapter * lAdapter;
    unsigned int       lAdapterIndex;

    switch (sscanf_s(aArg, "%u", &lAdapterIndex))
    {
    case EOF:
        printf("Filter_Forward RemoveDestination\n");
        if (NULL == sAdapter)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
            return;
        }
        lAdapter = sAdapter;
        break;

    case 1:
        printf("Filter_Forward RemoveDestination %u\n", lAdapterIndex);

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

    if (NULL == sFilter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
        return;
    }

    OpenNet::Filter_Forward * lFF = dynamic_cast<OpenNet::Filter_Forward *>(sFilter);
    if (NULL == lFF)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "The selected filter is not a Filter_Forward");
        return;
    }

    ReportStatus(lFF->RemoveDestination(lAdapter), "Destination removed");
}

void Filter_Forward_ResetDestinations(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter_Forward ResetDestination %s\n", aArg);
    printf("Filter_Forward ResetDestination\n");

    if (NULL == sFilter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
        return;
    }

    OpenNet::Filter_Forward * lFF = dynamic_cast<OpenNet::Filter_Forward *>(sFilter);
    if (NULL == lFF)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "The selected filter is not a Filter_Forward");
        return;
    }

    ReportStatus(lFF->ResetDestinations(), "Destination reset");
}

void Filter_Create(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter Create %s\n", aArg);

    char lName[64];

    switch (sscanf_s(aArg, "%s", lName, static_cast<unsigned int>(sizeof(lName))))
    {
    case 1:
        printf("Filter Create %s\n", lName);

        if (sFilters.end() != sFilters.find(lName))
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "The name already exist");
            return;
        }

        sFilter = new OpenNet::Filter();

        sFilter->SetName(lName);

        sFilters.insert(FilterMap::value_type(lName, sFilter));

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Filter created");
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Filter_Delete(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter Delete %s\n", aArg);
    printf("Filter Delete\n");

    if (NULL == sFilter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
        return;
    }

    delete sFilter;

    for (FilterMap::iterator lIt = sFilters.begin(); lIt != sFilters.end(); lIt++)
    {
        assert(NULL != lIt->second);

        if (sFilter == lIt->second)
        {
            sFilters.erase(lIt);
            break;
        }
    }

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Filter deleted");
}

void Filter_Display(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter Display %s\n", aArg);
    printf("Filter Display\n");

    if (NULL == sFilter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
        return;
    }

    OpenNet::Status lStatus = sFilter->Display(stdout);
    assert(OpenNet::STATUS_OK == lStatus);
}

void Filter_Edit_Remove(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter Edit_Remove %s\n", aArg);

    char lRemove[64];

    switch (sscanf_s(aArg, "%s", lRemove, static_cast<unsigned int>(sizeof(lRemove))))
    {
    case 1:
        printf("Filter Edit_Remove %s\n", lRemove);

        if (NULL == sFilter)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
            return;
        }

        char lMsg[64];

        sprintf_s(lMsg, "%u occurence removed", sFilter->Edit_Remove(lRemove));

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, lMsg);
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Filter_Edit_Search(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter Edit_Search %s\n", aArg);

    char lSearch[64];

    switch (sscanf_s(aArg, "%s", lSearch, static_cast<unsigned int>(sizeof(lSearch))))
    {
    case 1:
        printf("Filter Edit_Search %s\n", lSearch);

        if (NULL == sFilter)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
            return;
        }

        char lMsg[64];

        sprintf_s(lMsg, "%u occurence found", sFilter->Edit_Search(lSearch));

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, lMsg);
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Filter_Edit_Replace(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter Edit_Search %s\n", aArg);

    char lSearch [64];
    char lReplace[64];

    switch (sscanf_s(aArg, "%s %s", lSearch, static_cast<unsigned int>(sizeof(lSearch)), lReplace, static_cast<unsigned int>(sizeof(lReplace))))
    {
    case 1 :
        strcpy_s(lReplace, "");
        // no break;

    case 2:
        printf("Filter Edit_Search %s %s\n", lSearch, lReplace);

        if (NULL == sFilter)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
            return;
        }

        char lMsg[64];

        sprintf_s(lMsg, "%u occurence replaced", sFilter->Edit_Replace(lSearch, lReplace));

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, lMsg);
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Filter_GetCodeSize(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter GetCodeSize %s\n", aArg);
    printf("Filter GetCodeSize\n");

    if (NULL == sFilter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
        return;
    }
 
    unsigned int lCodeSize_byte = sFilter->GetCodeSize();

    char lMsg[64];

    sprintf_s(lMsg, "%u bytes", lCodeSize_byte);

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, lMsg);
}

void Filter_List(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter List %s\n", aArg);
    printf("Filter List\n");

    for (FilterMap::iterator lIt = sFilters.begin(); lIt != sFilters.end(); lIt++)
    {
        assert(NULL != lIt->second);

        printf("Filter : %s\n", lIt->first.c_str());

        lIt->second->Display(stdout);
    }

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Filter listed");
}

void Filter_ResetCode(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter GetCodeSize %s\n", aArg);
    printf("Filter GetCodeSize\n");

    if (NULL == sFilter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
        return;
    }
 
    ReportStatus(sFilter->ResetCode(), "Code reset");
}

void Filter_Select(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter Select %s\n", aArg);

    FilterMap::iterator lIt;

    char lName[64];

    switch (sscanf_s(aArg, "%s", lName, static_cast<unsigned int>(sizeof(lName))))
    {
    case 1:
        printf("Filter Select %s\n", lName);

        lIt = sFilters.find(lName);

        if (sFilters.end() == lIt)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid name");
            return;
        }

        assert(NULL != lIt->second);

        sFilter = lIt->second;

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Filter selected");
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Filter_SetCode(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Filter SetCode %s\n", aArg);

    FilterMap::iterator lIt;

    char lFileName[64];

    switch (sscanf_s(aArg, "%s", lFileName, static_cast<unsigned int>(sizeof(lFileName))))
    {
    case 1:
        printf("Filter Select %s\n", lFileName);

        if (NULL == sFilter)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No filter selected");
            return;
        }

        ReportStatus(sFilter->SetCode(lFileName), "Code set");
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
		assert(NULL != sAdapter);

		KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Processor selected");
		break;

	default:
		KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
	}
}

void Test_Loop(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Test Loop %s\n", aArg);

    unsigned int lBufferQty      ;
    unsigned int lPacketSize_byte;
    unsigned int lPacketQty      ;

    switch (sscanf_s(aArg, "%u %u %u", &lBufferQty, &lPacketSize_byte, &lPacketQty))
    {
    case EOF:
        lBufferQty = 1;
        // no break;

    case 1:
        lPacketSize_byte = 1024;
        // no break;

    case 2:
        lPacketQty = 128;
        // No break

    case 3:
        printf("Test Loop %u %u %u\n", lBufferQty, lPacketSize_byte, lPacketQty);

        if (0 >= lBufferQty)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid buffer quantity");
            return;
        }

        if (0 >= lPacketQty)
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid packet quantity");
            return;
        }

        try
        {
            Test_Loop(sSystem, lBufferQty, lPacketSize_byte, lPacketQty);
        }
        catch (KmsLib::Exception * eE)
        {
            eE->Write(stdout);
        }
        System_Reset();
        break;

    default:
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "Invalid argument");
    }
}

void Stop(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    printf("Stop %s\n", aArg);
    printf("Stop\n");

    OpenNet::Status lStatus = sSystem->Stop(0);
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

    OpenNet::Status lStatus = sSystem->Start();
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

void System_Reset()
{
    printf("Reseting the System...\n");

    sAdapter   = NULL;
    sProcessor = NULL;

    sSystem->Delete();

    System_Connect();
}

// TODO  OpenNet_Tool  Use sub function and exception

// TODO  OpenNet_Tool  Move test function (other than direct command handler)
//                     into Test.c and .h
