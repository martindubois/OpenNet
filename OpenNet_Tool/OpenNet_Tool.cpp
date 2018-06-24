
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
#include <OpenNet/System.h>

// ===== Common =============================================================
#include "../Common/Version.h"

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
    { "GetStats" , Adapter_GetStats , "GetStats                      Display statistics"   , NULL },
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

static void SendPackets(OpenNet::Adapter * aA0, OpenNet::Adapter * aA1, unsigned int aPacketSize_byte, unsigned int aPacketQty);

static void System_Connect();
static void System_Reset  ();

static void Test_Loop(unsigned int aBufferQty, unsigned int aPacketSize_byte, unsigned int aPacketQty);

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
    printf("Adapter GetStats\n");

    if (NULL == sAdapter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_USER_ERROR, "No adapter selected");
        return;
    }

    OpenNet::Adapter::Stats lStats;

    OpenNet::Status lStatus = sAdapter->GetStats(&lStats);
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

        Test_Loop(lBufferQty, lPacketSize_byte, lPacketQty);
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

void SendPackets(OpenNet::Adapter * aA0, OpenNet::Adapter * aA1, unsigned int aPacketSize_byte, unsigned int aPacketQty)
{
    assert(NULL != aA0             );
    assert(NULL != aA1             );
    assert(0    <  aPacketSize_byte);
    assert(0    <  aPacketQty      );

    unsigned char * lPacket = new unsigned char[aPacketSize_byte];
    assert(NULL != lPacket);

    memset(lPacket, 0xff, aPacketSize_byte);

    for (unsigned int i = 0; i < aPacketQty; i++)
    {
        OpenNet::Status lS0 = aA0->Packet_Send(lPacket, aPacketSize_byte);
        OpenNet::Status lS1 = aA1->Packet_Send(lPacket, aPacketSize_byte);
        if ((OpenNet::STATUS_OK != lS0) || (OpenNet::STATUS_OK != lS1))
        {
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Adapter::Packet_Send( ,  ) failed");
            break;
        }
    }

    delete lPacket;
}

void System_Connect()
{
    sSystem = OpenNet::System::Create();
    if (NULL == sSystem)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_FATAL_ERROR, "OpenNet::System::Create() failed");
        exit(__LINE__);
    }
}

void System_Reset()
{
    sAdapter   = NULL;
    sProcessor = NULL;

    sSystem->Delete();

    System_Connect();
}

void Test_Loop(unsigned int aBufferQty, unsigned int aPacketSize_byte, unsigned int aPacketQty)
{
    OpenNet::Filter_Forward lFF0;
    OpenNet::Filter_Forward lFF1;

    printf("Retrieving adapters...\n");
    OpenNet::Adapter * lA0 = sSystem->Adapter_Get(0);
    OpenNet::Adapter * lA1 = sSystem->Adapter_Get(1);
    if ((NULL == lA0) || (NULL == lA1))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "This test need 2 adapters");
        return;
    }

    printf("Retriving processor...\n");
    OpenNet::Processor * lP0 = sSystem->Processor_Get(0);
    if (NULL == lP0)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "This test need 1 processor");
        return;
    }

    printf("Connecting adapters...\n");
    OpenNet::Status lS0 = sSystem->Adapter_Connect(lA0);
    OpenNet::Status lS1 = sSystem->Adapter_Connect(lA1);
    if ((OpenNet::STATUS_OK != lS0) || (OpenNet::STATUS_OK != lS1))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "System::Adapter_Connect(  ) failed");
        goto Error3;
    }

    printf("Setting processor...\n");
    lS0 = lA0->SetProcessor(lP0);
    lS1 = lA1->SetProcessor(lP0);
    if ((OpenNet::STATUS_OK != lS0) || (OpenNet::STATUS_OK != lS1))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Adapter::SetProcessor(  ) failed");
        goto Error3;
    }

    printf("Adding destination...\n");
    lS0 = lFF0.AddDestination(lA1);
    lS1 = lFF1.AddDestination(lA0);
    if ((OpenNet::STATUS_OK != lS0) || (OpenNet::STATUS_OK != lS1))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Filter::AddDestination(  ) failed");
        goto Error3;
    }

    printf("Setting input filter...\n");
    lS0 = lA0->SetInputFilter(&lFF0);
    lS1 = lA1->SetInputFilter(&lFF1);
    if ((OpenNet::STATUS_OK != lS0) || (OpenNet::STATUS_OK != lS1))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Adapter::SetInputFilter(  ) failed");
        goto Error3;
    }

    printf("Allocating buffers...\n");
    lS0 = lA0->Buffer_Allocate(aBufferQty);
    lS1 = lA1->Buffer_Allocate(aBufferQty);
    if ((OpenNet::STATUS_OK != lS0) || (OpenNet::STATUS_OK != lS1))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Adapter::Buffer_Allocate(  ) failed");
        goto Error1;
    }

    printf("Starting...\n");
    OpenNet::Status lS = sSystem->Start();
    if (OpenNet::STATUS_OK != lS)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "System::Start(  ) failed");
        goto Error0;
    }

    Sleep(2000);

    printf("Sending packets...\n");
    SendPackets(lA0, lA1, aPacketSize_byte, aPacketQty);

    printf("Stabilizing...\n");
    Sleep(2000);

    printf("Reseting statistics...\n");
    lS0 = lA0->ResetStats();
    lS1 = lA1->ResetStats();
    if ((OpenNet::STATUS_OK != lS0) || (OpenNet::STATUS_OK != lS1))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Adapter::Reset_State() failed");
    }

    printf("Running...\n");
    Sleep(10000);

    printf("Stopping...\n");
    lS = sSystem->Stop();
    if (OpenNet::STATUS_OK != lS)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "System::Stop(  ) failed");
    }

Error0:
    printf("Releasing buffers...\n");
    lS0 = lA0->Buffer_Release(aBufferQty);
    lS1 = lA1->Buffer_Release(aBufferQty);
    if ((OpenNet::STATUS_OK != lS0) || (OpenNet::STATUS_OK != lS1))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Adapter::Buffer_Release(  ) failed");
    }

Error1:
    printf("Reseting input filter...\n");
    lS0 = lA0->ResetInputFilter();
    lS1 = lA1->ResetInputFilter();
    if ((OpenNet::STATUS_OK != lS0) || (OpenNet::STATUS_OK != lS1))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Adapter::ResetInputFilter(  ) failed");
    }

    OpenNet::Adapter::Stats lStats0;
    OpenNet::Adapter::Stats lStats1;

    printf("Retrieving statistics...\n");
    lS0 = lA0->GetStats(&lStats0);
    lS1 = lA1->GetStats(&lStats1);
    if ((OpenNet::STATUS_OK != lS0) || (OpenNet::STATUS_OK != lS1))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Adapter::GetStats(  ) failed");
    }
    else
    {
        double lPackets = static_cast<double>(lStats0.mDriver.mHardware.mRx_Packet + lStats1.mDriver.mHardware.mRx_Packet);

        printf("%f packets/s\n", lPackets / 10.0);

        double lBytes = lPackets * aPacketSize_byte;

        printf("%f bytes/s\n", lPackets / 10.0);
        printf("%f KB/s\n"   , lBytes / 10.0 / 1024.0);
        printf("%f MB/s\n"   , lBytes / 10.0 / 1024.0 / 1024.0);
        printf("%f GB/s\n"   , lBytes / 10.0 / 1024.0 / 1024.0 / 1024.0);
    }

Error3:
    printf("Resting system...\n");
    System_Reset();
}
