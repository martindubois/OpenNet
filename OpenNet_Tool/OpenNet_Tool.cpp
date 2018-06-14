
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

static void Adapter_GetConfig(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Adapter_GetState (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Adapter_GetStats (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Adapter_List     (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Adapter_Select   (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo ADAPTER_COMMANDS[] =
{
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

static void Processor_List  (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Processor_Select(KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo PROCESSOR_COMMANDS[] =
{
	{ "List"  , Processor_List  , "List                          List the processors", NULL },
	{ "Select", Processor_Select, "Select {Index}                Select an processor", NULL },

	{ NULL, NULL, NULL, NULL }
};

static const KmsLib::ToolBase::CommandInfo COMMANDS[] =
{
	{ "Adapter"      , NULL                           , "Adapter ..."                                 , ADAPTER_COMMANDS   },
	{ "ExecuteScript", KmsLib::ToolBase::ExecuteScript, "ExecuteSript {Script}         Execute script", NULL               },
	{ "Exit"         , KmsLib::ToolBase::Exit         , "Exit                          Exit"          , NULL               },
    { "Filter"       , NULL                           , "Filter ..."                                  , FILTER_COMMANDS    },
	{ "Processor"    , NULL                           , "Processor ..."                               , PROCESSOR_COMMANDS },

	{ NULL, NULL, NULL, NULL }
};

// Static functions declaration
/////////////////////////////////////////////////////////////////////////////

static void ReportStatus(OpenNet::Status aStatus, const char * aMsgOK);

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

	sSystem = OpenNet::System::Create();
	if (NULL == sSystem)
	{
		KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_FATAL_ERROR, "OpenNet::System::Create() failed");
		return __LINE__;
	}

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
		printf("Adapter %u of %u\n", i, lCount);

		sSystem->Adapter_Get(i)->Display(stdout);
	}

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Adapter listed");
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
            printf("Filter_Forward %s\n", lIt->first.c_str());

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

        printf("Filter %s\n", lIt->first.c_str());

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

void Processor_List(KmsLib::ToolBase * aToolBase, const char * aArg)
{
	assert(NULL != aArg);

	printf("Processor List %s\n", aArg);
	printf("Processor List\n");

	unsigned int lCount = sSystem->Processor_GetCount();
	for (unsigned int i = 0; i < lCount; i++)
	{
		printf("Processor %u of %u\n", i, lCount);

        sSystem->Processor_Get(i)->Display(stdout);
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
