
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Tool/PacketGenerator.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>

// ===== C++ ================================================================
#include <map>
#include <string>

// ===== Import/Includes ====================================================
#include <KmsLib/ToolBase.h>

// ===== Includes ===========================================================
#include <OpenNet/PacketGenerator.h>

// ===== OpenNet_Tool =======================================================
#include "Globals.h"
#include "Utils.h"

#include "PacketGenerator.h"

// Commandes
/////////////////////////////////////////////////////////////////////////////

// ===== PacketGenerator Set ================================================

static void Set_Adapter    (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Set_Bandwidth  (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Set_IndexOffset(KmsLib::ToolBase * aToolBase, const char * aArg);
static void Set_PacketSize (KmsLib::ToolBase * aToolBase, const char * aArg);

static const KmsLib::ToolBase::CommandInfo SET_COMMANDS[] =
{
    { "Adapter"    , Set_Adapter    , "Adapter"           , NULL },
    { "Bandwidth"  , Set_Bandwidth  , "Bandwidth {MB/s}"  , NULL },
    { "IndexOffset", Set_IndexOffset, "IndexOffset [byte]", NULL },
    { "PacketSize" , Set_PacketSize , "PacketSize {byte}" , NULL },

    { NULL, NULL, NULL, NULL }
};

// ===== PacketGenerator ====================================================

static void Create       (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Delete       (KmsLib::ToolBase * aToolBase, const char * aArg);
static void DisplayConfig(KmsLib::ToolBase * aToolBase, const char * aArg);
static void List         (KmsLib::ToolBase * aToolBase, const char * aArg);
static void ResetConfig  (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Select       (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Start        (KmsLib::ToolBase * aToolBase, const char * aArg);
static void Stop         (KmsLib::ToolBase * aToolBase, const char * aArg);

const KmsLib::ToolBase::CommandInfo PACKET_GENERATOR_COMMANDS[] =
{
    { "Create"       , Create       , "Create {Name}" , NULL         },
    { "Delete"       , Delete       , "Delete"        , NULL         },
    { "DisplayConfig", DisplayConfig, "DisplayConfig" , NULL         },
    { "List"         , List         , "List"          , NULL         },
    { "ResetCongif"  , ResetConfig  , "ResetConfig"   , NULL         },
    { "Select"       , Select       , "Select {Name}" , NULL         },
    { "Set"          , NULL         , "Set ..."       , SET_COMMANDS },
    { "Start"        , Start        , "Start"         , NULL         },
    { "Stop"         , Stop         , "Stop"          , NULL         },

    { NULL, NULL, NULL, NULL }
};

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef std::map<std::string, OpenNet::PacketGenerator * > PacketGeneratorMap;

// Static variables
/////////////////////////////////////////////////////////////////////////////

static OpenNet::PacketGenerator * sPacketGenerator = NULL;
static PacketGeneratorMap         sPacketGenerators;

// Macros
/////////////////////////////////////////////////////////////////////////////

#define VERIFY_SELECTED                                                                          \
    if (NULL == sPacketGenerator)                                                                \
    {                                                                                            \
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "No PacketGenerator selected"); \
        return;                                                                                  \
    }

// Functions
/////////////////////////////////////////////////////////////////////////////

void PacketGenerator_Exit()
{
    for (PacketGeneratorMap::iterator lIt = sPacketGenerators.begin(); lIt != sPacketGenerators.end(); lIt++)
    {
        assert(NULL != lIt->second);

        lIt->second->Delete();
    }

    sPacketGenerators.clear();
}

// Commands
/////////////////////////////////////////////////////////////////////////////

// ===== PacketGenerator Set ================================================

void Set_Adapter(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    VERIFY_SELECTED

    if (NULL == gAdapter)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "No selected Adapter");
    }
    else
    {
        OpenNet::Status lStatus = sPacketGenerator->SetAdapter(gAdapter);
        UTL_VERIFY_STATUS("OpenNet::PacketGenerator::GetConfig(  ) failed")

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Adapter set");
    }
}

void Set_Bandwidth(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    double lBandwidth_MiB_s;

    UTL_PARSE_ARGUMENT("%lf", &lBandwidth_MiB_s)

    VERIFY_SELECTED

    OpenNet::PacketGenerator::Config lConfig;

    OpenNet::Status lStatus = sPacketGenerator->GetConfig(&lConfig);
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::GetConfig(  ) failed")

    lConfig.mBandwidth_MiB_s = lBandwidth_MiB_s;

    lStatus = sPacketGenerator->SetConfig(lConfig);
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::SetConfig(  ) failed")

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "Bandwidth set");
}

void Set_IndexOffset(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    unsigned int lIndexOffset_byte;

    UTL_PARSE_ARGUMENT("%u", &lIndexOffset_byte)

    VERIFY_SELECTED

    OpenNet::PacketGenerator::Config lConfig;

    OpenNet::Status lStatus = sPacketGenerator->GetConfig(&lConfig);
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::GetConfig(  ) failed")

    lConfig.mIndexOffset_byte = lIndexOffset_byte;

    lStatus = sPacketGenerator->SetConfig(lConfig);
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::SetConfig(  ) failed")

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "IndexOffset set");
}

void Set_PacketSize(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    unsigned int lPacketSize_byte;

    UTL_PARSE_ARGUMENT("%u", &lPacketSize_byte);

    VERIFY_SELECTED

    OpenNet::PacketGenerator::Config lConfig;

    OpenNet::Status lStatus = sPacketGenerator->GetConfig(&lConfig);
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::GetConfig(  ) failed")

    lConfig.mPacketSize_byte = lPacketSize_byte;

    lStatus = sPacketGenerator->SetConfig(lConfig);
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::SetConfig(  ) failed")

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "IndexOffset set");
}

// ===== PacketGenetator ====================================================

void Create(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    char lName[64];

    if (1 != sscanf_s(aArg, "%[A-Za-z0-9_]", lName SIZE_INFO(static_cast<unsigned int>(sizeof(lName)))))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Invalid command line");
        return;
    }

    OpenNet::PacketGenerator * lPacketGenerator = OpenNet::PacketGenerator::Create();
    if (NULL == lPacketGenerator)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "OpenNet::PacketGenerator::Create() failed");
        return;
    }

    sPacketGenerator = lPacketGenerator;

    sPacketGenerators.insert(PacketGeneratorMap::value_type(lName, sPacketGenerator));

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PacketGenerator created");
}

void Delete(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    VERIFY_SELECTED

    for (PacketGeneratorMap::iterator lIt = sPacketGenerators.begin(); lIt != sPacketGenerators.end(); lIt++)
    {
        if (sPacketGenerator == lIt->second)
        {
            sPacketGenerators.erase(lIt);
            break;
        }
    }

    sPacketGenerator->Delete();

    if (sPacketGenerators.empty())
    {
        sPacketGenerator = NULL;
    }
    else
    {
        sPacketGenerator = sPacketGenerators.begin()->second;
    }
}

void DisplayConfig(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    VERIFY_SELECTED

    OpenNet::PacketGenerator::Config lConfig;

    OpenNet::Status lStatus = sPacketGenerator->GetConfig(&lConfig);
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::GetConfig(  ) failed")

    lStatus = OpenNet::PacketGenerator::Display(lConfig, stdout);
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::GetConfig(  ) failed")
}

void List(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    for (PacketGeneratorMap::iterator lIt = sPacketGenerators.begin(); lIt != sPacketGenerators.end(); lIt++)
    {
        printf("    %s\n", lIt->first.c_str());
    }

    printf("%zu PacketGenerator\n", sPacketGenerators.size());
}

void ResetConfig(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    VERIFY_SELECTED

    OpenNet::Status lStatus = sPacketGenerator->ResetConfig();
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::ResetConfig(  ) failed")

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "PacketGenerator config reset");
}

void Select(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    assert(NULL != aArg);

    char lName[64];

    if (1 != sscanf_s(aArg, "%[A-Za-z0-9_]", lName SIZE_INFO(static_cast<unsigned int>(sizeof(lName)))))
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Invalid command line");
        return;
    }

    PacketGeneratorMap::iterator lIt = sPacketGenerators.find(lName);
    if (sPacketGenerators.end() == lIt)
    {
        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_ERROR, "Invalid PacketGenerator name");
    }
    else
    {
        sPacketGenerator = lIt->second;

        KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PacketGenerator selected");
    }
}

void Start(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    VERIFY_SELECTED

    OpenNet::Status lStatus = sPacketGenerator->Start();
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::Start(  ) failed")

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PacketGenerator started");
}

void Stop(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    VERIFY_SELECTED

    OpenNet::Status lStatus = sPacketGenerator->Stop();
    UTL_VERIFY_STATUS("OpenNet::PacketGenerator::Stop(  ) failed")

    KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PacketGenerator stopped");
}
