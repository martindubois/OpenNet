
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Tool/Test.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>

// ===== Import/Includes ====================================================
#include <KmsLib/ToolBase.h>

// ===== Common =============================================================
#include "../Common/TestLib/TestFactory.h"

// ===== OpenNet_Tool =======================================================
#include "Test.h"

// Commandes
/////////////////////////////////////////////////////////////////////////////

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
static void Test_Summary      (KmsLib::ToolBase * aToolBase, const char * aArg);

const KmsLib::ToolBase::CommandInfo TEST_COMMANDS[] =
{
    { "DisplayConfig", Test_DisplayConfig, "DisplayConfig"                                            , NULL },
    { "Info"         , Test_Info         , "Info {TestName}"                                          , NULL },
    { "ResetCongif"  , Test_ResetConfig  , "ResetConfig"                                              , NULL },
	{ "Run"          , Test_Run          , "Run {TestName}"                                           , NULL },
    { "Search"       , NULL              , "Search ..."                                               , TEST_SEARCH_COMMANDS },
    { "SetBandwidth" , Test_SetBandwidth , "SetBandwidth {Bandwidth_MiB/s}"                           , NULL },
    { "SetBufferQty" , Test_SetBufferQty , "SetBufferQty {BufferQty}"                                 , NULL },
    { "SetCode"      , Test_SetCode      , "SetCode DEFAULT|FORWARD|NOTHING|REPLY|REPLY_ON_ERROR|... ", NULL },
    { "SetMode"      , Test_SetMode      , "SetMode DEFAULT|FUNCTION|KERNEL"                          , NULL },
    { "SetPacketSize", Test_SetPacketSize, "SetPacketSize {PacketSize_byte}"                          , NULL },
    { "SetProfiling" , Test_SetProfiling , "SetProfiling false|true"                                  , NULL },
    { "Verify"       , NULL              , "Verify {TestName}"                                        , TEST_VERIFY_COMMANDS },
    { "StartStop"    , Test_StartStop    , "StatStop {TestName}"                                      , NULL },
    { "Summary"      , Test_Summary      , "Summary"                                                  , NULL },

	{ NULL, NULL, NULL, NULL }
};

// Global variable
/////////////////////////////////////////////////////////////////////////////

static TestLib::TestFactory sTestFactory;
static unsigned int         sTest_Failed = 0;
static unsigned int         sTest_Passed = 0;

// Macros
/////////////////////////////////////////////////////////////////////////////

#define TEST_EXEC(M,F)                                                           \
    void Test_##F(KmsLib::ToolBase * aToolBase, const char * aArg)               \
    {                                                                            \
        printf("Test " M " %s\n", aArg);                                         \
        TestLib::Test * lTest = sTestFactory.CreateTest(aArg);                   \
        if (NULL != lTest)                                                       \
        {                                                                        \
            unsigned int lRet = lTest->F();                                      \
            if (0 == lRet)                                                       \
            {                                                                    \
                KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, "PASSED"); \
                sTest_Passed ++;                                                 \
            }                                                                    \
            else                                                                 \
            {                                                                    \
                sTest_Failed ++;                                                 \
            }                                                                    \
                                                                                 \
            delete lTest;                                                        \
        }                                                                        \
    }

#define TEST_SET(P)                                                           \
    void Test_Set##P(KmsLib::ToolBase * aToolBase, const char * aArg)         \
    {                                                                         \
        printf("Test Set" #P " %s\n", aArg);                                  \
        unsigned int lRet = sTestFactory.Set##P( aArg );                      \
        if (0 == lRet)                                                        \
        {                                                                     \
            KmsLib::ToolBase::Report(KmsLib::ToolBase::REPORT_OK, #P " set"); \
        }                                                                     \
    }

// Commands
/////////////////////////////////////////////////////////////////////////////

TEST_EXEC( "Run"              , Run               )
TEST_EXEC( "Search Bandwidth" , Search_Bandwidth  )
TEST_EXEC( "Search BufferQty" , Search_BufferQty  )
TEST_EXEC( "Search PacketSize", Search_PacketSize )
TEST_EXEC( "StartStop"        , StartStop         )
TEST_EXEC( "Verify Bandwidth" , Verify_Bandwidth  )
TEST_EXEC( "Verify BufferQty" , Verify_BufferQty  )
TEST_EXEC( "Verify PacketSize", Verify_PacketSize )

TEST_SET( Bandwidth  )
TEST_SET( BufferQty  )
TEST_SET( Code       )
TEST_SET( Mode       )
TEST_SET( PacketSize )
TEST_SET( Profiling  )

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

void Test_Summary(KmsLib::ToolBase * aToolBase, const char * aArg)
{
    printf("Test Summary : %u tests FAILED and %u tests PASSED\n", sTest_Failed, sTest_Passed );

    sTest_Failed = 0;
    sTest_Passed = 0;
}
