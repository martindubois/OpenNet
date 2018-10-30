
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     TestLib/Tester.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== Includes ===========================================================
#include <OpenNet/Kernel.h>
#include <OpenNetK/Hardware_Statistics.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/OpenNet/Adapter_Statistics.h"
#include "../Common/OpenNetK/Adapter_Statistics.h"

#include "../Common/TestLib/Tester.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define BANDWIDTH_MIN_MiB_s (  0.1)
#define BANDWIDTH_MAX_MiB_s (120.0)

#define EOL "\n"

static const char * FUNCTION_FORWARD_0 =
"#include <OpenNetK/Kernel.h>"                                                       EOL
                                                                                     EOL
"OPEN_NET_FUNCTION_DECLARE( Forward0 )"                                              EOL
"{"                                                                                  EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                        EOL
                                                                                     EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED | ( 1 << ADAPTER_INDEX );" EOL
                                                                                     EOL
"    OPEN_NET_FUNCTION_END"                                                          EOL
"}"                                                                                  EOL;

static const char * FUNCTION_FORWARD_1 =
"#include <OpenNetK/Kernel.h>"                                                       EOL
                                                                                     EOL
"OPEN_NET_FUNCTION_DECLARE( Forward0 )"                                              EOL
"{"                                                                                  EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                        EOL
                                                                                     EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED | ( 1 << ADAPTER_INDEX );" EOL
                                                                                     EOL
"    OPEN_NET_FUNCTION_END"                                                          EOL
"}"                                                                                  EOL;

static const char * FUNCTION_NOTHING_0 =
"#include <OpenNetK/Kernel.h>"                              EOL
                                                            EOL
"OPEN_NET_FUNCTION_DECLARE( Nothing0 )"                     EOL
"{"                                                         EOL
"    OPEN_NET_FUNCTION_BEGIN"                               EOL
                                                            EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED;" EOL
                                                            EOL
"    OPEN_NET_FUNCTION_END"                                 EOL
"}"                                                         EOL;

static const char * FUNCTION_NOTHING_1 =
"#include <OpenNetK/Kernel.h>"                              EOL
                                                            EOL
"OPEN_NET_FUNCTION_DECLARE( Nothing1 )"                     EOL
"{"                                                         EOL
"    OPEN_NET_FUNCTION_BEGIN"                               EOL
                                                            EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED;" EOL
                                                            EOL
"    OPEN_NET_FUNCTION_END"                                 EOL
"}"                                                         EOL;

static const char * FUNCTION_REPLY_ON_ERROR_0 =
"#include <OpenNetK/Kernel.h>"                                                  EOL
                                                                                EOL
"OPEN_NET_FUNCTION_DECLARE( ReplyOnError0 )"                                    EOL
"{"                                                                             EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                   EOL
                                                                                EOL
"        __global unsigned char * lData   = lBase + lPacketInfo->mOffset_byte;" EOL
"        unsigned int             lResult = OPEN_NET_PACKET_PROCESSED;"         EOL
"        unsigned int             i;"                                           EOL
                                                                                EOL
"        for ( unsigned int j = 0; j < 10; j ++ )"                              EOL
"        {"                                                                     EOL
"            for ( i = 0; i < 6; i ++)"                                         EOL
"            {"                                                                 EOL
"                if ( 0xff != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
                                                                                EOL
"            for ( i = 6; i < 12; i ++)"                                        EOL
"            {"                                                                 EOL
"                if ( 0x00 != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
                                                                                EOL
"            for ( i = 12; i < 14; i ++)"                                       EOL
"            {"                                                                 EOL
"                if ( 0x88 != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
                                                                                EOL
"            for ( i = 14; i < lPacketInfo->mSize_byte; i ++)"                  EOL
"            {"                                                                 EOL
"                if ( 0x00 != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
"        }"                                                                     EOL
                                                                                EOL
"        lPacketInfo->mSendTo = lResult;"                                       EOL
                                                                                EOL
"    OPEN_NET_FUNCTION_END"                                                     EOL
"}"                                                                             EOL;

static const char * FUNCTION_REPLY_ON_ERROR_1 =
"#include <OpenNetK/Kernel.h>"                                                  EOL
                                                                                EOL
"OPEN_NET_FUNCTION_DECLARE( ReplyOnError1 )"                                    EOL
"{"                                                                             EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                   EOL
                                                                                EOL
"        __global unsigned char * lData   = lBase + lPacketInfo->mOffset_byte;" EOL
"        unsigned int             lResult = OPEN_NET_PACKET_PROCESSED;"         EOL
"        unsigned int             i;"                                           EOL
                                                                                EOL
"        for ( unsigned int j = 0; j < 10; j ++ )"                              EOL
"        {"                                                                     EOL
"            for ( i = 0; i < 6; i ++)"                                         EOL
"            {"                                                                 EOL
"                if ( 0xff != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
                                                                                EOL
"            for ( i = 6; i < 12; i ++)"                                        EOL
"            {"                                                                 EOL
"                if ( 0x00 != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
                                                                                EOL
"            for ( i = 12; i < 14; i ++)"                                       EOL
"            {"                                                                 EOL
"                if ( 0x88 != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
                                                                                EOL
"            for ( i = 14; i < lPacketInfo->mSize_byte; i ++)"                  EOL
"            {"                                                                 EOL
"                if ( 0x00 != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
"        }"                                                                     EOL
                                                                                EOL
"        lPacketInfo->mSendTo = lResult;"                                       EOL
                                                                                EOL
"    OPEN_NET_FUNCTION_END"                                                     EOL
"}"                                                                             EOL;

static const char * KERNEL_FORWARD =
"#include <OpenNetK/Kernel.h>"                                                       EOL
                                                                                     EOL
"OPEN_NET_KERNEL_DECLARE"                                                            EOL
"{"                                                                                  EOL
"    OPEN_NET_KERNEL_BEGIN"                                                          EOL
                                                                                     EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED | ( 1 << ADAPTER_INDEX );" EOL
                                                                                     EOL
"    OPEN_NET_KERNEL_END"                                                            EOL
"}"                                                                                  EOL;

static const char * KERNEL_NOTHING =
"#include <OpenNetK/Kernel.h>"                              EOL
                                                            EOL
"OPEN_NET_KERNEL_DECLARE"                                   EOL
"{"                                                         EOL
"    OPEN_NET_KERNEL_BEGIN"                                 EOL
                                                            EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED;" EOL
                                                            EOL
"    OPEN_NET_KERNEL_END"                                   EOL
"}"                                                         EOL;

static const char * KERNEL_REPLY_1 =
"#include <OpenNetK/Kernel.h>"                                                       EOL
                                                                                     EOL
"OPEN_NET_KERNEL_DECLARE"                                                            EOL
"{"                                                                                  EOL
"    OPEN_NET_KERNEL_BEGIN"                                                          EOL
                                                                                     EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED | ( 1 << ADAPTER_INDEX );" EOL
                                                                                     EOL
"    OPEN_NET_KERNEL_END"                                                            EOL
"}"                                                                                  EOL;

static const char * KERNEL_REPLY_ON_ERROR =
"#include <OpenNetK/Kernel.h>"                                                  EOL
                                                                                EOL
"OPEN_NET_KERNEL_DECLARE"                                                       EOL
"{"                                                                             EOL
"    OPEN_NET_KERNEL_BEGIN"                                                     EOL
                                                                                EOL
"        __global unsigned char * lData   = lBase + lPacketInfo->mOffset_byte;" EOL
"        unsigned int             lResult = OPEN_NET_PACKET_PROCESSED;"         EOL
"        unsigned int             i;"                                           EOL
                                                                                EOL
"        for ( unsigned int j = 0; j < 10; j ++ )"                              EOL
"        {"                                                                     EOL
"            for ( i = 0; i < 6; i ++)"                                         EOL
"            {"                                                                 EOL
"                if ( 0xff != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
                                                                                EOL
"            for ( i = 6; i < 12; i ++)"                                        EOL
"            {"                                                                 EOL
"                if ( 0x00 != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
                                                                                EOL
"            for ( i = 12; i < 14; i ++)"                                       EOL
"            {"                                                                 EOL
"                if ( 0x88 != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
                                                                                EOL
"            for ( i = 14; i < lPacketInfo->mSize_byte; i ++)"                  EOL
"            {"                                                                 EOL
"                if ( 0x00 != lData[ i ] )"                                     EOL
"                {"                                                             EOL
"                    lResult |= 1 << ADAPTER_INDEX;"                            EOL
"                }"                                                             EOL
"            }"                                                                 EOL
"        }"                                                                     EOL
                                                                                EOL
"        lPacketInfo->mSendTo = lResult;"                                       EOL
                                                                                EOL
"    OPEN_NET_KERNEL_END"                                                       EOL
"}"                                                                             EOL;

static const unsigned char MASK_E[6] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe };
static const unsigned char MASK_1[6] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static void DisplayConnections_1_Card ();
static void DisplayConnections_2_Cards();

namespace TestLib
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void Tester::Describe(char aTest)
    {
        switch (aTest)
        {
        case 'A': A_Describe(); break;
        case 'B': B_Describe(); break;
        case 'C': C_Describe(); break;
        case 'D': D_Describe(); break;
        case 'E': E_Describe(); break;
        case 'F': F_Describe(); break;

        default: assert(false);
        }
    }

    void Tester::A_Describe()
    {
        DisplayConnections_1_Card();

        printf(
            "===== Sequence ===============================\n"
            "    Internel   Ethernet   Internal\n"
            "\n"
            "Dropped <--- 0 <------- 1 <--- Generator\n"
            "\n"
            "Packets     x1    +    x1 = x2\n"
            "\n"
            "===== Bandwidth ==============================\n"
            "                 Send\n"
            "                 1   Read    Write   Total\n"
            "Ethernet         x1                   x1\n"
            "PCIe                  x1      x1      x2\n"
            "Memory - GPU                  x1      x1\n"
            "Memory - Main         x1              x1\n"
            "==============================================\n");
    }

    void Tester::B_Describe()
    {
        DisplayConnections_1_Card();

        printf(
            "===== Sequence ===============================\n"
            "Internel   Ethernet   Internal\n"
            "\n"
            "    +---   <-------   <--- Generator\n"
            "    |    0          1\n"
            "    +-->   ------->   ---> Dropped\n"
            "\n"
            "Packets x2    +    x1 = x3\n"
            "\n"
            "===== Bandwidth ==============================\n"
            "                 Send\n"
            "                 0   1   Read    Write   Total\n"
            "Ethernet         x1  x1                   x2\n"
            "PCIe                      x2      x2      x4\n"
            "Memory - GPU              x1      x2      x3\n"
            "Memory - Main             x1              x1\n"
            "==============================================\n");
    }

    void Tester::C_Describe()
    {
        DisplayConnections_2_Cards();

        printf(
            "===== Sequence ===============================\n"
            "    Internel   Ethernet   Internal\n"
            "\n"
            "Dropped <-a-\n"
            "             0 <------- 1 <-- Generator\n"
            "        +-b-\n"
            "        |\n"
            "        +-->   ------->   Not received\n"
            "\n"
            "a) No error\n"
            "b) Error\n"
            "\n"
            "Packets    x1    +    x1 = x2\n"
            "\n"
            "===== Bandwidth ==============================\n"
            "                 Send\n"
            "                 1   Read    Write   Total\n"
            "Ethernet         x1                    x1\n"
            "PCIe                  x1      x1       x2\n"
            "Memory - GPU          x1      x1       x2\n"
            "Memory - Main         x1               x1\n"
            "==============================================\n");
    }

    void Tester::D_Describe()
    {
        DisplayConnections_2_Cards();

        printf(
            "===== Sequence ===============================\n"
            "    Internel   Ethernet   Internal\n"
            "\n"
            "Dropped <--- 0 <------- 2 <-- Generator\n"
            "Dropped <--- 1 <------- 3 <-- Generator\n"
            "\n"
            "Packets     x2    +    x2 = x4\n"
            "\n"
            "===== Bandwidth ==============================\n"
            "                 Send\n"
            "                 2   3   Read    Write   Total\n"
            "Ethernet         x1  x1                   x2\n"
            "PCIe                      x2      x2      x4\n"
            "Memory - GPU                      x2      x2\n"
            "Memory - Main             x2              x2\n"
            "==============================================\n");
    }

    void Tester::E_Describe()
    {
        DisplayConnections_2_Cards();

        printf(
            "===== Sequence ===============================\n"
            "Internel   Ethernet   Internal\n"
            "\n"
            "    +---   <-------   <--- Generator\n"
            "    |    0          1\n"
            "    +-->   ------->   ---> Dropped\n"
            "\n"
            "Packets x2    +    x1 = x3\n"
            "\n"
            "===== Bandwidth ==============================\n"
            "                 Send\n"
            "                 0   1   Read    Write   Total\n"
            "Ethernet         x1  x1                   x2\n"
            "PCIe                      x2      x2      x4\n"
            "Memory - GPU              x1      x2      x3\n"
            "Memory - Main             x1              x1\n"
            "==============================================\n");
    }

    void Tester::F_Describe()
    {
        DisplayConnections_2_Cards();

        printf(
            "===== Sequence ===============================\n"
            "Internel   Ethernet   Internal\n"
            "\n"
            "    +--- 0 <------- 2 <--- Generator\n"
            "    |\n"
            "    +--> 1 -------> 3  ---> Dropped\n"
            "\n"
            "Packets x2    +    x1 = x3\n"
            "\n"
            "===== Bandwidth ==============================\n"
            "                 Send\n"
            "                 1   2   Read    Write   Total\n"
            "Ethernet         x1  x1                   x2\n"
            "PCIe                      x2      x2      x4\n"
            "Memory - GPU              x1      x2      x3\n"
            "Memory - Main             x1              x1\n"
            "==============================================\n");
    }

    // aMode       See MODE_...
    // aProfiling  Enable profiling at processor level
    Tester::Tester(Mode aMode, bool aProfiling)
        : mAdapterCount0   (                0  )
        , mAdapterCount1   (                0  )
        , mBandwidth_MiB_s (BANDWIDTH_MAX_MiB_s)
        , mMode            (aMode              )
        , mGeneratorCount  (                0  )
        , mPacketThroughput(                0.0)
        , mProcessor       (NULL               )
        , mProfiling       (aProfiling         )
        , mSystem          (NULL               )
    {
        // printf(__FUNCTION__ "( %d, %s)\n", aMode, aProfiling ? "true" : "false");

        unsigned int i;

        for (i = 0; i < ADAPTER_QTY; i++)
        {
            mAdapters [i] = NULL     ;
            mBufferQty[i] =         8;
            mCodes    [i] = CODE_NONE;
        }

        for (i = 0; i < GENERATOR_QTY; i++)
        {
            mGenerators[i] = NULL;
        }

        memset(&mGeneratorConfig, 0, sizeof(mGeneratorConfig));
        memset(&mNos            , 0, sizeof(mNos            ));
        memset(&mStatistics     , 0, sizeof(mStatistics     ));

        mGeneratorConfig.mBandwidth_MiB_s = BANDWIDTH_MAX_MiB_s;
        mGeneratorConfig.mPacketSize_byte =                1024;
    }

    Tester::~Tester()
    {
        if (NULL != mSystem)
        {
            Uninit();
        }
    }

    void Tester::SetBandwidth(double aBandwidth_MiB_s)
    {
        // printf(__FUNCTION__ "( %f MiB/s )\n", aBandwidth_MiB_s);

        assert(BANDWIDTH_MIN_MiB_s <= aBandwidth_MiB_s);
        assert(BANDWIDTH_MAX_MiB_s >= aBandwidth_MiB_s);

        mGeneratorConfig.mBandwidth_MiB_s = aBandwidth_MiB_s;
    }

    void Tester::SetPacketSize(unsigned int aPacketSize_byte)
    {
        // printf(__FUNCTION__ "( %u bytes )\n", aPacketSize_byte);

        assert(64                   <= aPacketSize_byte);
        assert(PACKET_SIZE_MAX_byte >= aPacketSize_byte);

        mGeneratorConfig.mPacketSize_byte = aPacketSize_byte;
    }

    unsigned int Tester::A(unsigned int aBufferQty)
    {
        // printf(__FUNCTION__ "( %u )\n", aBufferQty);

        mAdapterCount0  =            1;
        mCodes[0]       = CODE_NOTHING;
        mGeneratorCount =            1;

        unsigned int lResult = A_Init(aBufferQty);
        if (0 != lResult)
        {
            return lResult;
        }

        assert(NULL != mGenerators[0]);
        assert(NULL != mAdapters  [1]);

        OpenNet::Status lStatus = mGenerators[0]->SetAdapter(mAdapters[1]);
        assert(OpenNet::STATUS_OK == lStatus);

        Adapters_SetProcessing();

        Start();

        Sleep(100);

        ResetAdapterStatistics();

        Sleep(1000);

        GetAdapterStatistics();

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 12700;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =    53;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 22300;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =    78;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 22300;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =    78;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 22300;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =    78;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1170;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 13400;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =   153;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 1430000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =    4990;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 1430000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =    4990;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =   1 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 1430000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =   13800;

        lResult = Adapter_VerifyStatistics(0);

        Stop  ();
        Uninit();

        return lResult;
    }

    unsigned int Tester::B(unsigned int aBufferQty)
    {
        assert(0 < aBufferQty);

        mAdapterCount0  =            2;
        mCodes[0]       = CODE_REPLY  ;
        mCodes[1]       = CODE_NOTHING;
        mGeneratorCount =            1;

        unsigned int lResult = B_Init(aBufferQty);
        if (0 != lResult)
        {
            return lResult;
        }

        assert(NULL != mAdapters  [1]);
        assert(NULL != mGenerators[0]);

        OpenNet::Status lStatus = mGenerators[0]->SetAdapter(mAdapters[1]);
        assert(OpenNet::STATUS_OK == lStatus);

        Adapters_SetProcessing();

        Start();

        Sleep(100);

        ResetAdapterStatistics();

        Sleep(1000);

        GetAdapterStatistics();

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1075;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 9830;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =  149;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 9830;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =  149;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 18700;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   158;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMax = 629000;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  4210;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 629000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 629000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 630000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 639000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 998000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 630000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin =   9530;

        lResult = Adapter_VerifyStatistics(0);

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1680;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 9300;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =  149;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 9300;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =  149;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 18700;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   158;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMax = 995000;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMin =  14000;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  4210;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 595000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 995000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin =  14000;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 594000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 975000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 630000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 998000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin =   9530;

        lResult += Adapter_VerifyStatistics(1);

        Stop  ();
        Uninit();

        return lResult;
    }

    unsigned int Tester::C(unsigned int aBufferQty)
    {
        assert(0 < aBufferQty);

        mAdapterCount0  =                   1;
        mCodes[0]       = CODE_REPLY_ON_ERROR;
        mGeneratorCount =                   1;

        unsigned int lResult = C_Init(aBufferQty);
        if (0 != lResult)
        {
            return lResult;
        }

        assert(NULL != mAdapters  [1]);
        assert(NULL != mGenerators[0]);

        OpenNet::Status lStatus = mGenerators[0]->SetAdapter(mAdapters[1]);
        assert(OpenNet::STATUS_OK == lStatus);

        Adapters_SetProcessing();

        Start();

        Sleep(100);

        ResetAdapterStatistics();

        Sleep(1000);

        GetAdapterStatistics();

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1640;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 22100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =   216;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 22100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =   216;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 22100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   216;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  7490;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 1420000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =   13800;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 1420000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =   13800;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 1 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 1420000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =   13800;

        lResult = Adapter_VerifyStatistics(0);

        Stop  ();
        Uninit();

        return lResult;
    }

    unsigned int Tester::D(unsigned int aBufferQty)
    {
        assert(0 < aBufferQty);

        mAdapterCount0  = 2;
        mCodes[0]       = CODE_NOTHING;
        mCodes[1]       = CODE_NOTHING;
        mGeneratorCount = 2;

        unsigned int lResult = D_Init(aBufferQty);
        if (0 != lResult)
        {
            return lResult;
        }

        assert(NULL != mAdapters  [2]);
        assert(NULL != mAdapters  [3]);
        assert(NULL != mGenerators[0]);
        assert(NULL != mGenerators[1]);

        OpenNet::Status lStatus = mGenerators[0]->SetAdapter(mAdapters[2]);
        assert(OpenNet::STATUS_OK == lStatus);

        lStatus = mGenerators[1]->SetAdapter(mAdapters[3]);
        assert(OpenNet::STATUS_OK == lStatus);

        Adapters_SetProcessing();

        Start();

        Sleep(100);

        ResetAdapterStatistics();

        Sleep(1000);

        GetAdapterStatistics();

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10300;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =    64;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 25400;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =   110;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 25400;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =   110;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 27100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   220;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1170;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 11500;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =   158;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 1620000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =    7040;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 1620000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =    7010;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =   1 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 1620000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =    7010;

        lResult = Adapter_VerifyStatistics(0);

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10300;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =    64;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 25400;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =   110;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 25400;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =   110;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 27100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   220;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1170;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 11500;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =   158;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 1620000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =    7040;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 1620000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =    7010;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =   1 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 1620000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =    7010;

        lResult += Adapter_VerifyStatistics(1);

        Stop();
        Uninit();

        return lResult;
    }

    unsigned int Tester::E(unsigned int aBufferQty)
    {
        assert(0 < aBufferQty);

        mAdapterCount0  =            2;
        mCodes[0]       = CODE_REPLY  ;
        mCodes[1]       = CODE_NOTHING;
        mGeneratorCount =            1;

        unsigned int lResult = E_Init(aBufferQty);
        if (0 != lResult)
        {
            return lResult;
        }

        assert(NULL != mAdapters  [1]);
        assert(NULL != mGenerators[0]);

        OpenNet::Status lStatus = mGenerators[0]->SetAdapter(mAdapters[1]);
        assert(OpenNet::STATUS_OK == lStatus);

        Adapters_SetProcessing();

        Start();

        Sleep(100);

        ResetAdapterStatistics();

        Sleep(1000);

        GetAdapterStatistics();

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =   926;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 10700;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =   149;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 10700;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =   149;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 19400;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   158;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMax = 683000;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  4210;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 683000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 683000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 683000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 683000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 1220000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =    9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 683000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin =   9530;

        lResult = Adapter_VerifyStatistics(0);

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1680;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 10700;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =   149;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 10700;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =   149;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 19400;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   158;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMax = 1250000;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMin =   14000;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  4210;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 619000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 1250000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin =   14000;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 683000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 1180000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin =    9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 643000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =   9530;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 1220000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin =    9530;

        lResult += Adapter_VerifyStatistics(1);

        Stop  ();
        Uninit();

        return lResult;
    }

    unsigned int Tester::F(unsigned int aBufferQty)
    {
        assert(0 < aBufferQty);

        mAdapterCount0  =            2;
        mCodes[0]       = CODE_FORWARD;
        mCodes[1]       = CODE_NOTHING;
        mGeneratorCount =            1;

        unsigned int lResult = F_Init(aBufferQty);
        if (0 != lResult)
        {
            return lResult;
        }

        assert(NULL != mAdapters  [2]);
        assert(NULL != mGenerators[0]);

        OpenNet::Status lStatus = mGenerators[0]->SetAdapter(mAdapters[2]);
        assert(OpenNet::STATUS_OK == lStatus);

        Adapters_SetProcessing();

        Start();

        Sleep(100);

        ResetAdapterStatistics();

        Sleep(1000);

        GetAdapterStatistics();

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1010;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 12100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =    25;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 12100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =    25;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 12100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =    25;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  1160;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 771000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =  13800;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 771000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =  13800;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 771000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =  13800;

        lResult = Adapter_VerifyStatistics(0);

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  8370;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 12100;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   217;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMax = 771000;
        mConstraints[TestLib::Tester::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMin =  13800;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  8650;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 771000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin =  13800;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 771000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin =  13800;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin =  15 * 1024 * 1024;

        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 771000;
        mConstraints[TestLib::Tester::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin =  13800;

        lResult += Adapter_VerifyStatistics(1);

        Stop  ();
        Uninit();

        return lResult;
    }

    double Tester::Adapter_GetBandwidth() const
    {
        return mBandwidth_MiB_s;
    }

    double Tester::Adapter_GetPacketThroughput() const
    {
        return mPacketThroughput;
    }

    void Tester::DisplaySpeed()
    {
        double lRx_byte_s   [2];
        double lRx_KiB_s    [2];
        double lRx_MiB_s    [2];
        double lRx_packet_s [2];
        double lSum_MiB_s   [2];
        double lSum_packet_s[2];
        double lTx_byte_s   [2];
        double lTx_KiB_s    [2];
        double lTx_MiB_s    [2];
        double lTx_packet_s [2];

        for (unsigned int i = 0; i < 2; i++)
        {
            double lDuration_s = static_cast<double>(mStatistics[i][ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms]) / 1000.0; // ms ==> s

            lRx_byte_s  [i] = mStatistics[i][HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte  ];
            lRx_packet_s[i] = mStatistics[i][HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet];
            lTx_byte_s  [i] = mStatistics[i][HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte  ];
            lTx_packet_s[i] = mStatistics[i][HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet];

            lSum_packet_s[i] = lRx_packet_s[i] + lTx_packet_s[i];

            lRx_byte_s   [i] /= lDuration_s;
            lRx_packet_s [i] /= lDuration_s;
            lSum_packet_s[i] /= lDuration_s;
            lTx_byte_s   [i] /= lDuration_s;
            lTx_packet_s [i] /= lDuration_s;

            lRx_KiB_s[i] = lRx_byte_s[i] / 1024;
            lRx_MiB_s[i] = lRx_KiB_s [i] / 1024;
            lTx_KiB_s[i] = lTx_byte_s[i] / 1024;
            lTx_MiB_s[i] = lTx_KiB_s [i] / 1024;

            lSum_MiB_s[i] = lRx_MiB_s[i] + lTx_MiB_s[i];
        }

        printf("\t\tAdapter 0\t\t\t\t\tAdapters 1\n");
        printf("\t\tRx\t\tTx\t\tTotal\t\tRx\t\tTx\t\tTotal\n");
        printf("Packets/s\t%f\t%f\t%f\t%f\t%f\t%f\n", lRx_packet_s[0], lTx_packet_s[0], lSum_packet_s[0], lRx_packet_s[1], lTx_packet_s[1], lSum_packet_s[1]);
        printf("MiB/s\t\t%f\t%f\t%f\t%f\t%f\t%f\n"  , lRx_MiB_s   [0], lTx_MiB_s   [0], lSum_MiB_s   [0], lRx_MiB_s   [1], lTx_MiB_s   [1], lSum_MiB_s   [1]);

        mBandwidth_MiB_s  = lRx_MiB_s   [0];
        mPacketThroughput = lRx_packet_s[0];
    }

    unsigned int Tester::Search(char aTest, unsigned int aBufferQty)
    {
        assert(0 < aBufferQty);

        double lMin_MiB_s = BANDWIDTH_MIN_MiB_s;
        double lMax_MiB_s = BANDWIDTH_MAX_MiB_s;
        double lCenter_MiB_s;

        while ((lMin_MiB_s + 0.1) < lMax_MiB_s)
        {
            lCenter_MiB_s = (lMax_MiB_s + lMin_MiB_s) / 2.0;

            printf("Search  %f MiB/s\n", lCenter_MiB_s);

            SetBandwidth(lCenter_MiB_s);

            if (0 >= Test(aTest, aBufferQty))
            {
                lMin_MiB_s = lCenter_MiB_s;
            }
            else
            {
                lMax_MiB_s = lCenter_MiB_s;
            }
        }

        return 0;
    }

    // TODO TestLib.DualTest
    //      Replacer les exception par un retour de numero de ligne

    // Exception  KmsLib::Exception *  CODE_ERROR
    void Tester::Start()
    {
        assert(   1 <= mGeneratorCount);
        assert(NULL != mSystem        );

        OpenNet::Status lStatus;
        unsigned int    i      ;

        for (i = 0; i < mGeneratorCount; i++)
        {
            assert(NULL != mGenerators[i]);

            OpenNet::Status lStatus = mGenerators[i]->SetConfig(mGeneratorConfig);
            assert(OpenNet::STATUS_OK == lStatus);
        }

        lStatus = mSystem->Start(OpenNet::System::START_FLAG_LOOPBACK);
        if (OpenNet::STATUS_OK != lStatus)
        {
            OpenNet::Kernel * lKernel = mSystem->Kernel_Get(0);

            if (NULL != lKernel)
            {
                lKernel->Display(stdout);
            }

            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "System::Start(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }

        Sleep(100);

        for (i = 0; i < mGeneratorCount; i++)
        {
            lStatus = mGenerators[i]->Start();
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void Tester::Stop()
    {
        assert(   1 <= mGeneratorCount);
        assert(NULL != mSystem        );

        OpenNet::Status lStatus = mSystem->Stop();
        assert(OpenNet::STATUS_OK == lStatus);

        for (unsigned int i = 0; i < mGeneratorCount; i++)
        {
            assert(NULL != mGenerators[i]);

            lStatus = mGenerators[i]->Stop();
            assert(OpenNet::STATUS_OK == lStatus);
        }

        ResetInputFilter();
    }

    unsigned int Tester::Test(char aTest, unsigned int aBufferQty)
    {
        unsigned int lResult;

        switch (aTest)
        {
        case 'A': lResult = A(aBufferQty); break;
        case 'B': lResult = B(aBufferQty); break;
        case 'C': lResult = C(aBufferQty); break;
        case 'D': lResult = D(aBufferQty); break;
        case 'E': lResult = E(aBufferQty); break;
        case 'F': lResult = F(aBufferQty); break;

        default: assert(false);
        }

        return lResult;
    }

    unsigned int Tester::Verify(char aTest, unsigned int aBufferQty)
    {
        assert(0 < aBufferQty);

        for (unsigned int i = 0; i < 5; i++)
        {
            printf("Verify  %f MiB/s\n", mGeneratorConfig.mBandwidth_MiB_s);

            if (0 < Test(aTest, aBufferQty))
            {
                mGeneratorConfig.mBandwidth_MiB_s -= 0.1;
                i = 0;
            }
        }

        return 0;
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    unsigned int Tester::A_Init(unsigned int aBufferQty)
    {
        assert(0 < aBufferQty);

        mBufferQty[0] = aBufferQty;

        unsigned int lResult = Init0();
        if (0 == lResult)
        {
            lResult = Init_2_SameCard();
            if (0 == lResult)
            {
                lResult = Init1();
            }
        }

        return lResult;
    }

    unsigned int Tester::B_Init(unsigned int aBufferQty)
    {
        mBufferQty[0] = aBufferQty;
        mBufferQty[1] = aBufferQty;

        unsigned int lResult = Init0();
        if (0 == lResult)
        {
            lResult = Init_2_SameCard();
            if (0 == lResult)
            {
                lResult = Init1();
            }
        }

        return lResult;
    }

    unsigned int Tester::C_Init(unsigned int aBufferQty)
    {
        assert(0 < aBufferQty);

        mBufferQty[0] = aBufferQty;

        unsigned int lResult = Init0();
        if (0 == lResult)
        {
            lResult = Init_2_SamePort();
            if (0 == lResult)
            {
                lResult = Init1();
            }
        }

        return lResult;
    }

    unsigned int Tester::D_Init(unsigned int aBufferQty)
    {
        assert(0 < aBufferQty);

        mBufferQty[0] = aBufferQty;
        mBufferQty[1] = aBufferQty;

        unsigned int lResult = Init0();
        if (0 == lResult)
        {
            lResult = Init_4_SameCard();
            if (0 == lResult)
            {
                lResult = Init1();
            }
        }

        return lResult;
    }

    unsigned int Tester::E_Init(unsigned int aBufferQty)
    {
        mBufferQty[0] = aBufferQty;
        mBufferQty[1] = aBufferQty;

        unsigned int lResult = Init0();
        if (0 == lResult)
        {
            lResult = Init_2_SamePort();
            if (0 == lResult)
            {
                lResult = Init1();
            }
        }

        return lResult;
    }

    unsigned int Tester::F_Init(unsigned int aBufferQty)
    {
        mBufferQty[0] = aBufferQty;
        mBufferQty[1] = aBufferQty;

        unsigned int lResult = Init0();
        if (0 == lResult)
        {
            lResult = Init_4_SameCard();
            if (0 == lResult)
            {
                lResult = Init1();
            }
        }

        return lResult;
    }

    unsigned int Tester::Init0()
    {
        assert(            1 <= mGeneratorCount);
        assert(GENERATOR_QTY >= mGeneratorCount);
        assert(NULL          == mProcessor     );
        assert(NULL          == mSystem        );

        for (unsigned int i = 0; i < mGeneratorCount; i++)
        {
            assert(NULL == mGenerators[i]);

            mGenerators[i] = OpenNet::PacketGenerator::Create();
            assert(NULL != mGenerators[i]);
        }

        mSystem = OpenNet::System::Create();
        assert(NULL != mSystem);

        mAdapters[0] = mSystem->Adapter_Get(0);
        if (NULL == mAdapters[0])
        {
            printf(__FUNCTION__ " - System::Adapter_Get(  ) failed\n");
            return __LINE__;
        }

        return 0;
    }

    unsigned int Tester::Init1()
    {
        mProcessor = mSystem->Processor_Get(0);
        if (NULL == mProcessor)
        {
            printf(__FUNCTION__ " - System::Processor_Get() failed\n");
            return __LINE__;
        }

        if (mProfiling)
        {
            Processor_EnableProfiling();
        }

        Adapter_Connect    ();
        Adapters_RetrieveNo();
        SetProcessor       ();
        SetConfig          ();

        return 0;
    }

    unsigned int Tester::Init_2_SameCard()
    {
        assert(NULL != mAdapters[0]);
        assert(NULL == mAdapters[1]);

        mAdapterCount1 = 2;

        OpenNet::Adapter::Info lInfo;

        OpenNet::Status lStatus = mAdapters[0]->GetInfo(&lInfo);
        assert(OpenNet::STATUS_OK == lStatus);

        mAdapters[1] = mSystem->Adapter_Get(lInfo.mEthernetAddress.mAddress, MASK_E, MASK_1);
        if (NULL == mAdapters[1])
        {
            printf(__FUNCTION__ " - System::Adapter_Get( , ,  ) failed\n");
            return __LINE__;
        }

        return 0;
    }

    unsigned int Tester::Init_2_SamePort()
    {
        assert(NULL != mAdapters[0]);
        assert(NULL == mAdapters[1]);

        mAdapterCount1 = 2;

        OpenNet::Adapter::Info lInfo;

        OpenNet::Status lStatus = mAdapters[0]->GetInfo(&lInfo);
        assert(OpenNet::STATUS_OK == lStatus);

        mAdapters[1] = mSystem->Adapter_Get(lInfo.mEthernetAddress.mAddress, MASK_1, MASK_E);
        if (NULL == mAdapters[1])
        {
            printf(__FUNCTION__ " - System::Adapter_Get( , ,  ) failed\n");
            return __LINE__;
        }

        return 0;
    }

    unsigned int Tester::Init_4_SameCard()
    {
        assert(NULL != mAdapters[0]);
        assert(NULL == mAdapters[1]);
        assert(NULL == mAdapters[2]);
        assert(NULL == mAdapters[3]);

        mAdapterCount1 = 4;

        OpenNet::Adapter::Info lInfo;

        OpenNet::Status lStatus = mAdapters[0]->GetInfo(&lInfo);
        assert(OpenNet::STATUS_OK == lStatus);

        mAdapters[1] = mSystem->Adapter_Get(lInfo.mEthernetAddress.mAddress, MASK_E, MASK_1);
        if (NULL == mAdapters[1])
        {
            printf(__FUNCTION__ " - System::Adapter_Get( , ,  ) failed\n");
            return __LINE__;
        }

        mAdapters[2] = mSystem->Adapter_Get(lInfo.mEthernetAddress.mAddress, MASK_1, MASK_E);
        if (NULL == mAdapters[1])
        {
            printf(__FUNCTION__ " - System::Adapter_Get( , ,  ) failed\n");
            return __LINE__;
        }

        lStatus = mAdapters[2]->GetInfo(&lInfo);
        assert(OpenNet::STATUS_OK == lStatus);

        mAdapters[3] = mSystem->Adapter_Get(lInfo.mEthernetAddress.mAddress, MASK_E, MASK_1);
        if (NULL == mAdapters[1])
        {
            printf(__FUNCTION__ " - System::Adapter_Get( , ,  ) failed\n");
            return __LINE__;
        }

        return 0;
    }

    unsigned int Tester::Uninit()
    {
        assert(   1 <= mAdapterCount0 );
        assert(   2 <= mAdapterCount1 );
        assert(   1 <= mGeneratorCount);
        assert(NULL != mSystem        );

        for (unsigned int i = 0; i < mGeneratorCount; i++)
        {
            assert(NULL != mGenerators[i]);

            mGenerators[i]->Delete();
        }

        mSystem->Delete();

        mAdapterCount0  =    0;
        mAdapterCount1  =    0;
        mGeneratorCount =    0;
        mProcessor      = NULL;
        mSystem         = NULL;

        memset(&mAdapters  , 0, sizeof(mAdapters  ));
        memset(&mGenerators, 0, sizeof(mGenerators));

        return 0;
    }

    // Exception  KmsLib::Exception *  CODE_ERROR
    void Tester::Adapter_Connect()
    {
        assert(   1 <= mAdapterCount1);
        assert(NULL != mSystem       );

        for (unsigned int i = 0; i < mAdapterCount1; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mSystem->Adapter_Connect(mAdapters[i]);
            if (OpenNet::STATUS_OK != lStatus)
            {
                throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                    "System::Adapter_Connect(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
            }
        }
    }

    void Tester::Adapter_InitialiseConstraints()
    {
        KmsLib::ValueVector::Constraint_Init(mConstraints, STATS_QTY);

        mConstraints[ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_RESET].mMax = 0xffffffff;
    }

    void Tester::Adapter_SetProcessing(OpenNet::Adapter * aAdapter, OpenNet::Function * aFunction, const char * aCode, const char * aName, const char * aIndex)
    {
        assert(NULL != aAdapter );
        assert(NULL != aFunction);
        assert(NULL != aCode    );
        assert(NULL != aName    );

        OpenNet::Status lStatus = aFunction->SetCode(aCode, static_cast<unsigned int>(strlen(aCode)));
        assert(OpenNet::STATUS_OK == lStatus);

        lStatus = aFunction->SetFunctionName(aName);
        assert(OpenNet::STATUS_OK == lStatus);

        if (NULL != aIndex)
        {
            unsigned int lRet = aFunction->Edit_Replace("ADAPTER_INDEX", aIndex);
            assert(0 < lRet);
        }

        lStatus = aAdapter->SetInputFilter(aFunction);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    void Tester::Adapter_SetProcessing(OpenNet::Adapter * aAdapter, OpenNet::Kernel * aKernel, const char * aCode, const char * aIndex)
    {
        assert(NULL != aAdapter);
        assert(NULL != aKernel );
        assert(NULL != aCode   );

        OpenNet::Status lStatus = aKernel->SetCode(aCode, static_cast<unsigned int>(strlen(aCode)));
        assert(OpenNet::STATUS_OK == lStatus);

        if (NULL != aIndex)
        {
            unsigned int lRet = aKernel->Edit_Replace("ADAPTER_INDEX", aIndex);
            assert(0 < lRet);
        }

        lStatus = aAdapter->SetInputFilter(aKernel);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    unsigned int Tester::Adapter_VerifyStatistics(unsigned int aAdapter)
    {
        assert(ADAPTER_QTY > aAdapter);

        OpenNet::Adapter * lAdapter = mAdapters[aAdapter];
        assert(NULL != lAdapter);

        return KmsLib::ValueVector::Constraint_Verify(mStatistics[aAdapter], lAdapter->GetStatisticsQty(), mConstraints, stdout, reinterpret_cast<const KmsLib::ValueVector::Description *>(lAdapter->GetStatisticsDescriptions()));
    }

    void Tester::Adapters_RetrieveNo()
    {
        assert(          1 <= mAdapterCount0);
        assert(ADAPTER_QTY >= mAdapterCount0);

        for (unsigned int i = 0; i < mAdapterCount0; i++)
        {
            unsigned int lNo;

            OpenNet::Status lStatus = mAdapters[i]->GetAdapterNo(&lNo);
            assert(OpenNet::STATUS_OK == lStatus);

            sprintf_s(mNos[i], "%u", lNo);
        }
    }

    void Tester::Adapters_SetProcessing()
    {
        assert(          1 <= mAdapterCount0);
        assert(ADAPTER_QTY >= mAdapterCount0);

        for (unsigned int i = 0; i < mAdapterCount0; i++)
        {
            OpenNet::Adapter * lA     = mAdapters[i];
            const char       * lIndex = mNos     [i];
            unsigned int       lOther = (i + 1) % mAdapterCount0;

            assert(NULL != lAdapter);

            switch (mMode)
            {
            case MODE_FUNCTION:
                OpenNet::Function * lF;
                
                lF = mFunctions + i;

                switch (mCodes[i])
                {
                case CODE_FORWARD       : Adapter_SetProcessing(lA, lF, (0 == i) ? FUNCTION_FORWARD_0        : FUNCTION_FORWARD_1       , (0 == i) ? "Forward0"      : "Forward1"    , mNos[lOther]); break;
                case CODE_NONE          :                                                                                                                                                             break;
                case CODE_NOTHING       : Adapter_SetProcessing(lA, lF, (0 == i) ? FUNCTION_NOTHING_0        : FUNCTION_NOTHING_1       , (0 == i) ? "Nothing0"      : "Nothing1"    , NULL        ); break;
                case CODE_REPLY         : Adapter_SetProcessing(lA, lF, (0 == i) ? FUNCTION_FORWARD_0        : FUNCTION_FORWARD_1       , (0 == i) ? "Forward0"      : "Forward1"    , lIndex      ); break;
                case CODE_REPLY_ON_ERROR: Adapter_SetProcessing(lA, lF, (0 == i) ? FUNCTION_REPLY_ON_ERROR_0 : FUNCTION_REPLY_ON_ERROR_1, (0 == i) ? "ReplyOnError0" : "ReplyOnError", lIndex      ); break;

                default: assert(false);
                }
                break;

            case MODE_KERNEL:
                OpenNet::Kernel * lK;
                
                lK = mKernels + i;

                switch (mCodes[i])
                {
                case CODE_FORWARD       : Adapter_SetProcessing(lA, lK, KERNEL_FORWARD       , mNos[lOther]); break;
                case CODE_NONE          :                                                                     break;
                case CODE_NOTHING       : Adapter_SetProcessing(lA, lK, KERNEL_NOTHING       , NULL        ); break;
                case CODE_REPLY         : Adapter_SetProcessing(lA, lK, KERNEL_FORWARD       , lIndex      ); break;
                case CODE_REPLY_ON_ERROR: Adapter_SetProcessing(lA, lK, KERNEL_REPLY_ON_ERROR, lIndex      ); break;

                default: assert(false);
                }
                break;

            default: assert(false);
            }
        }
    }

    void Tester::GetAdapterStatistics()
    {
        assert(          1 <= mAdapterCount1);
        assert(ADAPTER_QTY >= mAdapterCount1);

        for (unsigned int i = 0; i < mAdapterCount1; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->GetStatistics(mStatistics[i], sizeof(mStatistics[i]), NULL, true);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void Tester::Processor_EnableProfiling()
    {
        assert(NULL != mProcessor);

        OpenNet::Processor::Config lConfig;

        OpenNet::Status lStatus = mProcessor->GetConfig(&lConfig);
        assert(OpenNet::STATUS_OK == lStatus);

        lConfig.mFlags.mProfilingEnabled = true;

        lStatus = mProcessor->SetConfig(lConfig);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    void Tester::ResetAdapterStatistics()
    {
        assert(          1 <= mAdapterCount1);
        assert(ADAPTER_QTY >= mAdapterCount1);

        for (unsigned int i = 0; i < mAdapterCount1; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->ResetStatistics();
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void Tester::ResetInputFilter()
    {
        assert(          1 <= mAdapterCount1);
        assert(ADAPTER_QTY >= mAdapterCount1);

        for (unsigned int i = 0; i < mAdapterCount1; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->ResetInputFilter();
            assert((OpenNet::STATUS_OK == lStatus) || (OpenNet::STATUS_FILTER_NOT_SET == lStatus));
        }
    }

    // Exception  KmsLib::Exception *  CODE_ERROR
    void Tester::SetConfig()
    {
        assert(          1 <= mAdapterCount1);
        assert(ADAPTER_QTY >= mAdapterCount1);

        for (unsigned int i = 0; i < mAdapterCount1; i++)
        {
            assert(0 < mBufferQty[i]);

            OpenNet::Adapter * lAdapter = mAdapters[i];
            assert(NULL != lAdapter);

            OpenNet::Adapter::Config lConfig;

            OpenNet::Status lStatus = lAdapter->GetConfig(&lConfig);
            assert(OpenNet::STATUS_OK == lStatus);

            lConfig.mBufferQty = mBufferQty[i];

            lAdapter->SetConfig(lConfig);
            if (OpenNet::STATUS_OK != lStatus)
            {
                throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                    "Adapter::SetConfig(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
            }
        }
    }

    void Tester::SetProcessor()
    {
        assert(          1 <= mAdapterCount1);
        assert(ADAPTER_QTY >= mAdapterCount1);
        assert(NULL        != mProcessor    );

        for (unsigned int i = 0; i < mAdapterCount1; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->SetProcessor(mProcessor);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

}

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

void DisplayConnections_1_Card()
{
    printf(
        "===== Connections ============================\n"
        "Card\n"
        "\n"
        " P0\n"
        " |\n"
        " P1\n"
        "\n");
}

void DisplayConnections_2_Cards()
{
    printf(
        "===== Connections ============================\n"
        "Card 0   Card 1\n"
        "\n"
        " P0-------P0\n"
        "\n"
        " P1-------P1\n"
        "\n");
}
