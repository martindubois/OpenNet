
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       TestLib/Test.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <string.h>

#ifdef _KMS_WINDOWS_
#include <io.h>
#endif

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Includes ===========================================================
#include <OpenNetK/Constants.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/TestLib/Test.h"

// OpenCL
/////////////////////////////////////////////////////////////////////////////

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
"OPEN_NET_FUNCTION_DECLARE( Forward1 )"                                              EOL
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
"        for ( i = 0; i < 6; i ++)"                                             EOL
"        {"                                                                     EOL
"            if ( 0xff != lData[ i ] )"                                         EOL
"            {"                                                                 EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                EOL
"            }"                                                                 EOL
"        }"                                                                     EOL
                                                                                EOL
"        for ( i = 12; i < 14; i ++)"                                           EOL
"        {"                                                                     EOL
"            if ( 0x0a != lData[ i ] )"                                         EOL
"            {"                                                                 EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                EOL
"            }"                                                                 EOL
"        }"                                                                     EOL
                                                                                EOL
"        for ( i = 14; i < lPacketInfo->mSize_byte; i ++)"                      EOL
"        {"                                                                     EOL
"            if ( 0x00 != lData[ i ] )"                                         EOL
"            {"                                                                 EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                EOL
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
"        for ( i = 0; i < 6; i ++)"                                             EOL
"        {"                                                                     EOL
"            if ( 0xff != lData[ i ] )"                                         EOL
"            {"                                                                 EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                EOL
"            }"                                                                 EOL
"        }"                                                                     EOL
                                                                                EOL
"        for ( i = 12; i < 14; i ++)"                                           EOL
"        {"                                                                     EOL
"            if ( 0x0a != lData[ i ] )"                                         EOL
"            {"                                                                 EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                EOL
"            }"                                                                 EOL
"        }"                                                                     EOL
                                                                                EOL
"        for ( i = 14; i < lPacketInfo->mSize_byte; i ++)"                      EOL
"        {"                                                                     EOL
"            if ( 0x00 != lData[ i ] )"                                         EOL
"            {"                                                                 EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                EOL
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

static const char * KERNEL_REPLY =
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
"        for ( i = 0; i < 6; i ++)"                                             EOL
"        {"                                                                     EOL
"            if ( 0xff != lData[ i ] )"                                         EOL
"            {"                                                                 EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                EOL
"            }"                                                                 EOL
"        }"                                                                     EOL
                                                                                EOL
"        for ( i = 12; i < 14; i ++)"                                           EOL
"        {"                                                                     EOL
"            if ( 0x0a != lData[ i ] )"                                         EOL
"            {"                                                                 EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                EOL
"            }"                                                                 EOL
"        }"                                                                     EOL
                                                                                EOL
"        for ( i = 14; i < lPacketInfo->mSize_byte; i ++)"                      EOL
"        {"                                                                     EOL
"            if ( 0x00 != lData[ i ] )"                                         EOL
"            {"                                                                 EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                EOL
"            }"                                                                 EOL
"        }"                                                                     EOL
                                                                                EOL
"        lPacketInfo->mSendTo = lResult;"                                       EOL
                                                                                EOL
"    OPEN_NET_KERNEL_END"                                                       EOL
"}"                                                                             EOL;

namespace TestLib
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    const double Test::BANDWIDTH_MAX_MiB_s = 120.0;
    const double Test::BANDWIDTH_MIN_MiB_s =   0.1;

    const unsigned int Test::BUFFER_QTY_MAX = OPEN_NET_BUFFER_QTY - 1;
    const unsigned int Test::BUFFER_QTY_MIN =                       1;

    const char * Test::CODE_NAMES[CODE_QTY] = { "DEFAULT", "FORWARD", "NONE", "NOTHING", "REPLY", "REPLY_ON_ERROR" };
    const char * Test::MODE_NAMES[MODE_QTY] = { "DEFAULT", "FUNCTION", "KERNEL" };

    const unsigned char Test::MASK_E[6] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe };
    const unsigned char Test::MASK_1[6] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };

    const unsigned int Test::TEST_PACKET_SIZE_MAX_byte = 9000;
    const unsigned int Test::TEST_PACKET_SIZE_MIN_byte =   64;

    // aName [---;R--] DEFAULT|FORWARD|NONE|NOTHING|REPLY|REPLY_ON_ERROR
    // aOut  [---;-W-] The method writes the enum value there
    //
    // Return
    //      0  OK
    //  Other  Error
    unsigned int Test::CodeFromName(const char * aName, Code * aOut)
    {
        assert(NULL != aName);
        assert(NULL != aOut );

        for (int i = 0; i < CODE_QTY; i++)
        {
            if (0 == _stricmp(CODE_NAMES[i], aName))
            {
                (*aOut) = static_cast<Code>(i);
                return 0;
            }
        }

        printf(__FUNCTION__ " - %s is not a code name\n", aName);
        return __LINE__;
    }

    // aName [---;R--] DEFAULT|FUNCTION|KERNEL
    // aOut  [---;-W-] The method writes the enum value there
    //
    // Return
    //      0  OK
    //  Other  Error
    unsigned int Test::ModeFromName(const char * aName, Mode * aOut)
    {
        assert(NULL != aName);
        assert(NULL != aOut );

        for (int i = 0; i < MODE_QTY; i++)
        {
            if (0 == _strnicmp(MODE_NAMES[i], aName, strlen(MODE_NAMES[i])))
            {
                (*aOut) = static_cast<Mode>(i);
                return 0;
            }
        }

        printf(__FUNCTION__ " - %s is not a mode name\n", aName);
        return __LINE__;
    }

    Test::~Test()
    {
        if (NULL != mSystem)
        {
            Uninit();
        }
    }

    // aConfig [---;R--]
    void Test::SetConfig(const Config & aConfig)
    {
        assert(CODE_QTY > aConfig.mCode);
        assert(MODE_QTY > aConfig.mMode);

        assert(CODE_DEFAULT != mConfig.mCode );
        assert(CODE_QTY     >  mConfig.mCode );
        assert(MODE_DEFAULT != mConfig.mMode );
        assert(MODE_QTY     >  mConfig.mMode );
        assert(CODE_DEFAULT != mDefault.mCode);
        assert(CODE_QTY     >  mDefault.mCode);
        assert(MODE_DEFAULT != mDefault.mMode);
        assert(MODE_QTY     >  mDefault.mMode);

        mConfig = aConfig;

        if (CODE_DEFAULT == aConfig.mCode)
        {
            mConfig.mCode = mDefault.mCode;
        }

        if (MODE_DEFAULT == aConfig.mMode)
        {
            mConfig.mMode = mDefault.mMode;
        }
    }

    // Return
    //      0  OK
    //  Other  Error
    unsigned int Test::Run()
    {
        unsigned int lResult = InitAndPrepare();
        if (0 == lResult)
        {
            lResult = Execute();
            if (0 == lResult)
            {
                DisplayAndWriteResult("Run");
            }

            Uninit();
        }
            
        return lResult;
    }

    // Return
    //      0  OK
    //  Other  Error
    unsigned int Test::Search_Bandwidth()
    {
        unsigned int lResult = InitAndPrepare();
        if (0 == lResult)
        {
            lResult = __LINE__;

            double lCenter_MiB_s;
            double lMax_MiB_s = BANDWIDTH_MAX_MiB_s;
            double lMin_MiB_s = BANDWIDTH_MIN_MiB_s;

            for (;;)
            {
                lCenter_MiB_s = (lMax_MiB_s + lMin_MiB_s) / 2.0;

                printf("Search  %.1f MiB / s\n", lCenter_MiB_s);

                mConfig.mBandwidth_MiB_s = lCenter_MiB_s;

                if (0 >= Execute())
                {
                    if (lMin_MiB_s > (lCenter_MiB_s - 0.1)) { lResult = 0; break; }
                    lMin_MiB_s = lCenter_MiB_s;
                }
                else
                {
                    if (lMax_MiB_s < (lCenter_MiB_s + 0.1))
                    {
                        if (lMax_MiB_s < (lMin_MiB_s + 0.1)) { lMin_MiB_s -= 0.1; }
                        lMax_MiB_s -= 0.1;
                    }
                    else
                    {
                        lMax_MiB_s = lCenter_MiB_s;
                    }
                }
            }

            if (0 == lResult)
            {
                DisplayAndWriteResult("Search Bandwidth");
            }

            Uninit();
        }

        return lResult;
    }

    // Return
    //      0  OK
    //  Other  Error
    unsigned int Test::Search_BufferQty()
    {
        unsigned int lResult = InitAndPrepare();
        if (0 == lResult)
        {
            lResult = __LINE__;

            unsigned int lCenter;
            unsigned int lMax = BUFFER_QTY_MAX;
            unsigned int lMin = BUFFER_QTY_MIN;

            for (;;)
            {
                lCenter = (lMax + lMin) / 2;

                printf("Search  %u buffers\n", lCenter);

                mConfig.mBufferQty = lCenter;

                if (0 >= Execute())
                {
                    if (lMax == lCenter) { lResult = 0;  break; }
                    lMax = lCenter;
                }
                else
                {
                    if (lMin == lCenter)
                    {
                        if (lMin == lMax) { lMax++; }
                        lMin ++;
                    }
                    else
                    {
                        lMin = lCenter;
                    }
                }
            }

            if (0 == lResult) { DisplayAndWriteResult("Search Buffer Quantity"); }

            Uninit();
        }

        return lResult;
    }

    // Return
    //      0  OK
    //  Other  Error
    unsigned int Test::Search_PacketSize()
    {
        unsigned int lResult = InitAndPrepare();
        if (0 == lResult)
        {
            lResult = __LINE__;

            unsigned int lCenter_byte;
            unsigned int lMax_byte = TEST_PACKET_SIZE_MAX_byte;
            unsigned int lMin_byte = TEST_PACKET_SIZE_MIN_byte;

            for (;;)
            {
                lCenter_byte = (lMax_byte + lMin_byte) / 2;

                printf("Search  %u bytes / packet\n", lCenter_byte);

                mConfig.mPacketSize_byte = lCenter_byte;

                if (0 >= Execute())
                {
                    if (lMax_byte == lCenter_byte) { lResult = 0; break; }
                    lMax_byte = lCenter_byte;
                }
                else
                {
                    if (lMin_byte == lCenter_byte)
                    {
                        if (lMin_byte == lMax_byte)
                        {
                            if (TEST_PACKET_SIZE_MAX_byte <= lMax_byte) { break; }
                            lMax_byte++;
                        }
                        lMin_byte++;
                    }
                    else
                    {
                        lMin_byte = lCenter_byte;
                    }
                }
            }

            if (0 == lResult)
            {
                DisplayAndWriteResult("Search Packet Size");
            }

            Uninit();
        }

        return lResult;
    }

    // Return
    //      0  OK
    //  Other  Error
    unsigned int Test::Verify_Bandwidth()
    {
        unsigned int lResult = InitAndPrepare();
        if (0 == lResult)
        {
            lResult = __LINE__;

            unsigned int lCount = 0;

            while (BANDWIDTH_MIN_MiB_s <= mConfig.mBandwidth_MiB_s)
            {
                printf("Verify  %.1f MiB / s\n", mConfig.mBandwidth_MiB_s);

                lResult = Execute();
                if (0 >= lResult)
                {
                    lCount++;
                    if (5 <= lCount)
                    {
                        DisplayAndWriteResult("Verify Bandwidth");
                        break;
                    }
                }
                else
                {
                    mConfig.mBandwidth_MiB_s -= 0.1;
                    lCount = 0;
                }
            }
        }

        return lResult;
    }

    // Return
    //      0  OK
    //  Other  Error
    unsigned int Test::Verify_BufferQty()
    {
        unsigned int lResult = InitAndPrepare();
        if (0 == lResult)
        {
            lResult = __LINE__;

            unsigned int lCount = 0;

            while (BUFFER_QTY_MAX >= mConfig.mBufferQty)
            {
                printf("Verify  %u buffers\n", mConfig.mBufferQty);

                lResult = Execute();
                if (0 >= lResult)
                {
                    lCount++;
                    if (5 <= lCount)
                    {
                        DisplayAndWriteResult("Verify Buffer Quantity");
                        break;
                    }
                }
                else
                {
                    mConfig.mBufferQty++;
                    lCount = 0;
                }
            }
        }

        return lResult;
    }

    // Return
    //      0  OK
    //  Other  Error
    unsigned int Test::Verify_PacketSize()
    {
        unsigned int lResult = InitAndPrepare();
        if (0 == lResult)
        {
            lResult = __LINE__;

            unsigned int lCount = 0;

            while (TEST_PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte)
            {
                printf("Verify  %u bytes / packet\n", mConfig.mPacketSize_byte);

                lResult = Execute();
                if (0 >= lResult)
                {
                    lCount++;
                    if (5 <= lCount)
                    {
                        DisplayAndWriteResult("Verify Packet Size");
                        break;
                    }
                }
                else
                {
                    mConfig.mPacketSize_byte++;
                    lCount = 0;
                }
            }
        }

        return lResult;
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    void Test::Connections_Display_1_Card()
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

    void Test::Connections_Display_2_Cards()
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

    // aName [---;R--]
    // aCode
    // aMode
    Test::Test(const char * aName, Code aCode, Mode aMode)
        : mAdapterCount0 (   1)
        , mAdapterCount1 (   1)
        , mGeneratorCount(   1)
        , mProcessor     (NULL)
        , mSystem        (NULL)
    {
        assert(NULL         != aName);
        assert(CODE_DEFAULT != aCode);
        assert(CODE_QTY     >  aCode);
        assert(MODE_DEFAULT != aMode);
        assert(MODE_QTY     >  aMode);

        mConfig.mBandwidth_MiB_s =    0.0;
        mConfig.mBufferQty       =    2  ;
        mConfig.mCode            = aCode ;
        mConfig.mMode            = aMode ;
        mConfig.mPacketSize_byte = 1024  ;
        mConfig.mProfiling       = false ;

        mDefault.mCode = aCode;
        mDefault.mMode = aMode;

        mResult.mBandwidth_MiB_s           = BANDWIDTH_MAX_MiB_s;
        mResult.mPacketThroughput_packet_s =                 0.0;

        memset(&mAdapterStats  , 0, sizeof(mAdapterStats  ));
        memset(&mName      , 0, sizeof(mName      ));
        memset(&mNos       , 0, sizeof(mNos       ));

        strncpy_s(mName, aName, sizeof(mName) - 1);

        unsigned int i;

        for (i = 0; i < ADAPTER_QTY; i++)
        {
            mAdapters [i] = NULL     ;
            mBufferQty[i] =         2;
            mCodes    [i] = CODE_NONE;
        }

        for (i = 0; i < GENERATOR_QTY; i++)
        {
            mGenerators[i] = NULL;
        }
    }

    // aIndex
    // aCounter
    //
    // Return
    unsigned int Test::GetAdapterStats(unsigned int aIndex, unsigned int aCounter)
    {
        assert(ADAPTER_QTY > aIndex  );
        assert(STATS_QTY   > aCounter);

        return mAdapterStats[aIndex][aCounter];
    }

    // Return
    const Test::Config * Test::GetConfig() const
    {
        return (&mConfig);
    }

    // aIndex
    //
    // Return
    OpenNet::PacketGenerator * Test::GetGenerator(unsigned int aIndex)
    {
        assert(GENERATOR_QTY > aIndex);

        assert(NULL != mGenerators[aIndex]);

        return mGenerators[aIndex];
    }

    // Return
    OpenNet::System * Test::GetSystem()
    {
        assert(NULL != mSystem);

        return mSystem;
    }

    // aCount
    void Test::SetAdapterCount0(unsigned int aCount)
    {
        assert(          1 <= aCount);
        assert(ADAPTER_QTY >= aCount);

        assert(          1 <= mAdapterCount0);
        assert(ADAPTER_QTY >= mAdapterCount0);

        mAdapterCount0 = aCount;
    }

    // aCount
    void Test::SetAdapterCount1(unsigned int aCount)
    {
        assert(          1 <= aCount);
        assert(ADAPTER_QTY >= aCount);

        assert(          1 <= mAdapterCount1);
        assert(ADAPTER_QTY >= mAdapterCount1);

        mAdapterCount1 = aCount;
    }

    // aIndex
    // aQty
    void Test::SetBufferQty(unsigned int aIndex, unsigned int aQty)
    {
        assert(ADAPTER_QTY    >  aIndex);
        assert(BUFFER_QTY_MAX >= aQty  );
        assert(BUFFER_QTY_MIN <= aQty  );

        assert(BUFFER_QTY_MAX >= mBufferQty[aIndex]);
        assert(BUFFER_QTY_MIN <= mBufferQty[aIndex]);

        mBufferQty[aIndex] = aQty;
    }

    // aIndex
    // aCode
    void Test::SetCode(unsigned int aIndex, Code aCode)
    {
        assert(ADAPTER_QTY >  aIndex);
        assert(CODE_QTY    >= aCode );

        assert(CODE_DEFAULT != mCodes[aIndex]);
        assert(CODE_QTY     >  mCodes[aIndex]);

        mCodes[aIndex] = aCode;
    }

    // aCount
    void Test::SetGeneratorCount(unsigned int aCount)
    {
        assert(            1 <= aCount);
        assert(GENERATOR_QTY >= aCount);

        assert(1             <= mGeneratorCount);
        assert(GENERATOR_QTY >= mGeneratorCount);

        mGeneratorCount = aCount;
    }

    void Test::DisplayAdapterStats(unsigned int aIndex)
    {
        assert(ADAPTER_QTY > aIndex);

        printf("Displaying statistics of adapter %u\n", aIndex);

        OpenNet::Adapter * lAdapter = mAdapters[aIndex];
        assert(NULL != lAdapter);

        OpenNet::Status lStatus = lAdapter->DisplayStatistics(mAdapterStats[aIndex], sizeof(mAdapterStats[aIndex]), stdout);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    // Return
    //      0  OK
    //  Ohter  Error
    unsigned int Test::Init()
    {
        assert(NULL          == mAdapters[0]   );
        assert(CODE_QTY      >  mConfig.mCode  );
        assert(GENERATOR_QTY >= mGeneratorCount);
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
        if (NULL == mSystem)
        {
            printf(__FUNCTION__ " - No adapter\n");
            return __LINE__;
        }

        mProcessor = mSystem->Processor_Get(0);
        if (NULL == mProcessor)
        {
            printf(__FUNCTION__ " - No processor\n");
            return __LINE__;
        }

        ConfigProcessor();

        return 0;
    }

    void Test::InitAdapterConstraints()
    {
        KmsLib::ValueVector::Constraint_Init(mConstraints, STATS_QTY);

        mConstraints[ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_RESET].mMax = 0xffffffff;
    }

    // Return
    //      0  OK
    //  Ohter  Error
    unsigned int Test::Start()
    {
        assert(            1 <= mGeneratorCount);
        assert(GENERATOR_QTY >= mGeneratorCount);
        assert(NULL          != mSystem        );

        ConfigAdapters  ();
        ConfigGenerators();

        OpenNet::Status lStatus = mSystem->Start(OpenNet::System::START_FLAG_LOOPBACK);
        if (OpenNet::STATUS_OK != lStatus)
        {
            OpenNet::Kernel * lKernel = mSystem->Kernel_Get(0);
            if (NULL != lKernel)
            {
                lKernel->Display(stdout);
            }

            printf(__FUNCTION__ " - System::Start(  ) returned %u\n", lStatus);
            return __LINE__;
        }

        #ifdef _KMS_LINUX_
            int lRet = usleep(100000);
            assert(0 == lRet);
        #endif
        
        #ifdef _KMS_WINDOWS_
            Sleep(100);
        #endif

        for (unsigned int i = 0; i < mGeneratorCount; i++)
        {
            assert(NULL != mGenerators[i]);

            lStatus = mGenerators[i]->Start();
            assert(OpenNet::STATUS_OK == lStatus);
        }

        #ifdef _KMS_LINUX_
            lRet = usleep(100000);
            assert(0 == lRet);
        #endif
        
        #ifdef _KMS_WINDOWS_
            Sleep(100);
        #endif

        ResetStatistics();

        return 0;
    }

    // Return
    //      0  OK
    //  Ohter  Error
    unsigned int Test::Stop()
    {
        assert(            1 <= mGeneratorCount);
        assert(GENERATOR_QTY >= mGeneratorCount);
        assert(NULL          != mSystem        );

        RetrieveStatistics();

        OpenNet::Status lStatus;

        unsigned int i;

        if (mConfig.mProfiling)
        {
            unsigned int lStats[128];

            switch (mConfig.mMode)
            {
            case MODE_FUNCTION:
                OpenNet::Kernel * lKernel;

                lKernel = GetSystem()->Kernel_Get(0);
                assert(NULL != lKernel);

                lStatus = lKernel->GetStatistics(lStats, sizeof(lStats), NULL, false);
                assert(OpenNet::STATUS_OK == lStatus);

                lStatus = lKernel->DisplayStatistics(lStats, sizeof(lStats), stdout);
                assert(OpenNet::STATUS_OK == lStatus);
                break;

            case MODE_KERNEL:
                for (i = 0; i < mAdapterCount0; i++)
                {
                    lStatus = mKernels[i].GetStatistics(lStats, sizeof(lStats), NULL, false);
                    assert(OpenNet::STATUS_OK == lStatus);

                    lStatus = mKernels[i].DisplayStatistics(lStats, sizeof(lStats), stdout);
                    assert(OpenNet::STATUS_OK == lStatus);
                }
                break;

            default: assert(false);
            }
        }

        #ifdef _KMS_LINUX_
            mPriorityClass = 0;
        #endif
        
        #ifdef _KMS_WINDOWS_
            mPriorityClass = GetPriorityClass(GetCurrentProcess());
        #endif

        lStatus = mSystem->Stop();
        if (OpenNet::STATUS_OK != lStatus)
        {
            printf("WARNING  System::Stop() failed - ");
            OpenNet::Status_Display(lStatus, stdout);
            printf("\n");
        }

        for (i = 0; i < mGeneratorCount; i++)
        {
            assert(NULL != mGenerators[i]);

            lStatus = mGenerators[i]->Stop();
            assert(OpenNet::STATUS_OK == lStatus);
        }

        return 0;
    }

    // aAdapterIndex
    //
    // Return
    //      0  OK
    //  Other  Error
    unsigned int Test::VerifyAdapterStats(unsigned int aIndex)
    {
        assert(ADAPTER_QTY > aIndex);

        printf("Verifying statistics of adapter %u\n", aIndex);

        OpenNet::Adapter * lAdapter = mAdapters[aIndex];
        assert(NULL != lAdapter);

        unsigned int lRet = KmsLib::ValueVector::Constraint_Verify(mAdapterStats[aIndex], lAdapter->GetStatisticsQty(), mConstraints, stdout, reinterpret_cast<const KmsLib::ValueVector::Description *>(lAdapter->GetStatisticsDescriptions()));
        if (0 != lRet)
        {
            printf(__FUNCTION__ " - ValueVector::Constrain_Verify( , , , ,  ) returned %u\n", lRet);
            return __LINE__;
        }

        return 0;
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    void Test::ConfigAdapters()
    {
        assert(          1 <= mAdapterCount0);
        assert(ADAPTER_QTY >= mAdapterCount0);

        for (unsigned int i = 0; i < mAdapterCount0; i++)
        {
            assert(BUFFER_QTY_MAX >= mBufferQty[i]);
            assert(BUFFER_QTY_MIN <= mBufferQty[i]);

            OpenNet::Adapter * lAdapter = mAdapters[i];
            assert(NULL != lAdapter);

            OpenNet::Adapter::Config lConfig;

            OpenNet::Status lStatus = lAdapter->GetConfig(&lConfig);
            assert(OpenNet::STATUS_OK == lStatus);

            lConfig.mBufferQty = mBufferQty[i];

            lStatus = lAdapter->SetConfig(lConfig);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void Test::ConfigGenerators()
    {
        assert(BANDWIDTH_MAX_MiB_s       >= mConfig.mBandwidth_MiB_s);
        assert(BANDWIDTH_MIN_MiB_s       <= mConfig.mBandwidth_MiB_s);
        assert(TEST_PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);
        assert(TEST_PACKET_SIZE_MIN_byte <= mConfig.mPacketSize_byte);
        assert(                        1 <= mGeneratorCount         );
        assert(GENERATOR_QTY             >= mGeneratorCount         );

        for (unsigned int i = 0; i < mGeneratorCount; i++)
        {
            OpenNet::PacketGenerator * lGenerator = mGenerators[i];
            assert(NULL != lGenerator);

            OpenNet::PacketGenerator::Config lConfig;

            OpenNet::Status lStatus = lGenerator->GetConfig(&lConfig);
            assert(OpenNet::STATUS_OK == lStatus);

            lConfig.mBandwidth_MiB_s = mConfig.mBandwidth_MiB_s;
            lConfig.mPacketSize_byte = mConfig.mPacketSize_byte;

            lStatus = lGenerator->SetConfig(lConfig);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void Test::ConfigProcessor()
    {
        assert(NULL != mProcessor);

        OpenNet::Processor::Config lConfig;

        OpenNet::Status lStatus = mProcessor->GetConfig(&lConfig);
        assert(OpenNet::STATUS_OK == lStatus);

        lConfig.mFlags.mProfilingEnabled = mConfig.mProfiling;

        lStatus = mProcessor->SetConfig(lConfig);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    void Test::DisplayResult()
    {
        printf(
            "Test %s - PASSED\n"
            "    Buffer Quantity = %u\n"
            "    Packet Size     = %u bytes\n"
            "    Bandwidth       = %.1f MiB/s\n"
            "    Throughput      = %.1f packets/s\n",
            mName,
            mConfig.mBufferQty,
            mConfig.mPacketSize_byte,
            mResult.mBandwidth_MiB_s,
            mResult.mPacketThroughput_packet_s);
    }

    void Test::DisplayAndWriteResult(const char * aNote)
    {
        assert(NULL != aNote);

        DisplayResult();
        WriteResult  (aNote);
    }

    // Return
    //      0  OK
    //  Ohter  Error
    unsigned int Test::Execute()
    {
        unsigned int lResult = Start();
        if (0 == lResult)
        {
            #ifdef _KMS_LINUX_
                int lRet = sleep(1);
                assert(0 == lRet);
            #endif
            
            #ifdef _KMS_WINDOWS_
                Sleep(1000);
            #endif

            lResult = Stop();
        }

        return lResult;
    }

    // Return
    //      0  OK
    //  Ohter  Error
    unsigned int Test::InitAndPrepare()
    {
        unsigned int lResult = Init();
        if (0 == lResult)
        {
            Prepare();
        }

        return lResult;
    }

    void Test::Prepare()
    {
        assert(             0 <  mAdapterCount0);
        assert(ADAPTER_QTY    >= mAdapterCount0);
        assert(NULL           != mProcessor    );
        assert(NULL           != mSystem       );

        unsigned int i;

        for (i = 0; i < mAdapterCount0; i++)
        {
            OpenNet::Adapter * lAdapter = mAdapters[i];
            assert(NULL != lAdapter);

            OpenNet::Status lStatus = lAdapter->SetProcessor(mProcessor);
            assert(OpenNet::STATUS_OK == lStatus);

            lStatus = mSystem->Adapter_Connect(lAdapter);
            assert(OpenNet::STATUS_OK == lStatus);

            unsigned int lNo;

            lStatus = lAdapter->GetAdapterNo(&lNo);
            assert(OpenNet::STATUS_OK == lStatus);

            sprintf_s(mNos[i], "%u", lNo);
        }

        for (i = 0; i < mAdapterCount0; i++)
        {
            switch (mConfig.mMode)
            {
            case MODE_FUNCTION: SetFunction(i); break;
            case MODE_KERNEL  : SetKernel  (i); break;

            default: assert(false);
            }
        }
    }

    void Test::ResetInputFilters()
    {
        assert(          1 <= mAdapterCount0);
        assert(ADAPTER_QTY >= mAdapterCount0);

        for (unsigned int i = 0; i < mAdapterCount0; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->ResetInputFilter();
            if (OpenNet::STATUS_OK != lStatus)
            {
                printf("WARNING  Adapter::ResetInputFilter() failed - ");
                OpenNet::Status_Display(lStatus, stdout);
                printf("\n");
            }
        }
    }

    void Test::ResetStatistics()
    {
        // printf(__FUNCTION__ "()\n");

        assert(          1 <= mAdapterCount1);
        assert(ADAPTER_QTY >= mAdapterCount1);

        for (unsigned int i = 0; i < mAdapterCount1; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->ResetStatistics();
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void Test::RetrieveStatistics()
    {
        assert(            1 <= mAdapterCount1 );
        assert(ADAPTER_QTY   >= mAdapterCount1 );
        assert(            1 <= mGeneratorCount);
        assert(GENERATOR_QTY >= mGeneratorCount);

        for (unsigned int i = 0; i < mAdapterCount1; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->GetStatistics(mAdapterStats[i], sizeof(mAdapterStats[i]), NULL, true);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    // aAdapterIndex
    void Test::SetFunction(unsigned int aAdapterIndex)
    {
        assert(ADAPTER_QTY > aAdapterIndex);

        assert(ADAPTER_QTY   > mAdapterCount0);
        assert(aAdapterIndex < mAdapterCount0);

        OpenNet::Adapter * lAdapter = mAdapters[aAdapterIndex];
        assert(NULL != lAdapter);

        OpenNet::Function * lF          = mFunctions + aAdapterIndex;
        unsigned int        lOtherIndex = (aAdapterIndex + 1) % mAdapterCount0;

        switch (mCodes[aAdapterIndex])
        {
        case CODE_FORWARD       : SetFunction(lAdapter, lF, (0 == aAdapterIndex) ? FUNCTION_FORWARD_0        : FUNCTION_FORWARD_1       , (0 == aAdapterIndex) ? "Forward0"      : "Forward1"    , mNos[lOtherIndex  ]); break;
        case CODE_NONE          :                                                                                                                                                                                        break;
        case CODE_NOTHING       : SetFunction(lAdapter, lF, (0 == aAdapterIndex) ? FUNCTION_NOTHING_0        : FUNCTION_NOTHING_1       , (0 == aAdapterIndex) ? "Nothing0"      : "Nothing1"    , NULL               ); break;
        case CODE_REPLY         : SetFunction(lAdapter, lF, (0 == aAdapterIndex) ? FUNCTION_FORWARD_0        : FUNCTION_FORWARD_1       , (0 == aAdapterIndex) ? "Forward0"      : "Forward1"    , mNos[aAdapterIndex]); break;
        case CODE_REPLY_ON_ERROR: SetFunction(lAdapter, lF, (0 == aAdapterIndex) ? FUNCTION_REPLY_ON_ERROR_0 : FUNCTION_REPLY_ON_ERROR_1, (0 == aAdapterIndex) ? "ReplyOnError0" : "ReplyOnError", mNos[aAdapterIndex]); break;

        default: assert(false);
        }
    }

    // aAdapter  [---;RW-]
    // aFunction [-K-;RW-]
    // aCode     [---;R--]
    // aName     [---;R--]
    // aIndex    [---;R--]
    void Test::SetFunction(OpenNet::Adapter * aAdapter, OpenNet::Function * aFunction, const char * aCode, const char * aName, const char * aIndex)
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

    // aAdapterIndex
    void Test::SetKernel(unsigned int aAdapterIndex)
    {
        assert(ADAPTER_QTY > aAdapterIndex);

        assert(ADAPTER_QTY   > mAdapterCount0);
        assert(aAdapterIndex < mAdapterCount0);

        OpenNet::Adapter * lAdapter = mAdapters[aAdapterIndex];
        assert(NULL != lAdapter);

        OpenNet::Kernel * lK          = mKernels + aAdapterIndex;
        unsigned int      lOtherIndex = (aAdapterIndex + 1) % mAdapterCount0;

        switch (mCodes[aAdapterIndex])
        {
        case CODE_FORWARD       : SetKernel(lAdapter, lK, KERNEL_FORWARD       , mNos[lOtherIndex  ]); break;
        case CODE_NONE          :                                                                      break;
        case CODE_NOTHING       : SetKernel(lAdapter, lK, KERNEL_NOTHING       , NULL               ); break;
        case CODE_REPLY         : SetKernel(lAdapter, lK, KERNEL_FORWARD       , mNos[aAdapterIndex]); break;
        case CODE_REPLY_ON_ERROR: SetKernel(lAdapter, lK, KERNEL_REPLY_ON_ERROR, mNos[aAdapterIndex]); break;

        default: assert(false);
        }
    }

    // aAdapter [---;RW-]
    // aKernel  [-K-;RW-]
    // aCode    [---;R--]
    // aIndex   [---;R--]
    void Test::SetKernel(OpenNet::Adapter * aAdapter, OpenNet::Kernel * aKernel, const char * aCode, const char * aIndex)
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

    void Test::Uninit()
    {
        assert(             1 <= mAdapterCount0 );
        assert(ADAPTER_QTY    >= mAdapterCount0 );
        assert(mAdapterCount0 <= mAdapterCount1 );
        assert(ADAPTER_QTY    >= mAdapterCount1 );
        assert(             1 <= mGeneratorCount);
        assert(GENERATOR_QTY  >= mGeneratorCount);
        assert(NULL != mSystem);

        ResetInputFilters();

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
    }

    // aNode [---;R--]
    void Test::WriteResult(const char * aNote) const
    {
        assert(NULL != aNote);

        static const char * FILENAME = "K:\\Export\\OpenNet\\TestResults\\f02.txt";

        FILE * lFile;

        int lRet = _access(FILENAME, 2);
        if (0 == lRet)
        {
            lRet = fopen_s(&lFile, FILENAME, "a");
        }
        else
        {
            lRet = fopen_s(&lFile, FILENAME, "w");
            if (0 == lRet)
            {
                fprintf(lFile, "Name;MiB/s;Buffer;Code;Mode;bytes/packet;Profiling;MiB/s;packet/s;Note;Priority Class;TestLib\n");
            }
        }

        if (0 == lRet)
        {
            WriteResult(lFile, aNote);

            lRet = fclose(lFile);
            assert(0 == lRet);
        }
        else
        {
            printf(__FUNCTION__ " - fopen_s( , ,  ) returned %d\n", lRet);
        }
    }

    // aOut  [---;RW-]
    // aNode [---;RW-]
    void Test::WriteResult(FILE * aOut, const char * aNote) const
    {
        assert(NULL != aOut );
        assert(NULL != aNote);

        char lPriorityClass[32];

        switch (mPriorityClass)
        {
            #ifdef _KMS_WINDOWS_
                case ABOVE_NORMAL_PRIORITY_CLASS: strcpy_s(lPriorityClass, "ABOVE_NORMAL"); break;
                case BELOW_NORMAL_PRIORITY_CLASS: strcpy_s(lPriorityClass, "BELOW_NORMAL"); break;
                case HIGH_PRIORITY_CLASS        : strcpy_s(lPriorityClass, "HIGH"        ); break;
                case IDLE_PRIORITY_CLASS        : strcpy_s(lPriorityClass, "IDLE"        ); break;
                case NORMAL_PRIORITY_CLASS      : strcpy_s(lPriorityClass, "NORMAL"      ); break;
                case REALTIME_PRIORITY_CLASS    : strcpy_s(lPriorityClass, "REALTIME"    ); break;
            #endif

        default: sprintf_s(lPriorityClass, "0x%08x", mPriorityClass);
        }

        // TODO  OpenNet_Tool.Test
        //       Normal - Ajouter des informations : Le filtre des traces de
        //       debug du driver, la version du driver, la version de la DLL,
        //       la config du driver (Debug/Release), la config de la DLL
        //       (Debug/Release), la date de compilation de TestLib, la date
        //       de compilation de OpenNet.dll, la date de compilation du
        //       driver, la vitesse du lien ethernet, le type de carte
        //       graphique, nom de l'ordinateur, nom de l'utilisateur, la
        //       date et l'heure d'execution...

        // TODO  OpenNet_Tool.Test
        //       Normal - Limiter a une decimal le bandwith et a 0 decimal le
        //       nombre de paquet.

        fprintf(aOut, "%s;%.1f;%u;%s;%s;%u;%s;%.1f;%.1f;%s;%s;"
#ifdef _DEBUG
            "Debug"
#else
            "Release"
#endif
            " - Compiled at " __TIME__ " on " __DATE__ "\n",
            mName                     ,
            mConfig.mBandwidth_MiB_s  ,
            mConfig.mBufferQty        ,
            CODE_NAMES[mConfig.mCode] ,
            MODE_NAMES[mConfig.mMode] ,
            mConfig.mPacketSize_byte  ,
            mConfig.mProfiling ? "true" : "false",
            mResult.mBandwidth_MiB_s  ,
            mResult.mPacketThroughput_packet_s,
            aNote,
            lPriorityClass);
    }

}
