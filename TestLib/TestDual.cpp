
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     TestLib/TestDual.cpp

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

#include "../Common/TestLib/TestDual.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define EOL "\n"

static const char * CODE =
"#include <OpenNetK/Kernel.h>"                                                EOL
                                                                              EOL
"OPEN_NET_KERNEL_DECLARE"                                                     EOL
"{"                                                                           EOL
"    OPEN_NET_KERNEL_BEGIN"                                                   EOL
                                                                              EOL
"        __global unsigned char * lData   = lBase + lPacketInfo->mOffset_byte;" EOL
"        unsigned int             lResult = OPEN_NET_PACKET_PROCESSED;"       EOL
"        unsigned int             i;"                                         EOL
                                                                              EOL
"        for ( i = 0; i < 6; i ++)"                                           EOL
"        {"                                                                   EOL
"            if ( 0xff != lData[ i ] )"                                       EOL
"            {"                                                               EOL
"                lResult |= 1;"                                               EOL
"            }"                                                               EOL
"        }"                                                                   EOL
                                                                              EOL
"        for ( i = 6; i < 12; i ++)"                                          EOL
"        {"                                                                   EOL
"            if ( 0x00 != lData[ i ] )"                                       EOL
"            {"                                                               EOL
"                lResult |= 1;"                                               EOL
"            }"                                                               EOL
"        }"                                                                   EOL
                                                                              EOL
"        for ( i = 12; i < 14; i ++)"                                         EOL
"        {"                                                                   EOL
"            if ( 0x88 != lData[ i ] )"                                       EOL
"            {"                                                               EOL
"                lResult |= 1;"                                               EOL
"            }"                                                               EOL
"        }"                                                                   EOL
                                                                              EOL
"        for ( i = 14; i < lPacketInfo->mSize_byte; i ++)"                    EOL
"        {"                                                                   EOL
"            if ( 0x00 != lData[ i ] )"                                       EOL
"            {"                                                               EOL
"                lResult |= 1;"                                               EOL
"            }"                                                               EOL
"        }"                                                                   EOL
                                                                              EOL
"        lPacketInfo->mSendTo = lResult;"                                     EOL
                                                                              EOL
"    OPEN_NET_KERNEL_END"                                                     EOL
"}"                                                                           EOL;


namespace TestLib
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    // aProfiling   Enable profiling at processor level
    TestDual::TestDual(Mode aMode, bool aProfiling) : mMode(aMode), mPacketGenerator(NULL), mProcessor(NULL), mProfiling(aProfiling), mSystem(NULL)
    {
        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            mAdapters [i] = NULL;
            mBufferQty[i] =    8;
        }

        mKernel.SetCode(CODE, static_cast<unsigned int>(strlen(CODE)));
    }

    TestDual::~TestDual()
    {
        if (NULL != mSystem)
        {
            Uninit();
        }
    }

    unsigned int TestDual::A(unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s, AdapterSelect aSelect)
    {
        A_Init(aBufferQty, aSelect);

        mPacketGenerator_Config.mBandwidth_MiB_s = aBandwidth_MiB_s;
        mPacketGenerator_Config.mPacketSize_byte = aPacketSize_byte;

        OpenNet::Status lStatus = mPacketGenerator->SetAdapter(mAdapters[1]);
        if (OpenNet::STATUS_OK != lStatus)
        {
            printf(__FUNCTION__ " - PacketGenerator::SetAdapter(  ) returned %u\n", lStatus);
            return __LINE__;
        }

        switch (mMode)
        {
        case MODE_FUNCTION: Adapter_SetInputFunction(0); break;
        case MODE_KERNEL  : Adapter_SetInputKernel  (0); break;

        default: assert(false);
        }
        
        Start();

        Sleep(100);

        ResetAdapterStatistics();

        Sleep(1000);

        GetAdapterStatistics();

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 12674;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =    53;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 22300;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =    78;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 22300;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =    78;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 22300;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =    78;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1162;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 13363;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =   153;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 1430000;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =    4992;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 1430000;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =    4992;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin =   1 * 1024 * 1024;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 1430000;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =   13870;

        unsigned int lResult = Adapter_VerifyStatistics(0);

        Stop  ();
        Uninit();

        return lResult;
    }

    unsigned int TestDual::A_Search(unsigned int aBufferQty, unsigned int aPacketSize_byte, AdapterSelect aSelect)
    {
        assert(0 < aBufferQty      );
        assert(0 < aPacketSize_byte);

        double lMin_MiB_s =   0.1;
        double lMax_MiB_s = 120.0;
        double lCenter_MiB_s;

        while ((lMin_MiB_s + 0.1) < lMax_MiB_s)
        {
            lCenter_MiB_s = (lMax_MiB_s + lMin_MiB_s) / 2.0;

            printf("Search  %f MiB/s\n", lCenter_MiB_s);

            if (0 >= A(aBufferQty, aPacketSize_byte, lCenter_MiB_s, aSelect))
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

    unsigned int TestDual::A_Verify(unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s, AdapterSelect aSelect)
    {
        double lBandwidth_MiB_s = aBandwidth_MiB_s;

        for (unsigned int i = 0; i < 5; i++)
        {
            printf("Verify  %f MiB/s\n", lBandwidth_MiB_s);

            if (0 < A(aBufferQty, aPacketSize_byte, lBandwidth_MiB_s, aSelect))
            {
                lBandwidth_MiB_s -= 0.1;
                i = 0;
            }
        }

        return 0;
    }

    unsigned int TestDual::B(unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s, AdapterSelect aSelect)
    {
        B_Init(aBufferQty, aSelect);

        mPacketGenerator_Config.mBandwidth_MiB_s = aBandwidth_MiB_s;
        mPacketGenerator_Config.mPacketSize_byte = aPacketSize_byte;

        OpenNet::Status lStatus = mPacketGenerator->SetAdapter(mAdapters[1]);
        assert(OpenNet::STATUS_OK == lStatus);

        switch (mMode)
        {
        case MODE_FUNCTION:
            lStatus = mForwardFs[0].AddDestination(mAdapters[0]);
            assert(OpenNet::STATUS_OK == lStatus);

            Adapter_SetInputFunctions();
            break;

        case MODE_KERNEL:
            lStatus = mForwardKs[0].AddDestination(mAdapters[0]);
            assert(OpenNet::STATUS_OK == lStatus);

            Adapter_SetInputKernels();
            break;

        default: assert(false);
        }
        Start();

        Sleep(100);

        ResetAdapterStatistics();

        Sleep(1000);

        GetAdapterStatistics();

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10026;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1075;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 5252;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =  149;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 5252;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =  149;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 10372;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   158;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1003;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMax = 336126;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_TX_packet].mMin =   9536;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10019;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  4219;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 336128;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =   9536;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 336126;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin =   9536;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 335846;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =   9531;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 336126;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin =   9530;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 85 * 1024 * 1024;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 15 * 1024 * 1024;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 665399;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =   9531;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 85 * 1024 * 1024;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin = 15 * 1024 * 1024;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 336126;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin =   9530;

        unsigned int lResult = Adapter_VerifyStatistics(0);

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 10027;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin =  1840;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 5181;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =  149;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 5181;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =  149;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 10372;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =   158;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMax = 665112;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_PACKET_SEND].mMin =  14027;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1002;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10019;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  4219;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 331584;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin =   9536;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMax = 665112;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_SEND].mMin =  14027;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 332212;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin =   9536;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMax = 665540;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet].mMin =   9537;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 85 * 1024 * 1024;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 15 * 1024 * 1024;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 336126;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin =   9537;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMax = 85 * 1024 * 1024;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte].mMin = 15 * 1024 * 1024;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMax = 665420;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet].mMin =   9537;

        lResult += Adapter_VerifyStatistics(1);

        Stop();
        Uninit();

        return lResult;
    }

    unsigned int TestDual::B_Search(unsigned int aBufferQty, unsigned int aPacketSize_byte, AdapterSelect aSelect)
    {
        assert(0 < aBufferQty      );
        assert(0 < aPacketSize_byte);

        double lMin_MiB_s =   0.1;
        double lMax_MiB_s = 120.0;
        double lCenter_MiB_s;

        while ((lMin_MiB_s + 0.1) < lMax_MiB_s)
        {
            lCenter_MiB_s = (lMax_MiB_s + lMin_MiB_s) / 2.0;

            printf("Search  %f MiB/s\n", lCenter_MiB_s);

            if (0 >= B(aBufferQty, aPacketSize_byte, lCenter_MiB_s, aSelect))
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

    unsigned int TestDual::B_Verify(unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s, AdapterSelect aSelect)
    {
        double lBandwidth_MiB_s = aBandwidth_MiB_s;

        for (unsigned int i = 0; i < 5; i++)
        {
            printf("Verify  %f MiB/s\n", lBandwidth_MiB_s);

            if (0 < B(aBufferQty, aPacketSize_byte, lBandwidth_MiB_s, aSelect))
            {
                lBandwidth_MiB_s -= 0.1;
                i = 0;
            }
        }

        return 0;
    }

    unsigned int TestDual::C(unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s, AdapterSelect aSelect)
    {
        A_Init(aBufferQty, aSelect);

        mPacketGenerator_Config.mBandwidth_MiB_s = aBandwidth_MiB_s;
        mPacketGenerator_Config.mPacketSize_byte = aPacketSize_byte;

        OpenNet::Status lStatus = mPacketGenerator->SetAdapter(mAdapters[1]);
        if (OpenNet::STATUS_OK != lStatus)
        {
            printf(__FUNCTION__ " - PacketGenerator::SetAdapter(  ) returned %u\n", lStatus);
            return __LINE__;
        }

        Adapter_SetInputKernel_C(0);

        Start();

        Sleep(100);

        ResetAdapterStatistics();

        Sleep(1000);

        GetAdapterStatistics();

        Adapter_InitialiseConstraints();

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMax = 9840;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFERS_PROCESS].mMin = 9800;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMax = 3740;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_RECEIVE].mMin =  217;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMax = 3740;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND].mMin =  217;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMax = 3740;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_BUFFER_SEND_PACKETS].mMin =  217;

        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMax = 1010;
        mConstraints[TestLib::TestDual::ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms].mMin = 1000;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMax = 10100;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_INTERRUPT_PROCESS].mMin =  9940;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMax = 23900;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_PACKET_RECEIVE].mMin = 13800;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMax = 23900;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet].mMin = 13800;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMax = 121 * 1024 * 1024;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte].mMin = 1 * 1024 * 1024;

        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMax = 23900;
        mConstraints[TestLib::TestDual::HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet].mMin = 13800;

        unsigned int lResult = Adapter_VerifyStatistics(0);

        Stop();
        Uninit();

        return lResult;
    }

    double TestDual::Adapter_GetBandwidth() const
    {
        return mBandwidth_MiB_s;
    }

    double TestDual::Adapter_GetPacketThroughput() const
    {
        return mPacketThroughput;
    }

    void TestDual::DisplaySpeed()
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

        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            double lDuration_s = static_cast<double>(mStatistics[i][ADAPTER_BASE + OpenNetK::ADAPTER_STATS_RUNNING_TIME_ms]) / 1000.0; // ms ==> s

            lRx_byte_s  [i] = mStatistics[i][HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_byte  ];
            lRx_packet_s[i] = mStatistics[i][HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_HOST_packet];
            lTx_byte_s  [i] = mStatistics[i][HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_byte  ];
            lTx_packet_s[i] = mStatistics[i][HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_HOST_packet];

            lSum_packet_s[i] = lRx_packet_s[i] + lTx_packet_s[i] ;

            lRx_byte_s   [i] /= lDuration_s;
            lRx_packet_s [i] /= lDuration_s;
            lSum_packet_s[i] /= lDuration_s;
            lTx_byte_s   [i] /= lDuration_s;
            lTx_packet_s [i] /= lDuration_s;

            lRx_KiB_s[i] = lRx_byte_s[i] / 1024;
            lRx_MiB_s[i] = lRx_KiB_s [i] / 1024;
            lTx_KiB_s[i] = lTx_byte_s[i] / 1024;
            lTx_MiB_s[i] = lTx_KiB_s [i] / 1024;

            lSum_MiB_s [i] = lRx_MiB_s [i] + lTx_MiB_s [i];
        }

        printf("\t\tAdapter 0\t\t\t\t\tAdapters 1\n");
        printf("\t\tRx\t\tTx\t\tTotal\t\tRx\t\tTx\t\tTotal\n");
        printf("Packets/s\t%f\t%f\t%f\t%f\t%f\t%f\n", lRx_packet_s[0], lTx_packet_s[0], lSum_packet_s[0], lRx_packet_s[1], lTx_packet_s[1], lSum_packet_s[1]);
        printf("MiB/s\t\t%f\t%f\t%f\t%f\t%f\t%f\n"  , lRx_MiB_s   [0], lTx_MiB_s   [0], lSum_MiB_s   [0], lRx_MiB_s   [1], lTx_MiB_s   [1], lSum_MiB_s   [1]);

        mBandwidth_MiB_s  = lRx_MiB_s   [0];
        mPacketThroughput = lRx_packet_s[0];
    }

    // TODO TestLib.DualTest
    //      Replacer les exception par un retour de numero de ligne

    // Exception  KmsLib::Exception *  CODE_ERROR
    void TestDual::Start()
    {
        assert(NULL != mSystem);

        OpenNet::Status lStatus = mPacketGenerator->SetConfig(mPacketGenerator_Config);
        assert(OpenNet::STATUS_OK == lStatus);

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

        lStatus = mPacketGenerator->Start();
        assert(OpenNet::STATUS_OK == lStatus);
    }

    void TestDual::Stop()
    {
        assert(NULL != mSystem);

        OpenNet::Status lStatus = mSystem->Stop();
        assert(OpenNet::STATUS_OK == lStatus);

        lStatus = mPacketGenerator->Stop();
        assert(OpenNet::STATUS_OK == lStatus);

        ResetInputFilter();
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    unsigned int TestDual::A_Init(unsigned int aBufferQty, AdapterSelect aSelect)
    {
        mBufferQty[0] = aBufferQty;
        mBufferQty[1] =          2;

        return Init(aSelect);
    }

    unsigned int TestDual::B_Init(unsigned int aBufferQty, AdapterSelect aSelect)
    {
        mBufferQty[0] = aBufferQty;
        mBufferQty[1] = aBufferQty;

        return Init(aSelect);
    }

    unsigned int TestDual::Init(AdapterSelect aSelect)
    {
        mPacketGenerator = OpenNet::PacketGenerator::Create();
        assert(NULL != mPacketGenerator);

        mSystem = OpenNet::System::Create();
        assert(NULL != mSystem);

        Adapter_Get(aSelect);

        mProcessor = mSystem->Processor_Get(0);
        if (NULL == mProcessor)
        {
            printf(__FUNCTION__ " - System::Processor_Get() failed\n");
            return __LINE__;
        }

        OpenNet::Status lStatus = mPacketGenerator->GetConfig(&mPacketGenerator_Config);
        assert(OpenNet::STATUS_OK == lStatus);

        if (mProfiling)
        {
            Processor_EnableProfiling();
        }

        lStatus = mForwardFs[0].SetFunctionName("Function_0");
        assert(OpenNet::STATUS_OK == lStatus);

        lStatus = mForwardFs[1].SetFunctionName("Function_1");
        assert(OpenNet::STATUS_OK == lStatus);

        Adapter_Connect();
        SetProcessor   ();
        SetConfig      ();

        return 0;
    }

    unsigned int TestDual::Uninit()
    {
        assert(NULL != mPacketGenerator);
        assert(NULL != mSystem         );

        mPacketGenerator->Delete();
        mSystem         ->Delete();

        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            mAdapters[i] = NULL;
        }

        mPacketGenerator = NULL;
        mProcessor       = NULL;
        mSystem          = NULL;

        return 0;
    }

    // Exception  KmsLib::Exception *  CODE_ERROR
    void TestDual::Adapter_Connect()
    {
        assert(NULL != mSystem);

        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
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

    // aOrder [--O;R--]
    //
    // Exception  KmsLib::Exception *  CODE_NOT_FOUND
    void TestDual::Adapter_Get(AdapterSelect aSelect)
    {
        assert(NULL != mSystem);

        OpenNet::Adapter::Info lInfo[2];

        mAdapters[0] = mSystem->Adapter_Get(0);
        if (NULL == mAdapters[0])
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_FOUND,
                "First adapter not found", NULL, __FILE__, __FUNCTION__, __LINE__, aSelect);
        }

        OpenNet::Status lStatus = mAdapters[0]->GetInfo(lInfo + 0);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::GetInfo(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }

        bool lFound = false;

        for (unsigned int i = 1; ! lFound; i++)
        {
            mAdapters[1] = mSystem->Adapter_Get(i);
            if (NULL == mAdapters[i])
            {
                throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_FOUND,
                    "Second adapter not found", NULL, __FILE__, __FUNCTION__, __LINE__, i);
            }

            lStatus = mAdapters[1]->GetInfo(lInfo + 1);
            if (OpenNet::STATUS_OK != lStatus)
            {
                throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                    "Adapter::GetInfo(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
            }

            switch (aSelect)
            {
            case ADAPTER_SELECT_CARD_DIFF: lFound = (0 != memcmp(lInfo[0].mEthernetAddress.mAddress, lInfo[1].mEthernetAddress.mAddress, 5)); break;
            case ADAPTER_SELECT_CARD_SAME: lFound = (0 == memcmp(lInfo[0].mEthernetAddress.mAddress, lInfo[1].mEthernetAddress.mAddress, 5)); break;

            default: assert(false);
            }
        }
    }

    void TestDual::Adapter_InitialiseConstraints()
    {
        KmsLib::ValueVector::Constraint_Init(mConstraints, STATS_QTY);

        mConstraints[ADAPTER_BASE + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_RESET].mMax = 0xffffffff;
    }

    void TestDual::Adapter_SetInputFunctions()
    {
        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->SetInputFilter(mForwardFs + i);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void TestDual::Adapter_SetInputFunction(unsigned int aAdapter)
    {
        assert(ADAPTER_QTY > aAdapter);

        assert(NULL != mAdapters[aAdapter]);

        OpenNet::Status lStatus = mAdapters[aAdapter]->SetInputFilter(mForwardFs + aAdapter);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    void TestDual::Adapter_SetInputKernels()
    {
        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->SetInputFilter(mForwardKs + i);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void TestDual::Adapter_SetInputKernel(unsigned int aAdapter)
    {
        assert(ADAPTER_QTY > aAdapter);

        assert(NULL != mAdapters[aAdapter]);

        OpenNet::Status lStatus = mAdapters[aAdapter]->SetInputFilter(mForwardKs + aAdapter);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    void TestDual::Adapter_SetInputKernel_C(unsigned int aAdapter)
    {
        assert(ADAPTER_QTY > aAdapter);

        assert(NULL != mAdapters[aAdapter]);

        OpenNet::Status lStatus = mAdapters[aAdapter]->SetInputFilter(&mKernel);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    unsigned int TestDual::Adapter_VerifyStatistics(unsigned int aAdapter)
    {
        assert(ADAPTER_QTY > aAdapter);

        OpenNet::Adapter * lAdapter = mAdapters[aAdapter];
        assert(NULL != lAdapter);

        return KmsLib::ValueVector::Constraint_Verify(mStatistics[aAdapter], lAdapter->GetStatisticsQty(), mConstraints, stdout, reinterpret_cast<const KmsLib::ValueVector::Description *>(lAdapter->GetStatisticsDescriptions()));
    }

    void TestDual::GetAdapterStatistics()
    {
        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->GetStatistics(mStatistics[i], sizeof(mStatistics[i]), NULL, true);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void TestDual::Processor_EnableProfiling()
    {
        assert(NULL != mProcessor);

        OpenNet::Processor::Config lConfig;

        OpenNet::Status lStatus = mProcessor->GetConfig(&lConfig);
        assert(OpenNet::STATUS_OK == lStatus);

        lConfig.mFlags.mProfilingEnabled = true;

        lStatus = mProcessor->SetConfig(lConfig);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    void TestDual::ResetAdapterStatistics()
    {
        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->ResetStatistics();
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void TestDual::ResetInputFilter()
    {
        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->ResetInputFilter();
            assert((OpenNet::STATUS_OK == lStatus) || (OpenNet::STATUS_FILTER_NOT_SET == lStatus));
        }
    }

    // Exception  KmsLib::Exception *  CODE_ERROR
    void TestDual::SetConfig()
    {
        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
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

    void TestDual::SetProcessor()
    {
        assert(NULL != mProcessor);

        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->SetProcessor(mProcessor);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

}
