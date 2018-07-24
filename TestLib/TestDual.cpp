
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

namespace TestLib
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    // aBufferQty0  Buffer quantity for adapter 0
    // aBufferQty1  Buffer quantity for adapter 1
    // aProfiling   Enable profiling at processor level
    //
    // Exception  KmsLib::Exception *  CODE_NOT_FOUND
    //                                 See TestDual::Adapter_Connect
    //                                 See TestDual::SetConfig
    TestDual::TestDual(unsigned int aBufferQty0, unsigned int aBufferQty1, bool aProfiling)
    {
        assert(0 < aBufferQty0);
        assert(0 < aBufferQty1);

        mBufferQty[0] = aBufferQty0;
        mBufferQty[1] = aBufferQty1;

        mPacketGenerator = OpenNet::PacketGenerator::Create();
        assert(NULL != mPacketGenerator);

        mSystem = OpenNet::System::Create();
        assert(NULL != mSystem);

        Adapter_Get();

        mProcessor = mSystem->Processor_Get(0);
        if (NULL == mProcessor)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_FOUND,
                "This test need 1 processor", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
        }

        OpenNet::Status lStatus = mPacketGenerator->GetConfig(&mPacketGenerator_Config);
        assert(OpenNet::STATUS_OK == lStatus);

        if (aProfiling)
        {
            Processor_EnableProfiling();
        }

        lStatus = mFunctions[0].SetFunctionName("Function_0");
        assert(OpenNet::STATUS_OK == lStatus);

        lStatus = mFunctions[1].SetFunctionName("Function_1");
        assert(OpenNet::STATUS_OK == lStatus);

        Adapter_Connect();
        SetProcessor   ();
        SetConfig      ();
    }

    TestDual::~TestDual()
    {
        assert(NULL != mPacketGenerator);
        assert(NULL != mSystem        );

        mPacketGenerator->Delete();
        mSystem         ->Delete();
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

            OpenNet::Status lStatus = mAdapters[i]->SetInputFilter(mFunctions + i);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void TestDual::Adapter_SetInputFunction(unsigned int aAdapter)
    {
        assert(ADAPTER_QTY > aAdapter);

        assert(NULL != mAdapters[aAdapter]);

        OpenNet::Status lStatus = mAdapters[aAdapter]->SetInputFilter(mFunctions + aAdapter);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    void TestDual::Adapter_SetInputKernels()
    {
        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->SetInputFilter(mKernels + i);
            assert(OpenNet::STATUS_OK == lStatus);
        }
    }

    void TestDual::Adapter_SetInputKernel(unsigned int aAdapter)
    {
        assert(ADAPTER_QTY > aAdapter);

        assert(NULL != mAdapters[aAdapter]);

        OpenNet::Status lStatus = mAdapters[aAdapter]->SetInputFilter(mKernels + aAdapter);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    unsigned int TestDual::Adapter_VerifyStatistics(unsigned int aAdapter)
    {
        assert(ADAPTER_QTY > aAdapter);

        OpenNet::Adapter * lAdapter = mAdapters[aAdapter];
        assert(NULL != lAdapter);

        return KmsLib::ValueVector::Constraint_Verify(mStatistics[aAdapter], lAdapter->GetStatisticsQty(), mConstraints, stdout, reinterpret_cast<const KmsLib::ValueVector::Description *>( lAdapter->GetStatisticsDescriptions()));
    }

    void TestDual::DisplayAdapterStatistics()
    {
        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->DisplayStatistics(mStatistics[i], sizeof(mStatistics[i]), stdout, 0);
            assert(OpenNet::STATUS_OK == lStatus);
        }
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

    void TestDual::GetAndDisplayKernelStatistics()
    {
        unsigned int lStats[16];

        OpenNet::Kernel * lKernel = mSystem->Kernel_Get( 0 );
        assert(NULL != lKernel);

        OpenNet::Status lStatus = lKernel->GetStatistics(lStats, sizeof(lStats), NULL, true);
        assert(OpenNet::STATUS_OK == lStatus);

        lStatus = lKernel->DisplayStatistics(lStats, sizeof(lStats), stdout, 0);
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

    // Exception  KmsLib::Exception *  CODE_NOT_FOUND
    void TestDual::Adapter_Get()
    {
        assert(NULL != mSystem);

        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            mAdapters[i] = mSystem->Adapter_Get(i);
            if (NULL == mAdapters[i])
            {
                throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_FOUND,
                    "This test need 2 adapters", NULL, __FILE__, __FUNCTION__, __LINE__, i);
            }
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

    void TestDual::ResetInputFilter()
    {
        for (unsigned int i = 0; i < ADAPTER_QTY; i++)
        {
            assert(NULL != mAdapters[i]);

            OpenNet::Status lStatus = mAdapters[i]->ResetInputFilter();
            assert(OpenNet::STATUS_OK == lStatus);
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
