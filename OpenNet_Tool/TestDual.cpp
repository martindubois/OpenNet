
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/TestDual.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== Includes ===========================================================
#include <OpenNet/Kernel.h>
#include <OpenNetK/Hardware_Statistics.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/OpenNet/Adapter_Statistics.h"
#include "../Common/OpenNetK/Adapter_Statistics.h"

// ===== OpenNet_Tool =======================================================
#include "TestDual.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define HARDWARE_BASE (OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_QTY)

// Public
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  CODE_NOT_FOUND
//                                 See Loop::Adapter_Connect
//                                 See Loop::Set_Processor
//                                 See Loop::AddDestination
//                                 See Loop::SetInputFilter
TestDual::TestDual(unsigned int aBufferQty0, unsigned int aBufferQty1)
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

    OpenNet::Processor::Config lConfig;

    lStatus = mProcessor->GetConfig(&lConfig);
    assert(OpenNet::STATUS_OK == lStatus);

    lConfig.mFlags.mProfilingEnabled = true;

    lStatus = mProcessor->SetConfig(lConfig);
    assert(OpenNet::STATUS_OK == lStatus);

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

void TestDual::DisplayAdapterStatistics()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        OpenNet::Status lStatus = mAdapters[i]->DisplayStatistics(mStatistics[i], sizeof(mStatistics[i]), stdout, 0);
        assert(OpenNet::STATUS_OK == lStatus);
    }
}

void TestDual::DisplaySpeed(double aDuration_s)
{
    assert(0 < mPacketGenerator_Config.mPacketSize_byte);

    double lRx_byte_s   [2];
    double lRx_KiB_s    [2];
    double lRx_MiB_s    [2];
    double lRx_packet_s [2];
    double lSum_byte_s  [2];
    double lSum_KiB_s   [2];
    double lSum_MiB_s   [2];
    double lSum_packet_s[2];
    double lTx_byte_s   [2];
    double lTx_KiB_s    [2];
    double lTx_MiB_s    [2];
    double lTx_packet_s [2];

    for (unsigned int i = 0; i < 2; i++)
    {
        lRx_packet_s[i] = mStatistics[i][HARDWARE_BASE + OpenNetK::HARDWARE_STATS_RX_packet];
        lTx_packet_s[i] = mStatistics[i][HARDWARE_BASE + OpenNetK::HARDWARE_STATS_TX_packet];

        lRx_byte_s   [i] = lRx_packet_s[i] * mPacketGenerator_Config.mPacketSize_byte;
        lSum_packet_s[i] = lRx_packet_s[i] + lTx_packet_s[i] ;
        lTx_byte_s   [i] = lTx_packet_s[i] * mPacketGenerator_Config.mPacketSize_byte;

        lRx_byte_s   [i] /= aDuration_s;
        lRx_packet_s [i] /= aDuration_s;
        lSum_packet_s[i] /= aDuration_s;
        lTx_byte_s   [i] /= aDuration_s;
        lTx_packet_s [i] /= aDuration_s;

        lRx_KiB_s[i] = lRx_byte_s[i] / 1024;
        lRx_MiB_s[i] = lRx_KiB_s [i] / 1024;
        lTx_KiB_s[i] = lTx_byte_s[i] / 1024;
        lTx_MiB_s[i] = lTx_KiB_s [i] / 1024;

        lSum_byte_s[i] = lRx_byte_s[i] + lTx_byte_s[i];
        lSum_KiB_s [i] = lRx_KiB_s [i] + lTx_KiB_s [i];
        lSum_MiB_s [i] = lRx_MiB_s [i] + lTx_MiB_s [i];
    }

    printf("\t\tAdapter 0\t\t\t\t\tAdapters 1\n");
    printf("\t\tRx\t\tTx\t\tTotal\t\tRx\t\tTx\t\tTotal\n");
    printf("Packets/s\t%f\t%f\t%f\t%f\t%f\t%f\n", lRx_packet_s[0], lTx_packet_s[0], lSum_packet_s[0], lRx_packet_s[1], lTx_packet_s[1], lSum_packet_s[1]);
//    printf("B/s\t\t%f\t%f\t%f\t%f\t%f\t%f\n"    , lRx_byte_s  [0], lTx_byte_s  [0], lSum_byte_s  [0], lRx_byte_s  [1], lTx_byte_s  [1], lSum_byte_s  [1]);
//    printf("KiB/s\t\t%f\t%f\t%f\t%f\t%f\t%f\n"  , lRx_KiB_s   [0], lTx_KiB_s   [0], lSum_KiB_s   [0], lRx_KiB_s   [1], lTx_KiB_s   [1], lSum_KiB_s   [1]);
    printf("MiB/s\t\t%f\t%f\t%f\t%f\t%f\t%f\n"  , lRx_MiB_s   [0], lTx_MiB_s   [0], lSum_MiB_s   [0], lRx_MiB_s   [1], lTx_MiB_s   [1], lSum_MiB_s   [1]);
//    printf("Rx/Tx\t\t%f %%\t\t\t\t\t%f %%\n"    , (lRx_byte_s[0] * 100.0) / lTx_byte_s[0]           , (lRx_byte_s[1] * 100.0) / lTx_byte_s[1]           );
//    printf("\t\t%f %%\t\t\t\t\t%f %%\n"         , (lRx_byte_s[0] * 100.0) / lTx_byte_s[1]           , (lRx_byte_s[1] * 100.0) / lTx_byte_s[0]           );
}

void TestDual::GetAdapterStatistics()
{
    for (unsigned int i = 0; i < 2; i++)
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

// Exception  KmsLib::Exception *  CODE_ERROR
void TestDual::ResetAdapterStatistics()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        OpenNet::Status lStatus = mAdapters[i]->ResetStatistics();
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::Reset_State() failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void TestDual::Start()
{
    assert(NULL != mSystem);

    OpenNet::Status lStatus = mPacketGenerator->SetConfig(mPacketGenerator_Config);
    assert(OpenNet::STATUS_OK == lStatus);

    SetInputFilter();

    lStatus = mSystem->Start();
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

    lStatus = mPacketGenerator->Start();
    assert(OpenNet::STATUS_OK == lStatus);
}

// Exception  KmsLib::Exception *  See Loop::ResetInputKernel
void TestDual::Stop()
{
    assert(NULL != mSystem);

    OpenNet::Status lStatus = mPacketGenerator->Stop();
    assert(OpenNet::STATUS_OK == lStatus);

    lStatus = mSystem->Stop(OpenNet::System::STOP_FLAG_LOOPBACK);
    assert(OpenNet::STATUS_OK == lStatus);

    ResetInputFilter();
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  CODE_ERROR
void TestDual::Adapter_Connect()
{
    assert(NULL != mSystem);

    for (unsigned int i = 0; i < 2; i++)
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

    for (unsigned int i = 0; i < 2; i++)
    {
        mAdapters[i] = mSystem->Adapter_Get(i);
        if (NULL == mAdapters[i])
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_FOUND,
                "This test need 2 adapters", NULL, __FILE__, __FUNCTION__, __LINE__, i);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void TestDual::ResetInputFilter()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        OpenNet::Status lStatus = mAdapters[i]->ResetInputFilter();
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::ResetInputKernel(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void TestDual::SetConfig()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(0 < mBufferQty[i]);

        OpenNet::Adapter * lAdapter = mAdapters[i];

        assert(NULL != lAdapter);

        OpenNet::Adapter::Config lConfig;

        OpenNet::Status lStatus = lAdapter->GetConfig(&lConfig);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::GetConfig(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }

        lConfig.mBufferQty = mBufferQty[i];

        lAdapter->SetConfig(lConfig);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::SetConfig(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void TestDual::SetInputFilter()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        OpenNet::Status lStatus = mAdapters[i]->SetInputFilter(mFunctions + i);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::SetInputFilter(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void TestDual::SetProcessor()
{
    assert(NULL != mProcessor);

    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        OpenNet::Status lStatus = mAdapters[i]->SetProcessor(mProcessor);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::SetProcessor(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}
