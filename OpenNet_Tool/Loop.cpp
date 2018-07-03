
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/Loop.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== Includes ===========================================================
#include <OpenNetK/Hardware_Statistics.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/OpenNet/Adapter_Statistics.h"
#include "../Common/OpenNetK/Adapter_Statistics.h"

// ===== OpenNet_Tool =======================================================
#include "Loop.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define HARDWARE_BASE (OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_QTY)

static const unsigned char PACKET[PACKET_SIZE_MAX_byte] =
{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88
};

// Public
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  CODE_ERROR
//                                 See Loop::Adapter_Connect
//                                 See Loop::Set_Processor
//                                 See Loop::AddDestination
//                                 See Loop::SetInputFilter
Loop::Loop(unsigned int aBufferQty, unsigned int aPacketSize_byte, unsigned int aPacketQty, Mode aMode) : mBufferQty(aBufferQty), mPacketSize_byte(aPacketSize_byte), mPacketQty(aPacketQty), mMode(aMode)
{
    assert(       0 < aBufferQty      );
    assert(       0 < aPacketSize_byte);
    assert(       0 < aPacketQty      );
    assert(MODE_QTY > aMode           );

    mSystem = OpenNet::System::Create();
    if (NULL == mSystem)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
            "OpenNet::System::Create() failed", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }

    Adapter_Get();

    mProcessor = mSystem->Processor_Get(0);
    if (NULL == mProcessor)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_FOUND,
            "This test need 1 processor", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }

    Adapter_Connect();
    SetProcessor   ();
    AddDestination ();
    SetConfig      ();
    SetInputFilter ();
}

Loop::~Loop()
{
    assert(NULL != mSystem);

    mSystem->Delete();
}

void Loop::Display()
{
    assert(0 < mPacketSize_byte);

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

        lRx_byte_s   [i] = lRx_packet_s[i] * mPacketSize_byte;
        lSum_packet_s[i] = lRx_packet_s[i] + lTx_packet_s[i] ;
        lTx_byte_s   [i] = lTx_packet_s[i] * mPacketSize_byte;

        lRx_byte_s   [i] /= 10.0;
        lRx_packet_s [i] /= 10.0;
        lSum_packet_s[i] /= 10.0;
        lTx_byte_s   [i] /= 10.0;
        lTx_packet_s [i] /= 10.0;

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
    printf("B/s\t\t%f\t%f\t%f\t%f\t%f\t%f\n"    , lRx_byte_s  [0], lTx_byte_s  [0], lSum_byte_s  [0], lRx_byte_s  [1], lTx_byte_s  [1], lSum_byte_s  [1]);
    printf("KiB/s\t\t%f\t%f\t%f\t%f\t%f\t%f\n"  , lRx_KiB_s   [0], lTx_KiB_s   [0], lSum_KiB_s   [0], lRx_KiB_s   [1], lTx_KiB_s   [1], lSum_KiB_s   [1]);
    printf("MiB/s\t\t%f\t%f\t%f\t%f\t%f\t%f\n"  , lRx_MiB_s   [0], lTx_MiB_s   [0], lSum_MiB_s   [0], lRx_MiB_s   [1], lTx_MiB_s   [1], lSum_MiB_s   [1]);
    printf("Rx/Tx\t\t%f %%\t\t\t\t\t%f %%\n"    , (lRx_byte_s[0] * 100.0) / lTx_byte_s[0]           , (lRx_byte_s[1] * 100.0) / lTx_byte_s[1]           );
    printf("\t\t%f %%\t\t\t\t\t%f %%\n"         , (lRx_byte_s[0] * 100.0) / lTx_byte_s[1]           , (lRx_byte_s[1] * 100.0) / lTx_byte_s[0]           );
}

// Exception  KmsLib::Exception *  CODE_ERROR
void Loop::GetAndDisplayStatistics()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        unsigned int lStats[16];

        OpenNet::Status lStatus = mFilters[i].GetStatistics(lStats, sizeof(lStats), NULL, true);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Filter::GetStatistics( , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }

        lStatus = mFilters[i].DisplayStatistics(lStats, sizeof(lStats), stdout, 0);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Filter::DisplayStatistics( , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void Loop::GetStatistics()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        OpenNet::Status lStatus = mAdapters[i]->GetStatistics(mStatistics[i], sizeof(mStatistics[i]), NULL, true);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::GetStats(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void Loop::ResetStatistics()
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
void Loop::SendPackets()
{
    assert(0 < mPacketQty      );
    assert(0 < mPacketSize_byte);

    for (unsigned int i = 0; i < mPacketQty; i++)
    {
        for (unsigned int j = 0; j < 2; j++)
        {
            assert(NULL != mAdapters[j]);

            OpenNet::Status lStatus = mAdapters[j]->Packet_Send(PACKET, mPacketSize_byte);
            if (OpenNet::STATUS_OK != lStatus)
            {
                throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                    "Adapter::Packet_Send( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
            }
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void Loop::Start()
{
    assert(NULL != mSystem);

    OpenNet::Status lStatus = mSystem->Start();
    if (OpenNet::STATUS_OK != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
            "System::Start(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
//                                 See Loop::Buffer_Release
//                                 See Loop::ResetInputFilter
void Loop::Stop()
{
    assert(NULL != mSystem);

    OpenNet::Status lStatus = mSystem->Stop(OpenNet::System::STOP_FLAG_LOOPBACK);
    if (OpenNet::STATUS_OK != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
            "System::Stop(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    ResetInputFilter();
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  CODE_ERROR
void Loop::Adapter_Connect()
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
void Loop::Adapter_Get()
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
void Loop::AddDestination()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        OpenNet::Status lStatus;
        
        switch (mMode)
        {
        case MODE_DOUBLE_CIRCLE: lStatus = mFilters[i].AddDestination(mAdapters[(i + 1) % 2]); break;
        case MODE_DOUBLE_MIRROR: lStatus = mFilters[i].AddDestination(mAdapters[ i         ]); break;

        default: assert(false);
        }

        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Filter::AddDestination(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void Loop::ResetInputFilter()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        OpenNet::Status lStatus = mAdapters[i]->ResetInputFilter();
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::ResetInputFilter(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void Loop::SetConfig()
{
    assert(0 < mBufferQty);

    for (unsigned int i = 0; i < 2; i++)
    {
        OpenNet::Adapter * lAdapter = mAdapters[i];

        assert(NULL != lAdapter);

        OpenNet::Adapter::Config lConfig;

        OpenNet::Status lStatus = lAdapter->GetConfig(&lConfig);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::GetConfig(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }

        lConfig.mBufferQty = mBufferQty;

        lAdapter->SetConfig(lConfig);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::SetConfig(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void Loop::SetInputFilter()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        OpenNet::Status lStatus = mFilters[i].EnableProfiling();
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Filter::EnableProfiling() failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }

        lStatus = mAdapters[i]->SetInputFilter(mFilters + i);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::SetInputFilter(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// Exception  KmsLib::Exception *  CODE_ERROR
void Loop::SetProcessor()
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
