
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/Test.h

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
#include <OpenNet/Adapter.h>
#include <OpenNet/Filter_Forward.h>

// ===== OpenNet_Tool =======================================================
#include "Test.h"

// Static functions declaration
/////////////////////////////////////////////////////////////////////////////

static void Loop_AddDestination  (OpenNet::Filter_Forward * aF, OpenNet::Adapter ** aA);
static void Loop_Adapter_Connect (OpenNet::System * aSystem, OpenNet::Adapter ** aA);
static void Loop_Adapter_Get     (OpenNet::System * aSystem, OpenNet::Adapter ** aA);
static void Loop_Buffer_Allocate (OpenNet::Adapter ** aA, unsigned int aBufferQty);
static void Loop_Buffer_Release  (OpenNet::Adapter ** aA, unsigned int aBufferQty);
static void Loop_Display         (const OpenNet::Adapter::Stats * aStats, unsigned int aPacketSize_byte);
static void Loop_GetStats        (OpenNet::Adapter ** aA, OpenNet::Adapter::Stats * aStats);
static void Loop_SendPackets     (OpenNet::Adapter ** aA, unsigned int aPacketSize_byte, unsigned int aPacketQty);
static void Loop_ResetInputFilter(OpenNet::Adapter ** aA);
static void Loop_ResetStats      (OpenNet::Adapter ** aA);
static void Loop_SetInputFilter  (OpenNet::Adapter ** aA, OpenNet::Filter_Forward * aF);
static void Loop_SetProcessor    (OpenNet::Adapter ** aA, OpenNet::Processor * aP0);

// Functions
/////////////////////////////////////////////////////////////////////////////

// aSystem [---;RW-]
//
// Exception  KmsLib::Exception *  CODE_ERROR
//                                 CODE_NOT_FOUND
//                                 See Loop_AddDestination
//                                 See Loop_Adapter_Connect
//                                 See Loop_Adapter_Get
//                                 See Loop_Buffer_Allocate
//                                 See Loop_Buffer_Release
//                                 See Loop_GetStats
//                                 See Loop_ResetInputFilter
//                                 See Loop_ResetStats
//                                 See Loop_SendPackets
//                                 See Loop_SetInputFilter
//                                 See Loop_SetProcessor
void Test_Loop(OpenNet::System * aSystem, unsigned int aBufferQty, unsigned int aPacketSize_byte, unsigned int aPacketQty)
{
    OpenNet::Adapter      * lA [2];
    OpenNet::Filter_Forward lFF[2];

    Loop_Adapter_Get(aSystem, lA);

    printf("Retriving processor...\n");
    OpenNet::Processor * lP0 = aSystem->Processor_Get(0);
    if (NULL == lP0)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_FOUND,
            "This test need 1 processor", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }

    Loop_Adapter_Connect(aSystem, lA);
    Loop_SetProcessor   (lA, lP0);
    Loop_AddDestination (lFF, lA);
    Loop_SetInputFilter (lA, lFF);
    Loop_Buffer_Allocate(lA, aBufferQty);

    printf("Starting...\n");
    OpenNet::Status lStatus = aSystem->Start();
    if (OpenNet::STATUS_OK != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
            "System::Start(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    Sleep(2000);

    Loop_SendPackets(lA, aPacketSize_byte, aPacketQty);

    printf("Stabilizing...\n");
    Sleep(2000);

    Loop_ResetStats(lA);

    printf("Running...\n");
    Sleep(10000);

    OpenNet::Adapter::Stats lStats[2];

    Loop_GetStats(lA, lStats);
    Loop_Display (lStats, aPacketSize_byte);

    printf("Stopping...\n");
    lStatus = aSystem->Stop(OpenNet::System::STOP_FLAG_LOOPBACK);
    if (OpenNet::STATUS_OK != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
            "System::Stop(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
    }

    Loop_Buffer_Release  (lA, aBufferQty);
    Loop_ResetInputFilter(lA);
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aF [---;RW-]
// aA [---;R--]
//
// Exception  KmsLib::Exception *  CODE_
void Loop_AddDestination(OpenNet::Filter_Forward * aF, OpenNet::Adapter ** aA)
{
    assert(NULL != aF);
    assert(NULL != aA);

    printf("Adding destination...\n");

    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != aA[i]);

        OpenNet::Status lStatus = aF[i].AddDestination(aA[(i + 1) % 2]);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Filter::AddDestination(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// aA [---;R--]
//
// Exception  KmsLib::Exception *  CODE_ERROR
void Loop_Adapter_Connect(OpenNet::System * aSystem, OpenNet::Adapter ** aA)
{
    assert(NULL != aSystem);
    assert(NULL != aA     );

    printf("Connecting adapters...\n");

    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != aA[i]);

        OpenNet::Status lStatus = aSystem->Adapter_Connect(aA[i]);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "System::Adapter_Connect(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// aA [---;-W-]
//
// Exception  KmsLib::Exception *  CODE_NOT_FOUND
void Loop_Adapter_Get(OpenNet::System * aSystem, OpenNet::Adapter ** aA)
{
    assert(NULL != aSystem);
    assert(NULL != aA     );

    printf("Retrieving adapters...\n");

    for (unsigned int i = 0; i < 2; i++)
    {
        aA[i] = aSystem->Adapter_Get(i);
        if (NULL == aA[i])
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_FOUND,
                "This test need 2 adapters", NULL, __FILE__, __FUNCTION__, __LINE__, i);
        }
    }
}

// aA [---;R--]
//
// Exception  KmsLib::Exception *  CODE_
void Loop_Buffer_Allocate(OpenNet::Adapter ** aA, unsigned int aBufferQty)
{
    assert(NULL != aA        );
    assert(   0 <  aBufferQty);

    printf("Allocating buffers...\n");

    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != aA[i]);

        OpenNet::Status lStatus = aA[i]->Buffer_Allocate(aBufferQty);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::Buffer_Allocate(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// aA [---;R--]
//
// Exception  KmsLib::Exception *  CODE_
void Loop_Buffer_Release(OpenNet::Adapter ** aA, unsigned int aBufferQty)
{
    assert(NULL != aA        );
    assert(   0 <  aBufferQty);

    printf("Releasing buffers...\n");

    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != aA[i]);

        OpenNet::Status lStatus = aA[i]->Buffer_Release(aBufferQty);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::Buffer_Release(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// aStats [---;R--]
//
// Exception  KmsLib::Exception *  CODE_
void Loop_Display(const OpenNet::Adapter::Stats * aStats, unsigned int aPacketSize_byte)
{
    assert(NULL != aStats);

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
        lRx_packet_s[i] = aStats[i].mDriver.mHardware.mRx_Packet;
        lTx_packet_s[i] = aStats[i].mDriver.mHardware.mTx_Packet;

        lRx_byte_s   [i] = lRx_packet_s[i] * aPacketSize_byte;
        lSum_packet_s[i] = lRx_packet_s[i] + lTx_packet_s[i] ;
        lTx_byte_s   [i] = lTx_packet_s[i] * aPacketSize_byte;

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

// aA     [---;R--]
// aStats [---;-W-]
//
// Exception  KmsLib::Exception *  CODE_
void Loop_GetStats(OpenNet::Adapter ** aA, OpenNet::Adapter::Stats * aStats)
{
    assert(NULL != aA    );
    assert(NULL != aStats);

    printf("Retrieving statistics...\n");

    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != aA[i]);

        OpenNet::Status lStatus = aA[i]->GetStats(aStats + i);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::GetStats(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// aA [---;R--]
//
// Exception  KmsLib::Exception *  CODE_
void Loop_SendPackets(OpenNet::Adapter ** aA, unsigned int aPacketSize_byte, unsigned int aPacketQty)
{
    assert(NULL != aA              );
    assert(0    <  aPacketSize_byte);
    assert(0    <  aPacketQty      );

    printf("Sending packets...\n");

    unsigned char * lPacket = new unsigned char[aPacketSize_byte];
    assert(NULL != lPacket);

    memset(lPacket, 0xff, aPacketSize_byte);

    for (unsigned int i = 0; i < aPacketQty; i++)
    {
        for (unsigned int j = 0; j < 2; j++)
        {
            assert(NULL != aA[j]);

            OpenNet::Status lStatus = aA[j]->Packet_Send(lPacket, aPacketSize_byte);
            if (OpenNet::STATUS_OK != lStatus)
            {
                delete[] lPacket;
                throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                    "Adapter::Packet_Send( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
            }
        }
    }
}

// aA [---;R--]
//
// Exception  KmsLib::Exception *  CODE_ERROR
void Loop_ResetInputFilter(OpenNet::Adapter ** aA)
{
    assert(NULL != aA);

    printf("Reseting input filter...\n");

    for (unsigned int i = 0; i < 2; i++)
    {
        OpenNet::Status lStatus = aA[i]->ResetInputFilter();
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::ResetInputFilter(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// aA [---;R--]
//
// Exception  KmsLib::Exception *  CODE_
void Loop_ResetStats(OpenNet::Adapter ** aA)
{
    assert(NULL != aA);

    printf("Reseting statistics...\n");

    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != aA[i]);

        OpenNet::Status lStatus = aA[i]->ResetStats();
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::Reset_State() failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// aA [---;R--]
// aF [---;---]
//
// Exception  KmsLib::Exception *  CODE_
void Loop_SetInputFilter(OpenNet::Adapter ** aA, OpenNet::Filter_Forward * aF)
{
    assert(NULL != aA);
    assert(NULL != aF);

    printf("Setting input filter...\n");

    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != aA[i]);

        OpenNet::Status lStatus = aA[i]->SetInputFilter(aF + i);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::SetInputFilter(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}

// aA  [---;R--]
// aP0 [---;---]
//
// Exception  KmsLib::Exception *  CODE_ERROR
void Loop_SetProcessor(OpenNet::Adapter ** aA, OpenNet::Processor * aP0)
{
    assert(NULL != aA );
    assert(NULL != aP0);

    printf("Setting processor...\n");

    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != aA[i]);

        OpenNet::Status lStatus = aA[i]->SetProcessor(aP0);
        if (OpenNet::STATUS_OK != lStatus)
        {
            throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
                "Adapter::SetProcessor(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lStatus);
        }
    }
}
