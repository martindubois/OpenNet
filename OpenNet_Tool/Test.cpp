
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

// ===== Includes ===========================================================
#include <OpenNet/PacketGenerator.h>

// ===== Common =============================================================
#include "../Common/TestLib/TestDual.h"

// ===== OpenNet_Tool =======================================================
#include "Test.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

void Test_A(unsigned int aBufferQty, unsigned int aPacketSize_byte, unsigned int aBandwidth_MiB_s)
{
    assert(0 < aBufferQty      );
    assert(0 < aPacketSize_byte);
    assert(0 < aBandwidth_MiB_s);

    TestLib::TestDual lTD(aBufferQty, 2, false);

    lTD.mPacketGenerator_Config.mBandwidth_MiB_s = aBandwidth_MiB_s;
    lTD.mPacketGenerator_Config.mPacketSize_byte = aPacketSize_byte;

    OpenNet::Status lStatus = lTD.mPacketGenerator->SetAdapter(lTD.mAdapters[1]);
    assert(OpenNet::STATUS_OK == lStatus);

    lTD.Adapter_SetInputFunction(0);
    lTD.Start                   ();

    Sleep(100);

    lTD.ResetAdapterStatistics();

    Sleep(1000);

    lTD.GetAdapterStatistics    ();
    lTD.DisplayAdapterStatistics();
    lTD.DisplaySpeed            ();

    lTD.Stop();
}

// Internal   Ethernet   Internal
//
//     +---   <-------   <--- Generator
//     |    0          1
//     +-->   ------->   ---> Dropped
void Test_B(unsigned int aBufferQty, unsigned int aPacketSize_byte, unsigned int aBandwidth_MiB_s)
{
    assert(0 < aBufferQty);
    assert(0 < aPacketSize_byte);
    assert(0 < aBandwidth_MiB_s);

    TestLib::TestDual lTD(aBufferQty, aBufferQty, false);

    lTD.mPacketGenerator_Config.mBandwidth_MiB_s = aBandwidth_MiB_s;
    lTD.mPacketGenerator_Config.mPacketSize_byte = aPacketSize_byte;

    OpenNet::Status lStatus = lTD.mPacketGenerator->SetAdapter(lTD.mAdapters[1]);
    assert(OpenNet::STATUS_OK == lStatus);

    lStatus = lTD.mFunctions[0].AddDestination(lTD.mAdapters[0]);
    assert(OpenNet::STATUS_OK == lStatus);

    lTD.Adapter_SetInputFunctions();
    lTD.Start                    ();

    Sleep(100);

    lTD.ResetAdapterStatistics();

    Sleep(1000);

    lTD.GetAdapterStatistics    ();
    lTD.DisplayAdapterStatistics();
    lTD.DisplaySpeed            ();

    lTD.Stop();
}

