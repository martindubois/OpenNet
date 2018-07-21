
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/TestDual.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Function_Forward.h>
#include <OpenNet/PacketGenerator.h>
#include <OpenNet/System.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class TestDual
{

public:

    TestDual(unsigned int aBufferQty0, unsigned int aBufferQty1);

    ~TestDual();

    void DisplayAdapterStatistics();
    void DisplaySpeed            ();

    void GetAdapterStatistics();

    void GetAndDisplayKernelStatistics();

    void ResetAdapterStatistics();

    void Start();
    void Stop ();

    OpenNet::Adapter               * mAdapters [2]          ;
    OpenNet::Function_Forward        mFunctions[2]          ;
    OpenNet::PacketGenerator       * mPacketGenerator       ;
    OpenNet::PacketGenerator::Config mPacketGenerator_Config;

private:

    void Adapter_Connect();
    void Adapter_Get    ();

    void ResetInputFilter();

    void SetConfig     ();
    void SetInputFilter();
    void SetProcessor  ();

    unsigned int         mBufferQty [2]     ;
    OpenNet::Processor * mProcessor         ;
    unsigned int         mStatistics[2][128];
    OpenNet::System    * mSystem            ;

};
