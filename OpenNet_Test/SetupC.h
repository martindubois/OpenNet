
// Author   KMA - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/SetupC.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Filter_Forward.h>
#include <OpenNet/System.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class SetupC
{

public:

    SetupC(unsigned int aBufferQty);

    ~SetupC();

    int Init ();
    int Start();
    int Stop (unsigned int aFlags);

    int Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aCount);

    int Stats_Get();
    int Stats_GetAndDisplay();
    int Stats_Reset();

    OpenNet::Adapter      * mAdapters[2];
    unsigned int            mBufferQty  ;
    OpenNet::Filter_Forward mFilters [2];
    OpenNet::Processor    * mProcessor  ;
    OpenNet::Adapter::Stats mStats   [2];
    OpenNet::System       * mSystem     ;

};
