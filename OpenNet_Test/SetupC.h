
// Author   KMA - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/SetupC.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsLib/ValueVector.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Kernel_Forward.h>
#include <OpenNet/System.h>

// ===== OpenNet_Test =======================================================
#include "Base.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class SetupC : public Base
{

public:

    SetupC(unsigned int aBufferQty);

    ~SetupC();

    int Init ();
    int Start();
    int Stop (unsigned int aFlags);

    int Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aCount);

    int Statistics_Get          ();
    int Statistics_GetAndDisplay(unsigned int aMinLevel = 0);
    int Statistics_Reset        ();
    int Statistics_Verify       (unsigned int aAdapter);

    OpenNet::Adapter      * mAdapters[2];
    unsigned int            mBufferQty  ;
    OpenNet::Kernel_Forward mKernels [2];
    OpenNet::Processor    * mProcessor  ;

    unsigned int mStatistics[2][STATISTICS_QTY];

};
