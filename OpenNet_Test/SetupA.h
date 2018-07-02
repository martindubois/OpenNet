
// Author   KMA - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/SetupA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsLib/ValueVector.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Filter_Forward.h>
#include <OpenNet/System.h>
#include <OpenNetK/Hardware_Statistics.h>

// ===== Common =============================================================
#include "../Common/OpenNet/Adapter_Statistics.h"
#include "../Common/OpenNetK/Adapter_Statistics.h"

// ===== OpenNet_Test =======================================================
#include "Base.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class SetupA : public Base
{

public:

    enum
    {
        STATISTICS_QTY = (OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_QTY + OpenNetK::HARDWARE_STATS_QTY),
    };

    SetupA(unsigned int aBufferQty);

    ~SetupA();

    int Init ();
    int Start();
    int Stop (unsigned int aFlags);

    int Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aCount);

    int Statistics_Get          ();
    int Statistics_GetAndDisplay(unsigned int aMinLevel = 0);
    int Statistics_Reset        ();
    int Statistics_Verify       (const KmsLib::ValueVector::Constraint_UInt32 * aConstraints);

    OpenNet::Adapter      * mAdapter  ;
    unsigned int            mBufferQty;
    OpenNet::Filter_Forward mFilter   ;
    OpenNet::Processor    * mProcessor;

    unsigned int mStatistics[STATISTICS_QTY];

};
