
// Author   KMA - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Base.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsLib/ValueVector.h>

// ===== Includes ===========================================================
#include <OpenNet/System.h>
#include <OpenNetK/Hardware_Statistics.h>

// ===== Common =============================================================
#include "../Common/OpenNet/Adapter_Statistics.h"
#include "../Common/OpenNetK/Adapter_Statistics.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Base
{

public:

    enum
    {
        STATISTICS_QTY = (OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_QTY + OpenNetK::HARDWARE_STATS_QTY),
    };

    Base();

    ~Base();

    int Init();

    void Constraint_Init();

    KmsLib::ValueVector::Constraint_UInt32 mConstraints[STATISTICS_QTY];

    OpenNet::System * mSystem;

};
