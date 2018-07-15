
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/System.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>
#include <KmsLib/ValueVector.h>

// ===== Includes ===========================================================
#include <OpenNet/System.h>

// ===== Common =============================================================
#include "../Common/OpenNet/System_Statistics.h"

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"
#include "System_Internal.h"

// Constants
////////////////////////////////////////////////////////////////////////////

static const OpenNet::StatisticsProvider::StatisticsDescription STATISTICS_DESCRIPTIONS[] =
{
    { "EXECUTION                  ", ""  , 0 }, //  0
    { "EXECUTION - DURATION - AVG ", "us", 1 },
    { "EXECUTION - DURATION - MAX ", "us", 1 },
    { "EXECUTION - DURATION - MIN ", "us", 1 },
    { "QUEUE     - DURATION - AVG ", "us", 1 },
    { "QUEUE     - DURATION - MAX ", "us", 1 }, //  5
    { "QUEUE     - DURATION - MIN ", "us", 1 },
    { "SUBMIT    - DURATION - AVG ", "us", 1 },
    { "SUBMIT    - DURATION - MAX ", "us", 1 },
    { "SUBMIT    - DURATION - MIN ", "us", 1 },

    VALUE_VECTOR_DESCRIPTION_RESERVED, // 10
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "EXECUTION - DURATION - LAST", "us", 0 },
    { "QUEUE     - DURATION - LAST", "us", 0 },
    { "SUBMIT    - DURATION - LAST", "us", 0 }, // 15
};

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    const unsigned int System::STOP_FLAG_LOOPBACK = 0x00000001;

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             System_Internal contructor raise an exception
    System * System::Create()
    {
        System * lResult;

        try
        {
            lResult = new System_Internal();
        }
        catch ( ... )
        {
            lResult = NULL;
        }

        return lResult;
    }

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             System_Internal destructor raise an exception
    void System::Delete()
    {
        try
        {
            delete this;
        }
        catch (...)
        {
        }
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    System::System() : StatisticsProvider(STATISTICS_DESCRIPTIONS, SYSTEM_STATS_QTY)
    {
    }

    System::~System()
    {
    }

}
