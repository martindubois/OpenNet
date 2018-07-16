
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/PacketGenerator.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Windows ============================================================
#include <Windows.h>

// ===== Import/Includes ====================================================
#include <KmsLib/ValueVector.h>

// ===== Common =============================================================
#include "../Common/OpenNet/PacketGenerator_Statistics.h"

// ===== OpenNet ============================================================
#include "PacketGenerator_Internal.h"

// Constants
////////////////////////////////////////////////////////////////////////////

static const OpenNet::StatisticsProvider::StatisticsDescription STATISTICS_DESCRIPTIONS[] =
{
    { "RUN - ENTRY            ", NULL     , 0 }, // 0
    { "RUN - EXIT             ", NULL     , 0 },
    { "SEND                   ", "cycles" , 0 },
    { "SEND                   ", "packets", 1 },
    { "SEND - ERROR           ", NULL     , 1 },

    VALUE_VECTOR_DESCRIPTION_RESERVED, //  5
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 10
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "STATISTICS - RESET (NR)", NULL, 0 }, // 15
};

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             PacketGenerator_Internal contructor raise an exception
    PacketGenerator * PacketGenerator::Create()
    {
        PacketGenerator * lResult;

        try
        {
            lResult = new PacketGenerator_Internal();
        }
        catch (...)
        {
            lResult = NULL;
        }

        return lResult;
    }

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             System_Internal destructor raise an exception
    void PacketGenerator::Delete()
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

    PacketGenerator::PacketGenerator() : StatisticsProvider(STATISTICS_DESCRIPTIONS, PACKET_GENERATOR_STATS_QTY)
    {
    }

    PacketGenerator::~PacketGenerator()
    {
    }

}
