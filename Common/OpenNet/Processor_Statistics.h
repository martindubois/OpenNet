
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNet/Processor_Statistics.h
//
// This file defines the index of the statistics counter for
// OpenNet::Processor.

#pragma once

namespace OpenNet
{

    // Data types
    /////////////////////////////////////////////////////////////////////////

    typedef enum
    {
        PROCESSOR_STATS_RESET_QTY = 1,

        PROCESSOR_STATS_QTY = 16,
    }
    Processor_Statistics;

};
