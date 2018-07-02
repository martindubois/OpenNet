
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNet/Filter_Statistics.h
//
// This file defines the index of the statistics counter for OpenNet::Filter.

#pragma once

namespace OpenNet
{

    // Data types
    /////////////////////////////////////////////////////////////////////////

    typedef enum
    {
        FILTER_STATS_EXECUTION                 = 0,
        FILTER_STATS_EXECUTION_DURATION_AVG_us = 1,
        FILTER_STATS_EXECUTION_DURATION_MAX_us = 2,
        FILTER_STATS_EXECUTION_DURATION_MIN_us = 3,
        FILTER_STATS_QUEUE_DURATION_AVG_us     = 4,
        FILTER_STATS_QUEUE_DURATION_MAX_us     = 5,
        FILTER_STATS_QUEUE_DURATION_MIN_us     = 6,
        FILTER_STATS_SUBMIT_DURATION_AVG_us    = 7,
        FILTER_STATS_SUBMIT_DURATION_MAX_us    = 8,
        FILTER_STATS_SUBMIT_DURATION_MIN_us    = 9,

        FILTER_STATS_RESET_QTY = 10,

        FILTER_STATS_EXECUTION_DURATION_LAST_us = 13,
        FILTER_STATS_QUEUE_DURATION_LAST_us     = 14,
        FILTER_STATS_SUBMIT_DURATION_LAST_us    = 15,

        FILTER_STATS_QTY = 16,
    }
    Filter_Statistics;

};
