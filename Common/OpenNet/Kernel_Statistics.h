
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNet/Kernel_Statistics.h
//
// This file defines the index of the statistics counter for OpenNet::Filter.

#pragma once

namespace OpenNet
{

    // Data types
    /////////////////////////////////////////////////////////////////////////

    typedef enum
    {
        KERNEL_STATS_EXECUTION                 = 0,
        KERNEL_STATS_EXECUTION_DURATION_AVG_us = 1,
        KERNEL_STATS_EXECUTION_DURATION_MAX_us = 2,
        KERNEL_STATS_EXECUTION_DURATION_MIN_us = 3,
        KERNEL_STATS_QUEUE_DURATION_AVG_us     = 4,
        KERNEL_STATS_QUEUE_DURATION_MAX_us     = 5,
        KERNEL_STATS_QUEUE_DURATION_MIN_us     = 6,
        KERNEL_STATS_SUBMIT_DURATION_AVG_us    = 7,
        KERNEL_STATS_SUBMIT_DURATION_MAX_us    = 8,
        KERNEL_STATS_SUBMIT_DURATION_MIN_us    = 9,

        KERNEL_STATS_RESET_QTY = 10,

        KERNEL_STATS_EXECUTION_DURATION_LAST_us = 13,
        KERNEL_STATS_QUEUE_DURATION_LAST_us     = 14,
        KERNEL_STATS_SUBMIT_DURATION_LAST_us    = 15,

        KERNEL_STATS_QTY = 16,
    }
    Kernel_Statistics;

};
