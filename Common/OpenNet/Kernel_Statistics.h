
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNet/Kernel_Statistics.h
//
// This file defines the index of the statistics counter for OpenNet::Kernel.

// CODE REVIEW  2019-07-16  KMS - Martin Dubois, ing.

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
        // 1.0.0

        KERNEL_STATS_BINARY_VERSION               = 23,
        KERNEL_STATS_CACHE_MODE_CA                = 24,
        KERNEL_STATS_CONST_SIZE_byte              = 25,
        KERNEL_STATS_LOCAL_SIZE_byte              = 26,
        KERNEL_STATS_MAX_DYNAMIC_SHARED_SIZE_byte = 27,
        KERNEL_STATS_MAX_THREADS_PER_BLOCK        = 28,
        KERNEL_STATS_NUM_REGS                     = 29,
        KERNEL_STATS_PTX_VERSION                  = 30,
        KERNEL_STATS_SHARED_SIZE_byte             = 31,

        KERNEL_STATS_QTY = 32,
    }
    Kernel_Statistics;

};
