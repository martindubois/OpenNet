
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNet/Adapter_Statistics.h
//
// This file containt the index of statistics counter for the
// OpenNet::Adapter class.

#pragma once

namespace OpenNet
{

    // Data types
    /////////////////////////////////////////////////////////////////////////

    typedef enum
    {
        ADAPTER_STATS_BUFFER_ALLOCATED              =  0,
        ADAPTER_STATS_BUFFER_RELEASED               =  1,
        ADAPTER_STATS_LOOP_BACK_PACKET              =  2,
        ADAPTER_STATS_PACKET_SEND                   =  3,
        ADAPTER_STATS_RUN_ENTRY                     =  4,
        ADAPTER_STATS_RUN_EXCEPTION                 =  5,
        ADAPTER_STATS_RUN_EXIT                      =  6,
        ADAPTER_STATS_RUN_ITERATION_QUEUE           =  7,
        ADAPTER_STATS_RUN_ITERATION_WAIT            =  8,
        ADAPTER_STATS_RUN_LOOP_EXCEPTION            =  9,
        ADAPTER_STATS_RUN_LOOP_UNEXPECTED_EXCEPTION = 10,
        ADAPTER_STATS_RUN_QUEUE                     = 11,
        ADAPTER_STATS_RUN_UNEXPECTED_EXCEPTION      = 12,
        ADAPTER_STATS_START                         = 13,
        ADAPTER_STATS_STOP_REQUEST                  = 14,
        ADAPTER_STATS_STOP_WAIT                     = 15,

        ADAPTER_STATS_RESET_QTY = 16,

        ADAPTER_STATS_QTY = 32,
    }
    Adapter_Statistics;

};
