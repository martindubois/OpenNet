
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

        ADAPTER_STATS_LOOP_BACK_PACKET              =  2,
        ADAPTER_STATS_PACKET_SEND                   =  3,

        ADAPTER_STATS_RESET_QTY = 4,

        ADAPTER_STATS_QTY = 32,
    }
    Adapter_Statistics;

};
