
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNetK/Adapter_Statistics.h
//
// This file defines the index of statistics counter for the
// OpenNetK::Adapter class.

// TODO  Common.OpenNetK.Adapter_Statistics
//       Add statistics about kernel execution time.

#pragma once

namespace OpenNetK
{

    // Data types
    /////////////////////////////////////////////////////////////////////////

    typedef enum
    {
        ADAPTER_STATS_BUFFERS_PROCESS      =  0,
        ADAPTER_STATS_BUFFER_INIT_HEADER   =  1,
        ADAPTER_STATS_BUFFER_QUEUE         =  2,
        ADAPTER_STATS_BUFFER_RECEIVE       =  3,
        ADAPTER_STATS_BUFFER_SEND          =  4,
        ADAPTER_STATS_BUFFER_SEND_PACKETS  =  5,
        ADAPTER_STATS_IOCTL                =  6,
        ADAPTER_STATS_IOCTL_CONFIG_GET     =  7,
        ADAPTER_STATS_IOCTL_CONFIG_SET     =  8,
        ADAPTER_STATS_IOCTL_CONNECT        =  9,
        ADAPTER_STATS_IOCTL_INFO_GET       = 10,
        ADAPTER_STATS_IOCTL_PACKET_SEND    = 11,
        ADAPTER_STATS_IOCTL_START          = 12,
        ADAPTER_STATS_IOCTL_STATE_GET      = 13,
        ADAPTER_STATS_IOCTL_STATISTICS_GET = 14,
        ADAPTER_STATS_IOCTL_STOP           = 15,
        ADAPTER_STATS_TX_packet            = 16,

        ADAPTER_STATS_RESET_QTY = 17,

        ADAPTER_STATS_IOCTL_LAST                 = 28,
        ADAPTER_STATS_IOCTL_LAST_RESULT          = 29,
        ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET = 30,
        ADAPTER_STATS_IOCTL_STATISTICS_RESET     = 31,

        ADAPTER_STATS_QTY = 32,
    }
    Adapter_Statistics;

};
