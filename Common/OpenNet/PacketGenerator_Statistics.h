
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNet/PacketGenerator_Statistics.h
//
// This file defines the index of the statistics counter for
// OpenNet::PacketGenerator.

#pragma once

namespace OpenNet
{

    // Data types
    /////////////////////////////////////////////////////////////////////////

    typedef enum
    {
        PACKET_GENERATOR_STATS_NO_PACKET_cycle       = 0,
        PACKET_GENERATOR_STATS_RUN_ENTRY             = 1,
        PACKET_GENERATOR_STATS_RUN_EXIT              = 2,
        PACKET_GENERATOR_STATS_SEND_cycle            = 3,
        PACKET_GENERATOR_STATS_SEND_ERROR_cycle      = 4,
        PACKET_GENERATOR_STATS_SENT_packet           = 5,
        PACKET_GENERATOR_STATS_TOO_MANY_PACKET_cycle = 6,

        PACKET_GENERATOR_STATS_RESET_QTY = 7,

        PACKET_GENERATOR_STATS_STATISTICS_RESET = 15,

        PACKET_GENERATOR_STATS_QTY = 16,
    }
    PacketGenerator_Statistics;

};
