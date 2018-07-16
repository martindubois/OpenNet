
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
        PACKET_GENERATOR_STATS_RUN_ENTRY   = 0,
        PACKET_GENERATOR_STATS_RUN_EXIT    = 1,
        PACKET_GENERATOR_STATS_SEND_cycle  = 2,
        PACKET_GENERATOR_STATS_SEND_packet = 3,
        PACKET_GENERATOR_STATS_SEND_ERROR  = 4,

        PACKET_GENERATOR_STATS_RESET_QTY = 5,

        PACKET_GENERATOR_STATS_STATISTICS_RESET = 15,

        PACKET_GENERATOR_STATS_QTY = 16,
    }
    PacketGenerator_Statistics;

};
