

// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Hardware_Statistics.h
///
/// This file define the index of statistics counters for the
/// OpenNetK::Hardware class.

#pragma once

namespace OpenNetK
{

    // Data types
    /////////////////////////////////////////////////////////////////////////

    typedef enum
    {
        HARDWARE_STATS_D0_ENTRY           =  0,
        HARDWARE_STATS_D0_EXIT            =  1,
        HARDWARE_STATS_INTERRUPT_DISABLE  =  2,
        HARDWARE_STATS_INTERRUPT_ENABLE   =  3,
        HARDWARE_STATS_INTERRUPT_PROCESS  =  4,
        HARDWARE_STATS_INTERRUPT_PROCESS2 =  5,
        HARDWARE_STATS_PACKET_RECEIVE     =  6,
        HARDWARE_STATS_PACKET_SEND        =  7,
        HARDWARE_STATS_RX_packet          =  8,
        HARDWARE_STATS_SET_CONFIG         =  9,
        HARDWARE_STATS_STATISTICS_GET     = 10,
        HARDWARE_STATS_TX_packet          = 11,

        HARDWARE_STATS_RX_BMC_MANAGEMENT_DROPPED_packet       = 36,
        HARDWARE_STATS_RX_CIRCUIT_BREAKER_DROPPED_packet      = 37,
        HARDWARE_STATS_RX_LENGTH_ERRORS_packet                = 38,
        HARDWARE_STATS_RX_MANAGEMENT_DROPPED_packet           = 39,
        HARDWARE_STATS_RX_MISSED_packet                       = 40,
        HARDWARE_STATS_RX_NO_BUFFER_packet                    = 41,
        HARDWARE_STATS_RX_OVERSIZE_packet                     = 42,
        HARDWARE_STATS_RX_UNDERSIZE_packet                    = 43,
        HARDWARE_STATS_TX_DEFER_EVENTS                        = 44,
        HARDWARE_STATS_TX_DISCARDED_packet                    = 45,
        HARDWARE_STATS_TX_NO_CRS_packet                       = 46,
        HARDWARE_STATS_TX_HOST_CIRCUIT_BREAKER_DROPPED_packet = 47,

        HARDWARE_STATS_RESET_QTY = 48,

        HARDWARE_STATS_INTERRUPT_PROCESS_LAST_MESSAGE_ID = 61,
        HARDWARE_STATS_STATISTICS_GET_RESET              = 62,
        HARDWARE_STATS_STATISTICS_RESET                  = 63,

        HARDWARE_STATS_QTY = 64,
    }
    Hardware_Statistics;

};
