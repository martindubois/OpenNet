
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/ValueVector.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNetK/Constants.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/IoCtl.h"

// ===== OpenNet ============================================================
#include "EthernetAddress.h"
#include "VersionInfo.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static const char * ADAPTER_TYPE_NAMES[OpenNetK::ADAPTER_TYPE_QTY] =
{
    "UNKNOWN" ,
    "ETHERNET",
    "NDIS"    ,
};

const OpenNet::StatisticsProvider::StatisticsDescription STATISTICS_DESCRIPTIONS[] =
{
    { "OpenNet::Adapter - BUFFER_ALLOCATED         ", "", 0 }, //  0

    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "OpenNet::Adapter - LOOP_BACK_PACKET         ", "", 0 },
    { "OpenNet::Adapter - PACKET_SEND              ", "", 1 },

    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, //  5
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 10
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 15
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 20
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 25
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 30
    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "OpenNetK::Adapter - BUFFERS - PROCESS       ", ""      , 0 },
    { "OpenNetK::Adapter - BUFFER - INIT_HEADER    ", ""      , 0 },
    { "OpenNetK::Adapter - BUFFER - QUEUE          ", ""      , 0 },
    { "OpenNetK::Adapter - BUFFER - RECEIVE        ", ""      , 1 }, // 35
    { "OpenNetK::Adapter - BUFFER - SEND           ", ""      , 1 },
    { "OpenNetK::Adapter - BUFFER - SEND_PACKETS   ", ""      , 1 },

    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "OpenNetK::Adapter - IOCTL - CONFIG - GET    ", ""      , 0 },
    { "OpenNetK::Adapter - IOCTL - CONFIG - SET    ", ""      , 0 }, // 40
    { "OpenNetK::Adapter - IOCTL - CONNECT         ", ""      , 0 },
    { "OpenNetK::Adapter - IOCTL - INFO_GET        ", ""      , 0 },
    { "OpenNetK::Adapter - IOCTL - PACKET_SEND     ", ""      , 1 },
    { "OpenNetK::Adapter - IOCTL - START           ", ""      , 0 },
    { "OpenNetK::Adapter - IOCTL - STATE_GET       ", ""      , 0 }, // 45
    { "OpenNetK::Adapter - IOCTL - STATISTICS - GET", ""      , 0 },
    { "OpenNetK::Adapter - IOCTL - STOP            ", ""      , 0 },
    { "OpenNetK::Adapter - RUNNING_TIME            ", "ms"    , 1 },
    { "OpenNetK::Adapter - TX                      ", "packet", 1 },

    VALUE_VECTOR_DESCRIPTION_RESERVED, // 50
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 55
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 60
    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "OpenNetK::Adapter - IOCTL - STATISTICS - GET - RESET (NR)", "", 0 },
    { "OpenNetK::Adapter - IOCTL - STATISTICS - RESET       (NR)", "", 0 },

    { "OpenNetK::Hardware - D0 - ENTRY             ", ""      , 0 },
    { "OpenNetK::Hardware - D0 - EXIT              ", ""      , 0 }, // 65
    { "OpenNetK::Hardware - INTERRUPT - DISABLE    ", ""      , 0 },
    { "OpenNetK::Hardware - INTERRUPT - ENABLE     ", ""      , 0 },
    { "OpenNetK::Hardware - INTERRUPT - PROCESS    ", ""      , 1 },

    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "OpenNetK::Hardware - PACKET - RECEIVE       ", ""      , 1 }, // 70
    { "OpenNetK::Hardware - PACKET - SEND          ", ""      , 1 },
    { "OpenNetK::Hardware - RX                     ", "packet", 1 },
    { "OpenNetK::Hardware - SET_CONFIG             ", ""      , 0 },
    { "OpenNetK::Hardware - STATISTICS - GET       ", ""      , 0 },
    { "OpenNetK::Hardware - TX                     ", "packet", 1 }, // 75

    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 80
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 85
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 90

    { "Hardware - RX - BMC_MANAGEMENT_DROPPED      ", "packet", 1 },
    { "Hardware - RX - CIRCUIT_BREAKER_DROPPED     ", "packet", 1 },
    { "Hardware - RX - HOST                        ", "byte"  , 1 },
    { "Hardware - RX - HOST                        ", "packet", 1 },
    { "Hardware - RX - LENGTH_ERRORS               ", "packet", 1 }, //  95
    { "Hardware - RX - MANAGEMENT_DROPPED          ", "packet", 1 },
    { "Hardware - RX - MISSED                      ", "packet", 1 },
    { "Hardware - RX - NO_BUFFER                   ", "packet", 0 },
    { "Hardware - RX - OVERSIZE                    ", "packet", 1 },
    { "Hardware - RX - QUEUE_DROPPED               ", "packet", 1 }, // 100
    { "Hardware - RX - UNDERSIZE                   ", "packet", 1 },
    { "Hardware - RX - XOFF                        ", "packet", 0 },
    { "Hardware - RX - XON                         ", "packet", 0 },
    { "Hardware - TX - DEFER_EVENTS                ", ""      , 0 },
    { "Hardware - TX - DISCARDED                   ", "packet", 1 }, // 105
    { "Hardware - TX - HOST                        ", "byte"  , 1 },
    { "Hardware - TX - HOST                        ", "packet", 1 },
    { "Hardware - TX - HOST_CIRCUIT_BREAKER_DROPPED", "packet", 1 },
    { "Hardware - TX - NO_CRS                      ", "packet", 1 },
    { "Hardware - TX - XOFF                        ", "packet", 0 }, // 110
    { "Hardware - TX - XON                         ", "packet", 0 },

    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 115
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 120
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "HARDWARE - INTERRUPT_PROCESS_LAST_MESSAGE_ID (NR)", "", 0 }, // 125

    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
};

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Status Adapter::Display(const Adapter::Config & aIn, FILE * aOut)
    {
        if (NULL == (&aIn)) { return STATUS_INVALID_REFERENCE        ; }
        if (NULL ==   aOut) { return STATUS_NOT_ALLOWED_NULL_ARGUMENT; }

        fprintf(aOut, "  Adapter::Config :\n");

        if ((OPEN_NET_BUFFER_QTY >= aIn.mBufferQty) && (0 < aIn.mBufferQty))
        {
            fprintf(aOut, "    Buffer Quantity = %u\n", aIn.mBufferQty);
        }
        else
        {
            fprintf(aOut, "    Buffer Quantity = %u <== ERROR  Invalid value\n", aIn.mBufferQty);
        }

        if ((PACKET_SIZE_MAX_byte >= aIn.mPacketSize_byte) && (PACKET_SIZE_MIN_byte <= aIn.mPacketSize_byte))
        {
            fprintf(aOut, "    Packet Size     = %u bytes\n", aIn.mPacketSize_byte);
        }
        else
        {
            fprintf(aOut, "    Packet Size     = %u bytes <== ERROR  Invalid value\n", aIn.mPacketSize_byte);
        }

        return STATUS_OK;
    }

    Status Adapter::Display(const Adapter::Info & aIn, FILE * aOut)
    {
        if (NULL == (&aIn))	{ return STATUS_INVALID_REFERENCE        ; }
        if (NULL ==   aOut)	{ return STATUS_NOT_ALLOWED_NULL_ARGUMENT; }

        fprintf(aOut, "  Adapter::Info :\n");

        if ((0 <= aIn.mAdapterType) && (OpenNetK::ADAPTER_TYPE_QTY > aIn.mAdapterType))
        {
            fprintf(aOut, "    Adapter Type       = %u - %s\n", aIn.mAdapterType, ADAPTER_TYPE_NAMES[aIn.mAdapterType]);
        }
        else
        {
            fprintf(aOut, "    Adapter Type       = %u - ERROR  Invalid value\n", aIn.mAdapterType);
        }

        fprintf(aOut, "    Comment            = %s\n"      , aIn.mComment);
        fprintf(aOut, "    Common Buffer Size = %u bytes\n", aIn.mCommonBufferSize_byte);

        if ((PACKET_SIZE_MAX_byte >= aIn.mPacketSize_byte) && (PACKET_SIZE_MIN_byte <= aIn.mPacketSize_byte))
        {
            fprintf(aOut, "    Packet Size        = %u bytes\n", aIn.mPacketSize_byte);
        }
        else
        {
            fprintf(aOut, "    Packet Size        = %u bytes - ERROR  Invalid value\n", aIn.mPacketSize_byte);
        }

        fprintf(aOut, "    Rx Descriptors     = %u\n", aIn.mRx_Descriptors);
        fprintf(aOut, "    Tx Descriptors     = %u\n", aIn.mTx_Descriptors);
        fprintf(aOut, "    Ethernet Address   = ");

        EthernetAddress_Display(aIn.mEthernetAddress, aOut);

        fprintf(aOut, "    Version - Driver   =\n");

        VersionInfo_Display(aIn.mVersion_Driver  , aOut);

        fprintf(aOut, "    Version - Hardware =\n");

        VersionInfo_Display(aIn.mVersion_Hardware, aOut);

        fprintf(aOut, "    Version - ONK_Lib  =\n");

        VersionInfo_Display(aIn.mVersion_ONK_Lib , aOut);

        return STATUS_OK;
    }

    Status Adapter::Display(const Adapter::State & aIn, FILE * aOut)
    {
        if (NULL == (&aIn))
        {
            return STATUS_INVALID_REFERENCE;
        }

        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "  Adapter::State :\n");

        if      (ADAPTER_NO_QTY     >  aIn.mAdapterNo) { fprintf(aOut, "    Adapter No  = %u\n"                       , aIn.mAdapterNo); }
        else if (ADAPTER_NO_UNKNOWN == aIn.mAdapterNo) { fprintf(aOut, "    Adapter No  = Unknown\n"                                  ); }
        else                                           { fprintf(aOut, "    Adapter No  = %u - ERROR  Invalid value\n", aIn.mAdapterNo); }

        fprintf(aOut, "    Full Duplxe = %s\n"     , aIn.mFlags.mFullDuplex ? "true" : "false");
        fprintf(aOut, "    Link Up     = %s\n"     , aIn.mFlags.mLinkUp     ? "true" : "false");
        fprintf(aOut, "    Tx Off      = %s\n"     , aIn.mFlags.mTx_Off     ? "true" : "false");
        fprintf(aOut, "    Speed       = %u MB/s\n", aIn.mSpeed_MB_s);

        return STATUS_OK;
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    Adapter::Adapter() : StatisticsProvider(STATISTICS_DESCRIPTIONS, sizeof(STATISTICS_DESCRIPTIONS) / sizeof(STATISTICS_DESCRIPTIONS[0]))
    {
    }

}
