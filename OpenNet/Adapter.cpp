
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Adapter.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// ===== Windows ============================================================
#include <Windows.h>

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
    { "ADAPTER BUFFER_ALLOCATED             ", "", 0 }, //  0

    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "ADAPTER - LOOP_BACK_PACKET             ", "", 0 },
    { "ADAPTER - PACKET_SEND                  ", "", 1 },

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

    { "ADAPTER - BUFFERS - PROCESS       ", ""      , 0 },
    { "ADAPTER - BUFFER - INIT_HEADER    ", ""      , 0 },
    { "ADAPTER - BUFFER - QUEUE          ", ""      , 0 },
    { "ADAPTER - BUFFER - RECEIVE        ", ""      , 1 }, // 35
    { "ADAPTER - BUFFER - SEND           ", ""      , 1 },
    { "ADAPTER - BUFFER - SEND_PACKETS   ", ""      , 1 },
    { "ADAPTER - IOCTL                   ", ""      , 0 },
    { "ADAPTER - IOCTL - CONFIG - GET    ", ""      , 0 },
    { "ADAPTER - IOCTL - CONFIG - SET    ", ""      , 0 }, // 40
    { "ADAPTER - IOCTL - CONNECT         ", ""      , 0 },
    { "ADAPTER - IOCTL - INFO_GET        ", ""      , 0 },
    { "ADAPTER - IOCTL - PACKET_SEND     ", ""      , 1 },
    { "ADAPTER - IOCTL - START           ", ""      , 0 },
    { "ADAPTER - IOCTL - STATE_GET       ", ""      , 0 }, // 45
    { "ADAPTER - IOCTL - STATISTICS - GET", ""      , 0 },
    { "ADAPTER - IOCTL - STOP            ", ""      , 0 },
    { "ADAPTER - RUNNING_TIME            ", "ms"    , 1 },
    { "ADAPTER - TX                      ", "packet", 1 },

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

    { "ADAPTER - IOCTL - LAST                     (NR)", "", 0 }, // 60
    { "ADAPTER - IOCTL - LAST - RESULT            (NR)", "", 0 },
    { "ADAPTER - IOCTL - STATISTICS - GET - RESET (NR)", "", 0 },
    { "ADAPTER - IOCTL - STATISTICS - RESET       (NR)", "", 0 },

    { "HARDWARE - D0 - ENTRY          ", ""      , 0 },
    { "HARDWARE - D0 - EXIT           ", ""      , 0 }, // 65
    { "HARDWARE - INTERRUPT - DISABLE ", ""      , 0 },
    { "HARDWARE - INTERRUPT - ENABLE  ", ""      , 0 },
    { "HARDWARE - INTERRUPT - PROCESS ", ""      , 1 },
    { "HARDWARE - INTERRUPT - PROCESS2", ""      , 1 },
    { "HARDWARE - PACKET - RECEIVE    ", ""      , 1 }, // 70
    { "HARDWARE - PACKET - SEND       ", ""      , 1 },
    { "HARDWARE - RX                  ", "packet", 1 },
    { "HARDWARE - SET_CONFIG          ", ""      , 0 },
    { "HARDWARE - STATISTICS - GET    ", ""      , 0 },
    { "HARDWARE - TX                  ", "packet", 1 }, // 75

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

    { "HARDWARE - RX - BMC_MANAGEMENT_DROPPED       (HW)", "packet", 1 },
    { "HARDWARE - RX - CIRCUIT_BREAKER_DROPPED      (HW)", "packet", 1 },
    { "HARDWARE - RX - HOST                         (HW)", "byte"  , 1 },
    { "HARDWARE - RX - HOST                         (HW)", "packet", 1 },
    { "HARDWARE - RX - LENGTH_ERRORS                (HW)", "packet", 1 }, //  95
    { "HARDWARE - RX - MANAGEMENT_DROPPED           (HW)", "packet", 1 },
    { "HARDWARE - RX - MISSED                       (HW)", "packet", 1 },
    { "HARDWARE - RX - NO_BUFFER                    (HW)", "packet", 0 },
    { "HARDWARE - RX - OVERSIZE                     (HW)", "packet", 1 },
    { "HARDWARE - RX - QUEUE_DROPPED                (HW)", "packet", 1 }, // 100
    { "HARDWARE - RX - UNDERSIZE                    (HW)", "packet", 1 },
    { "HARDWARE - RX - XOFF                         (HW)", "packet", 0 },
    { "HARDWARE - RX - XON                          (HW)", "packet", 0 },
    { "HARDWARE - TX - DEFER_EVENTS                 (HW)", ""      , 0 },
    { "HARDWARE - TX - DISCARDED                    (HW)", "packet", 1 }, // 105
    { "HARDWARE - TX - HOST                         (HW)", "byte"  , 1 },
    { "HARDWARE - TX - HOST                         (HW)", "packet", 1 },
    { "HARDWARE - TX - HOST_CIRCUIT_BREAKER_DROPPED (HW)", "packet", 1 },
    { "HARDWARE - TX - NO_CRS                       (HW)", "packet", 1 },
    { "HARDWARE - TX - XOFF                         (HW)", "packet", 0 }, // 110
    { "HARDWARE - TX - XON                          (HW)", "packet", 0 },

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
    { "HARDWARE - STATISTICS - GET - RESET          (NR)", "", 0 },
    { "HARDWARE - STATISTICS - RESET                (NR)", "", 0 },

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
