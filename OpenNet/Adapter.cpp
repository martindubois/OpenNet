
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
    { "ADAPTER BUFFER_RELEASED              ", "", 0 },
    { "ADAPTER LOOP_BACK_PACKET             ", "", 0 },
    { "ADAPTER PACKET_SEND                  ", "", 1 },
    { "ADAPTER RUN_ENTRY                    ", "", 0 },
    { "ADAPTER RUN_EXCEPTION                ", "", 1 }, //  5
    { "ADAPTER RUN_EXIT                     ", "", 0 },
    { "ADAPTER RUN_ITERATION_QUEUE          ", "", 0 },
    { "ADAPTER RUN_ITERATION_WAIT           ", "", 0 },
    { "ADAPTER RUN_LOOP_EXCEPTION           ", "", 0 },
    { "ADAPTER RUN_LOOP_UNEXPECTED_EXCEPTION", "", 1 }, // 10
    { "ADAPTER RUN_QUEUE                    ", "", 0 },
    { "ADAPTER RUN_UNEXPECTED_EXCEPTION     ", "", 1 },
    { "ADAPTER START                        ", "", 0 },
    { "ADAPTER STOP_REQUEST                 ", "", 0 },
    { "ADAPTER STOP_WAIT                    ", "", 0 }, // 15

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

    { "ADAPTER BUFFERS_PROCESS     ", ""      , 0 },
    { "ADAPTER BUFFER_INIT_HEADER  ", ""      , 0 },
    { "ADAPTER BUFFER_QUEUE        ", ""      , 0 },
    { "ADAPTER BUFFER_RECEIVE      ", ""      , 1 }, // 35
    { "ADAPTER BUFFER_SEND         ", ""      , 1 },
    { "ADAPTER BUFFER_SEND_PACKETS ", ""      , 1 },
    { "ADAPTER IOCTL               ", ""      , 0 },
    { "ADAPTER IOCTL_CONFIG_GET    ", ""      , 0 },
    { "ADAPTER IOCTL_CONFIG_SET    ", ""      , 0 }, // 40
    { "ADAPTER IOCTL_CONNECT       ", ""      , 0 },
    { "ADAPTER IOCTL_INFO_GET      ", ""      , 0 },
    { "ADAPTER IOCTL_PACKET_SEND   ", ""      , 1 },
    { "ADAPTER IOCTL_START         ", ""      , 0 },
    { "ADAPTER IOCTL_STATE_GET     ", ""      , 0 }, // 45
    { "ADAPTER IOCTL_STATISTICS_GET", ""      , 0 },
    { "ADAPTER IOCTL_STOP          ", ""      , 0 },
    { "ADAPTER TX                  ", "packet", 1 },

    VALUE_VECTOR_DESCRIPTION_RESERVED,
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

    { "ADAPTER IOCTL_LAST                 (NR)", "", 0 }, // 60
    { "ADAPTER IOCTL_LAST_RESULT          (NR)", "", 0 },
    { "ADAPTER IOCTL_STATISTICS_GET_RESET (NR)", "", 0 },
    { "ADAPTER IOCTL_STATISTICS_RESET     (NR)", "", 0 },

    { "HARDWRE D0_ENTRY          ", ""      , 0 },
    { "HARDWRE D0_EXIT           ", ""      , 0 }, // 65
    { "HARDWRE INTERRUPT_DISABLE ", ""      , 0 },
    { "HARDWRE INTERRUPT_ENABLE  ", ""      , 0 },
    { "HARDWRE INTERRUPT_PROCESS ", ""      , 1 },
    { "HARDWRE INTERRUPT_PROCESS2", ""      , 1 },
    { "HARDWRE PACKET_RECEIVE    ", ""      , 1 }, // 70
    { "HARDWRE PACKET_SEND       ", ""      , 1 },
    { "HARDWRE RX                ", "packet", 1 },
    { "HARDWRE SET_CONFIG        ", ""      , 0 },
    { "HARDWRE STATISTICS_GET    ", ""      , 0 },
    { "HARDWRE TX                ", "packet", 1 }, // 75

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
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED, // 95
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "HARDWARE RX_BMC_MANAGEMENT_DROPPED      ", "packet", 1 }, // 100
    { "HARDWARE RX_CIRCUIT_BREAKER_DROPPED     ", "packet", 1 },
    { "HARDWARE RX_LENGTH_ERRORS               ", "packet", 1 },
    { "HARDWARE RX_MANAGEMENT_DROPPED          ", "packet", 1 },
    { "HARDWARE RX_MISSED                      ", "packet", 1 },
    { "HARDWARE RX_NO_BUFFER                   ", "packet", 0 }, // 105
    { "HARDWARE RX_OVERSIZE                    ", "packet", 1 },
    { "HARDWARE RX_UNDERSIZE                   ", "packet", 1 },
    { "HARDWARE TX_DEFER_EVENTS                ", ""      , 0 },
    { "HARDWARE TX_DISCARDED                   ", "packet", 1 },
    { "HARDWARE TX_NO_CRS                      ", "packet", 1 }, // 110
    { "HARDWARE TX_HOST_CIRCUIT_BREAKER_DROPPED", "packet", 1 },

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

    { "HARDWARE INTERRUPT_PROCESS_LAST_MESSAGE_ID (NR)", "", 0 }, // 125
    { "HARDWARE STATISTICS_GET_RESET              (NR)", "", 0 },
    { "HARDWARE STATISTICS_RESET                  (NR)", "", 0 },

};

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static const char * GetIoCtlName(unsigned int aCode);

static void Display(const OpenNet::Adapter::Stats_Dll    & aIn, FILE * aOut);

static void DisplayStats(const char * aText, unsigned int aValue, FILE * aOut);

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Status Adapter::Display(const Adapter::Config & aIn, FILE * aOut)
    {
        if (NULL == (&aIn)) { return STATUS_INVALID_REFERENCE        ; }
        if (NULL ==   aOut) { return STATUS_NOT_ALLOWED_NULL_ARGUMENT; }

        fprintf(aOut, "  Adapter Configuration :\n");

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

        fprintf(aOut, "  Adapter Information :\n");

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

        fprintf(aOut, "    Version - Driver :\n");

        VersionInfo_Display(aIn.mVersion_Driver  , aOut);

        fprintf(aOut, "    Version - Hardware :\n");

        VersionInfo_Display(aIn.mVersion_Hardware, aOut);

        fprintf(aOut, "    Version - ONK_Lib :\n");

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

        fprintf(aOut, "  Adapter State :\n");

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

// Static functions
/////////////////////////////////////////////////////////////////////////////

const char * GetIoCtlName(unsigned int aCode)
{
    switch (aCode)
    {
    case IOCTL_CONFIG_GET      : return "CONFIG_GET"      ;
    case IOCTL_CONFIG_SET      : return "CONFIG_SET"      ;
    case IOCTL_CONNECT         : return "CONNECT"         ;
    case IOCTL_INFO_GET        : return "INFO_GET"        ;
    case IOCTL_PACKET_SEND     : return "PACKET_SEND"     ;
    case IOCTL_START           : return "START"           ;
    case IOCTL_STATE_GET       : return "STATE_GET"       ;
    case IOCTL_STATISTICS_GET  : return "STATISTICS_GET"  ;
    case IOCTL_STATISTICS_RESET: return "STATISTICS_RESET";
    case IOCTL_STOP            : return "STOP"            ;
    }

    return "Invalid IoCtl code";
}

void Display(const OpenNet::Adapter::Stats_Dll & aIn, FILE * aOut)
{
    assert(NULL != (&aIn));
    assert(NULL !=   aOut);

    fprintf(aOut, "  Dll Statistics :\n");
    fprintf(aOut, "    Buffer - Allocated                = %u\n", aIn.mBuffer_Allocated   );
    DisplayStats( "           - Released                 = %u\n", aIn.mBuffer_Released             , aOut);
    DisplayStats( "    Loop Back Packet                  = %u\n", aIn.mLoopBackPacket              , aOut);
    DisplayStats( "    Packet - Send                     = %u\n", aIn.mPacket_Send                 , aOut);
    fprintf(aOut, "    Run - Entry                       = %u\n", aIn.mRun_Entry          );
    DisplayStats( "        - Exception                   = %u\n", aIn.mRun_Exception               , aOut);
    DisplayStats( "        - Exit                        = %u\n", aIn.mRun_Exit                    , aOut);
    fprintf(aOut, "        - Iteration - Queue           = %u\n", aIn.mRun_Iteration_Queue);
    DisplayStats( "                    - Wait            = %u\n", aIn.mRun_Iteration_Wait          , aOut);
    fprintf(aOut, "        - Loop - Exception            = %u\n", aIn.mRun_Loop_Exception );
    DisplayStats( "               - Unexpected Exception = %u\n", aIn.mRun_Loop_UnexpectedException, aOut);
    DisplayStats( "        - Queue                       = %u\n", aIn.mRun_Queue                   , aOut);
    DisplayStats( "        - Unexpected Exception        = %u\n", aIn.mRun_UnexpectedException     , aOut);
    DisplayStats( "    Start                             = %u\n", aIn.mStart                       , aOut);
    fprintf(aOut, "    Stop - Request                    = %u\n", aIn.mStop_Request       );
    DisplayStats( "         - Wait                       = %u\n", aIn.mStop_Wait                   , aOut);
}

void DisplayStats(const char * aFormat, unsigned int aValue, FILE * aOut)
{
    assert(NULL != aFormat);
    assert(NULL != aOut   );

    if (0 < aValue)
    {
        fprintf(aOut, aFormat, aValue);
    }
}
