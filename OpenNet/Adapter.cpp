
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

// ===== Includes/OpenNet ===================================================
#include <OpenNet/EthernetAddress.h>
#include <OpenNet/VersionInfo.h>

#include <OpenNet/Adapter.h>

// Constants
/////////////////////////////////////////////////////////////////////////////

static const char * ADAPTER_MODE_NAMES[OPEN_NET_MODE_QTY] =
{
    "UNKNOWN"    ,
    "NORMAL"     ,
    "PROMISCUOUS",
};

static const char * ADAPTER_TYPE_NAMES[OPEN_NET_ADAPTER_TYPE_QTY] =
{
    "UNKNOWN" ,
    "ETHERNET",
    "NDIS"    ,
};

static const char * LINK_STATE_NAMES[OPEN_NET_LINK_STATE_QTY] =
{
    "UNKNOWN",
    "DOWN"   ,
    "UP"     ,
};

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static const char * GetIoCtlName(unsigned int aCode);

static void Display(const OpenNet::Adapter::Stats_Dll    & aIn, FILE * aOut);
static void Display(const OpenNet_Stats                  & aIn, FILE * aOut);
static void Display(const OpenNet_Stats_Adapter          & aIn, FILE * aOut);
static void Display(const OpenNet_Stats_Adapter_NoReset  & aIn, FILE * aOut);
static void Display(const OpenNet_Stats_Hardware         & aIn, FILE * aOut);
static void Display(const OpenNet_Stats_Hardware_NoReset & aIn, FILE * aOut);

static void DisplayStats(const char * aText, unsigned int aValue, FILE * aOut);

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Status Adapter::Display(const OpenNet::Adapter::Config & aIn, FILE * aOut)
    {
        if (NULL == (&aIn)) { return STATUS_INVALID_REFERENCE        ; }
        if (NULL ==   aOut) { return STATUS_NOT_ALLOWED_NULL_ARGUMENT; }

        fprintf(aOut, "  Adapter Configuration :\n");

        if (OPEN_NET_MODE_QTY > aIn.mMode)
        {
            fprintf(aOut, "    Mode        = %u - %s\n", aIn.mMode, ADAPTER_MODE_NAMES[aIn.mMode]);
        }
        else
        {
            fprintf(aOut, "    Mode        = %u - ERROR  Invalid value\n", aIn.mMode);
        }

        if ((OPEN_NET_PACKET_SIZE_MAX_byte >= aIn.mPacketSize_byte) && (OPEN_NET_PACKET_SIZE_MIN_byte <= aIn.mPacketSize_byte))
        {
            fprintf(aOut, "    Packet Size = %u bytes\n", aIn.mPacketSize_byte);
        }
        else
        {
            fprintf(aOut, "    Packet Size = %u bytes - ERROR  Invalid value\n", aIn.mPacketSize_byte);
        }

        fprintf(aOut, "    Ethernet Address\n");

        for (unsigned int i = 0; i < sizeof(aIn.mEthernetAddress) / sizeof(aIn.mEthernetAddress[0]); i++)
        {
            if (!EthernetAddress_IsZero(aIn.mEthernetAddress[i]))
            {
                fprintf(aOut, "        ");
                EthernetAddress_Display(aIn.mEthernetAddress[i], aOut);
            }
        }

        return STATUS_OK;
    }

    Status Adapter::Display(const OpenNet::Adapter::Info & aIn, FILE * aOut)
    {
        if (NULL == (&aIn))	{ return STATUS_INVALID_REFERENCE        ; }
        if (NULL ==   aOut)	{ return STATUS_NOT_ALLOWED_NULL_ARGUMENT; }

        fprintf(aOut, "  Adapter Information :\n");

        if (OPEN_NET_ADAPTER_TYPE_QTY > aIn.mAdapterType)
        {
            fprintf(aOut, "    Adapter Type       = %u - %s\n", aIn.mAdapterType, ADAPTER_TYPE_NAMES[aIn.mAdapterType]);
        }
        else
        {
            fprintf(aOut, "    Adapter Type       = %u - ERROR  Invalid value\n", aIn.mAdapterType);
        }

        fprintf(aOut, "    Comment            = %s\n"      , aIn.mComment);
        fprintf(aOut, "    Common Buffer Size = %u bytes\n", aIn.mCommonBufferSize_byte);

        if ((OPEN_NET_PACKET_SIZE_MAX_byte >= aIn.mPacketSize_byte) && (OPEN_NET_PACKET_SIZE_MIN_byte <= aIn.mPacketSize_byte))
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

    Status Adapter::Display(const OpenNet::Adapter::State & aIn, FILE * aOut)
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

        if      (OPEN_NET_ADAPTER_NO_QTY     >  aIn.mAdapterNo) { fprintf(aOut, "    Adapter No  = %u\n"                       , aIn.mAdapterNo); }
        else if (OPEN_NET_ADAPTER_NO_UNKNOWN == aIn.mAdapterNo) { fprintf(aOut, "    Adapter No  = %u - Unknown\n"             , aIn.mAdapterNo); }
        else                                                    { fprintf(aOut, "    Adapter No  = %u - ERROR  Invalid value\n", aIn.mAdapterNo); }

        fprintf(aOut, "    Full Duplxe = %s\n"     , aIn.mFlags.mFullDuplex ? "true" : "false");
        fprintf(aOut, "    Link Up     = %s\n"     , aIn.mFlags.mLinkUp     ? "true" : "false");
        fprintf(aOut, "    Tx Off      = %s\n"     , aIn.mFlags.mTx_Off     ? "true" : "false");
        fprintf(aOut, "    Speed       = %u MB/s\n", aIn.mSpeed_MB_s);

        return STATUS_OK;
    }

    Status Adapter::Display(const OpenNet::Adapter::Stats & aIn, FILE * aOut)
    {
        if (NULL == (&aIn))
        {
            return STATUS_INVALID_REFERENCE;
        }

        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "Adapter Statistics:\n");

        ::Display(aIn.mDll   , aOut);
        ::Display(aIn.mDriver, aOut);

        return STATUS_OK;
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    Adapter::Adapter()
    {
    }

}

// Static functions
/////////////////////////////////////////////////////////////////////////////

const char * GetIoCtlName(unsigned int aCode)
{
    switch (aCode)
    {
    case OPEN_NET_IOCTL_CONFIG_GET : return "CONFIG_GET" ;
    case OPEN_NET_IOCTL_CONFIG_SET : return "CONFIG_SET" ;
    case OPEN_NET_IOCTL_CONNECT    : return "CONNECT"    ;
    case OPEN_NET_IOCTL_INFO_GET   : return "INFO_GET"   ;
    case OPEN_NET_IOCTL_PACKET_SEND: return "PACKET_SEND";
    case OPEN_NET_IOCTL_START      : return "START"      ;
    case OPEN_NET_IOCTL_STATE_GET  : return "STATE_GET"  ;
    case OPEN_NET_IOCTL_STATS_GET  : return "STATS_GET"  ;
    case OPEN_NET_IOCTL_STATS_RESET: return "STATS_RESET";
    case OPEN_NET_IOCTL_STOP       : return "STOP"       ;
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
    DisplayStats( "    Packet - Send                     = %u\n", aIn.mPacket_Send                 , aOut);
    fprintf(aOut, "    Run - Entry                       = %u\n", aIn.mRun_Entry          );
    DisplayStats( "        - Exception                   = %u\n", aIn.mRun_Exception               , aOut);
    DisplayStats( "        - Exit                        = %u\n", aIn.mRun_Exit                    , aOut);
    fprintf(aOut, "        - Iteration - Queue           = %u\n", aIn.mRun_Iteration_Queue);
    DisplayStats( "                    - Wait            = %u\n", aIn.mRun_Iteration_Wait          , aOut);
    fprintf(aOut, "        - Loop - Exception            = %u\n", aIn.mRun_Loop_Exception );
    DisplayStats( "               - Unexpected Exception = %u\n", aIn.mRun_Loop_UnexpectedException, aOut);
    DisplayStats( "               - Wait                 = %u\n", aIn.mRun_Loop_Wait               , aOut);
    DisplayStats( "        - Queue                       = %u\n", aIn.mRun_Queue                   , aOut);
    DisplayStats( "        - Unexpected Exception        = %u\n", aIn.mRun_UnexpectedException     , aOut);
    DisplayStats( "    Start                             = %u\n", aIn.mStart                       , aOut);
    fprintf(aOut, "    Stop - Request                    = %u\n", aIn.mStop_Request       );
    DisplayStats( "         - Wait                       = %u\n", aIn.mStop_Wait                   , aOut);
}

void Display(const OpenNet_Stats & aIn, FILE * aOut)
{
    assert(NULL != (&aIn));
    assert(NULL !=   aOut);

    fprintf(aOut, "  Driver Statistics :\n");

    ::Display(aIn.mAdapter         , aOut);
    ::Display(aIn.mAdapter_NoReset , aOut);
    ::Display(aIn.mHardware        , aOut);
    ::Display(aIn.mHardware_NoReset, aOut);
}

void Display(const OpenNet_Stats_Adapter & aIn, FILE * aOut)
{
    assert(NULL != (&aIn));
    assert(NULL !=   aOut);

    fprintf(aOut, "    Adapter Statistics :\n");
    fprintf(aOut, "      Buffer - InitHeader  = %u\n", aIn.mBuffer_InitHeader);
    DisplayStats( "             - Queue       = %u\n", aIn.mBuffer_Queue      , aOut);
    DisplayStats( "             - Receive     = %u\n", aIn.mBuffer_Receive    , aOut);
    DisplayStats( "             - Send        = %u\n", aIn.mBuffer_Send       , aOut);
    DisplayStats( "             - SendPackets = %u\n", aIn.mBuffer_SendPackets, aOut);
    DisplayStats( "      Buffers - Process    = %u\n", aIn.mBuffers_Process   , aOut);
    fprintf(aOut, "      IoCtl                = %u\n", aIn.mIoCtl            );
    fprintf(aOut, "        - Config - Get     = %u\n", aIn.mIoCtl_Config_Get );
    DisplayStats( "                 - Set     = %u\n", aIn.mIoCtl_Config_Set  , aOut);
    DisplayStats( "        - Connect          = %u\n", aIn.mIoCtl_Connect     , aOut);
    DisplayStats( "        - Info - Get       = %u\n", aIn.mIoCtl_Info_Get    , aOut);
    DisplayStats( "        - Packet - Send    = %u\n", aIn.mIoCtl_Packet_Send , aOut);
    DisplayStats( "        - Start            = %u\n", aIn.mIoCtl_Start       , aOut);
    DisplayStats( "        - State - Get      = %u\n", aIn.mIoCtl_State_Get   , aOut);
    DisplayStats( "        - Stats - Get      = %u\n", aIn.mIoCtl_Stats_Get   , aOut);
    DisplayStats( "        - Stop             = %u\n", aIn.mIoCtl_Stop        , aOut);
    DisplayStats( "      Tx - Packet          = %u\n", aIn.mTx_Packet         , aOut);
}

void Display(const OpenNet_Stats_Adapter_NoReset & aIn, FILE * aOut)
{
    assert(NULL != (&aIn));
    assert(NULL !=   aOut);

    fprintf(aOut, "    Adapter statistics (No Reset) :\n");
    fprintf(aOut, "      IoCtl - Last          = 0x%08x - %s\n", aIn.mIoCtl_Last, GetIoCtlName(aIn.mIoCtl_Last));
    DisplayStats( "                - Result    = 0x%08x\n"     , aIn.mIoCtl_Last_Result, aOut);
    DisplayStats( "            - Stats - Reset = %u\n"         , aIn.mIoCtl_Stats_Reset, aOut);
}

void Display(const OpenNet_Stats_Hardware & aIn, FILE * aOut)
{
    assert(NULL != (&aIn));
    assert(NULL !=   aOut);

    fprintf(aOut, "    Hardware Statistics :\n");
    fprintf(aOut, "      D0 - Entry           = %u\n", aIn.mD0_Entry         );
    DisplayStats( "         - Exit            = %u\n", aIn.mD0_Exit           , aOut);
    fprintf(aOut, "      Interrupt - Disable  = %u\n", aIn.mInterrupt_Disable);
    DisplayStats( "                - Enable   = %u\n", aIn.mInterrupt_Enable  , aOut);
    DisplayStats( "                - Process  = %u\n", aIn.mInterrupt_Process , aOut);
    DisplayStats( "                - Process2 = %u\n", aIn.mInterrupt_Process2, aOut);
    fprintf(aOut, "      Packet - Receive     = %u\n", aIn.mPacket_Receive   );
    DisplayStats( "             - Send        = %u\n", aIn.mPacket_Send       , aOut);
    DisplayStats( "      Rx - Packet          = %u\n", aIn.mRx_Packet         , aOut);
    DisplayStats( "      SetConfig            = %u\n", aIn.mSetConfig         , aOut);
    DisplayStats( "      Stats - Get          = %u\n", aIn.mStats_Get         , aOut);
    DisplayStats( "      Tx - Packet          = %u\n", aIn.mTx_Packet         , aOut);
}

void Display(const OpenNet_Stats_Hardware_NoReset & aIn, FILE * aOut)
{
    assert(NULL != (&aIn));
    assert(NULL !=   aOut);

    fprintf(aOut, "    Hardware Statistics (No Reset) :\n");
    fprintf(aOut, "      Stats - Reset = %u\n", aIn.mStats_Reset);
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
