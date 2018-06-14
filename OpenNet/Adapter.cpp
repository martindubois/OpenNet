
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Adapter.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
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

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Status Adapter::Display(const OpenNet::Adapter::Config & aIn, FILE * aOut)
    {
        if (NULL == (&aIn)) { return STATUS_INVALID_REFERENCE        ; }
        if (NULL ==   aOut) { return STATUS_NOT_ALLOWED_NULL_ARGUMENT; }

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

        fprintf(aOut, "    Version Driver\n");

        VersionInfo_Display(aIn.mVersion_Driver  , aOut);

        fprintf(aOut, "    Version Hardware\n");

        VersionInfo_Display(aIn.mVersion_Hardware, aOut);

        fprintf(aOut, "    Version Driver\n");

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

        fprintf(aOut, "    IoCtl         = %u\n"    , aIn.mIoCtl            );
        fprintf(aOut, "      - Last      = 0x%08x\n", aIn.mIoCtl_Last       );
        fprintf(aOut, "         - Result = 0x%08x\n", aIn.mIoCtl_Last_Result);
        fprintf(aOut, "    Rx            = %u buffers\n"   , aIn.mRx_buffer     );
        fprintf(aOut, "                  = %llu bytes\n"   , aIn.mRx_byte       );
        fprintf(aOut, "                  = %u errors\n"    , aIn.mRx_error      );
        fprintf(aOut, "                  = %u interrupts\n", aIn.mRx_interrupt  );
        fprintf(aOut, "                  = %u packets\n"   , aIn.mRx_packet     );
        fprintf(aOut, "    Stats - Get   = %u\n"           , aIn.mStats_Get     );
        fprintf(aOut, "          - Reset = %u\n"           , aIn.mStats_Reset   );
        fprintf(aOut, "    Tx            = %u buffers\n"   , aIn.mTx_buffer     );
        fprintf(aOut, "                  = %llu bytes\n"   , aIn.mTx_byte       );
        fprintf(aOut, "                  = %u errors\n"    , aIn.mTx_error      );
        fprintf(aOut, "                  = %u interrupts\n", aIn.mTx_interrupt  );
        fprintf(aOut, "                  = %u packets\n"   , aIn.mTx_packet     );
        fprintf(aOut, "      - Send      = %u bytes\n"     , aIn.mTx_Send_byte  );
        fprintf(aOut, "                  = %u errors\n"    , aIn.mTx_Send_error );
        fprintf(aOut, "                  = %u packets\n"   , aIn.mTx_Send_packet);

        return STATUS_OK;
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    Adapter::Adapter()
    {
    }

}
