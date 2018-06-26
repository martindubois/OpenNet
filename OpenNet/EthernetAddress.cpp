
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/EthernetAddress.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Common =============================================================
#include "../Common/OpenNet/EthernetAddress.h"

// ===== OpenNet ============================================================
#include "EthernetAddress.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static const uint8_t MULTICAST[][2][6] =
{
    { { 0x01, 0x00, 0x0c, 0xcc, 0xcc, 0xcc }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff } },
    { { 0x01, 0x00, 0x0c, 0xcc, 0xcc, 0xcd }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff } },
    { { 0x01, 0x00, 0x5e, 0x00, 0x00, 0x00 }, { 0xff, 0xff, 0xff, 0x80, 0x00, 0x00 } },
    { { 0x01, 0x0c, 0xcd, 0x01, 0x00, 0x00 }, { 0xff, 0xff, 0xff, 0xff, 0xfe, 0x00 } },
    { { 0x01, 0x0c, 0xcd, 0x02, 0x00, 0x00 }, { 0xff, 0xff, 0xff, 0xff, 0xfe, 0x00 } },
    { { 0x01, 0x0c, 0xcd, 0x04, 0x00, 0x00 }, { 0xff, 0xff, 0xff, 0xff, 0xfe, 0x00 } },
    { { 0x01, 0x1b, 0x19, 0x00, 0x00, 0x00 }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff } },
    { { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x00 }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff } },
    { { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x00 }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff } },
    { { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x01 }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff } },
    { { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x02 }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff } },
    { { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x03 }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff } },
    { { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x08 }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff } },
    { { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x0e }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff } },
    { { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x30 }, { 0xff, 0xff, 0xff, 0xff, 0xff, 0xf0 } },
    { { 0x33, 0x33, 0x00, 0x00, 0x00, 0x00 }, { 0xff, 0xff, 0x00, 0x00, 0x00, 0x00 } },
};

namespace OpenNet
{

    // Functions
    /////////////////////////////////////////////////////////////////////////

    bool EthernetAddress_IsBroadcast(const OpenNet_EthernetAddress & aIn)
    {
        if (NULL == (&aIn))
        {
            return false;
        }

        for (unsigned int i = 0; i < sizeof(aIn.mAddress); i++)
        {
            if (0xff != aIn.mAddress[i])
            {
                return false;
            }
        }

        return true;
    }

    bool EthernetAddress_IsMulticast(const OpenNet_EthernetAddress & aIn)
    {
        if (NULL == (&aIn))
        {
            return false;
        }

        for (unsigned int i = 0; i < (sizeof(MULTICAST) / sizeof(MULTICAST[0])); i++)
        {
            unsigned int j;

            for (j = 0; j < 6; j++)
            {
                if (MULTICAST[i][0][j] != (aIn.mAddress[j] & MULTICAST[i][1][j]))
                {
                    break;
                }
            }

            if (6 == j)
            {
                return true;
            }
        }

        return false;
    }

    bool EthernetAddress_IsZero(const OpenNet_EthernetAddress & aIn)
    {
        if (NULL == (&aIn))
        {
            return false;
        }

        for (unsigned int i = 0; i < sizeof(aIn.mAddress); i++)
        {
            if (0 != aIn.mAddress[i])
            {
                return false;
            }
        }

        return true;
    }

    Status EthernetAddress_Display(const OpenNet_EthernetAddress & aIn, FILE * aOut)
    {
        if (NULL == (&aIn))
        {
            return STATUS_INVALID_REFERENCE;
        }

        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "%02x:%02x:%02x:%02x:%02x:%02x", aIn.mAddress[0], aIn.mAddress[1], aIn.mAddress[2], aIn.mAddress[3], aIn.mAddress[4], aIn.mAddress[5]);

        if      (EthernetAddress_IsBroadcast(aIn))
        {
            fprintf(aOut, " - Broadcast" );
        }
        else if (EthernetAddress_IsMulticast(aIn))
        {
            fprintf(aOut, " - Multicast" );
        }
        else if (EthernetAddress_IsZero     (aIn))
        {
            fprintf(aOut, " - Zero"      );
        }

        fprintf(aOut, "\n");

        return STATUS_OK;
    }

    Status EthernetAddress_GetText(const OpenNet_EthernetAddress & aIn, char * aOut, unsigned int aOutSize_byte)
    {
        if (NULL == (&aIn))
        {
            return STATUS_INVALID_REFERENCE;
        }

        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        if (20 > aOutSize_byte)
        {
            return STATUS_BUFFER_TOO_SMALL;
        }

        sprintf_s(aOut, aOutSize_byte, "%02x:%02x:%02x:%02x:%02x:%02x", aIn.mAddress[0], aIn.mAddress[1], aIn.mAddress[2], aIn.mAddress[3], aIn.mAddress[4], aIn.mAddress[5]);

        return STATUS_OK;
    }
}
