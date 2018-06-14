
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Include/OpenNet/EthernetAddress.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================

#include <OpenNetK/Interface.h>

#include <OpenNet/OpenNet.h>
#include <OpenNet/Status.h>

namespace OpenNet
{

    extern OPEN_NET_PUBLIC bool EthernetAddress_AreEqual   (const OpenNet_EthernetAddress & aA, const OpenNet_EthernetAddress & aB);
    extern OPEN_NET_PUBLIC bool EthernetAddress_IsBroadcast(const OpenNet_EthernetAddress & aIn);
    extern OPEN_NET_PUBLIC bool EthernetAddress_IsMulticast(const OpenNet_EthernetAddress & aIn);
    extern OPEN_NET_PUBLIC bool EthernetAddress_IsZero     (const OpenNet_EthernetAddress & aIn);

    extern OPEN_NET_PUBLIC Status EthernetAddress_GetText(const OpenNet_EthernetAddress & aIn, char * aOut, unsigned int aOutSize_byte);
    extern OPEN_NET_PUBLIC Status EthernetAddress_Display(const OpenNet_EthernetAddress & aIn, FILE * aOut);

    extern OPEN_NET_PUBLIC Status EthernetAddress_Parse(OpenNet_EthernetAddress * aOut, const char * aIn);

}
