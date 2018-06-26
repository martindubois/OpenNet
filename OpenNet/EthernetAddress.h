
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/EthernetAddress.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================

#include <OpenNetK/Interface.h>

#include <OpenNet/Status.h>

namespace OpenNet
{

    extern bool EthernetAddress_AreEqual   (const OpenNet_EthernetAddress & aA, const OpenNet_EthernetAddress & aB);

    extern Status EthernetAddress_GetText(const OpenNet_EthernetAddress & aIn, char * aOut, unsigned int aOutSize_byte);
    extern Status EthernetAddress_Display(const OpenNet_EthernetAddress & aIn, FILE * aOut);

    extern Status EthernetAddress_Parse(OpenNet_EthernetAddress * aOut, const char * aIn);

}
