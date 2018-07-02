
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/EthernetAddress.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Status.h>

// ===== Common =============================================================
#include "../Common/OpenNet/EthernetAddress.h"

namespace OpenNet
{

    extern bool   EthernetAddress_AreEqual(const EthernetAddress & aA  , const  EthernetAddress & aB);
    extern Status EthernetAddress_GetText (const EthernetAddress & aIn , char * aOut, unsigned int aOutSize_byte);
    extern Status EthernetAddress_Display (const EthernetAddress & aIn , FILE * aOut);
    extern Status EthernetAddress_Parse   (      EthernetAddress * aOut, const char * aIn);

}
