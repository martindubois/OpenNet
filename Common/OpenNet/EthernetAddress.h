
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNet/EthernetAddress.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================

#include <OpenNetK/Interface.h>

#include <OpenNet/OpenNet.h>
#include <OpenNet/Status.h>

namespace OpenNet
{

    extern OPEN_NET_PUBLIC bool EthernetAddress_IsBroadcast(const OpenNet_EthernetAddress & aIn);
    extern OPEN_NET_PUBLIC bool EthernetAddress_IsMulticast(const OpenNet_EthernetAddress & aIn);
    extern OPEN_NET_PUBLIC bool EthernetAddress_IsZero     (const OpenNet_EthernetAddress & aIn);

}
