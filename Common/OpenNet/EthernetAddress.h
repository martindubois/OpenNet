
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/OpenNet/EthernetAddress.h
//
// This file declares the helper function for the EthernetAddress type.

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/OpenNet.h>
#include <OpenNet/Status.h>
#include <OpenNetK/Adapter_Types.h>

namespace OpenNet
{

    // Data type
    /////////////////////////////////////////////////////////////////////////

    typedef OpenNetK::EthernetAddress EthernetAddress;

    // Function
    /////////////////////////////////////////////////////////////////////////

    extern OPEN_NET_PUBLIC bool EthernetAddress_IsBroadcast(const EthernetAddress & aIn);
    extern OPEN_NET_PUBLIC bool EthernetAddress_IsMulticast(const EthernetAddress & aIn);
    extern OPEN_NET_PUBLIC bool EthernetAddress_IsZero     (const EthernetAddress & aIn);

}
