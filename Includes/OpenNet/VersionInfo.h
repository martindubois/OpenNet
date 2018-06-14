
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Include/OpenNet/VersionInfo.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================

#include <OpenNetK/Interface.h>

#include <OpenNet/OpenNet.h>
#include <OpenNet/Status.h>

namespace OpenNet
{

    extern OPEN_NET_PUBLIC Status VersionInfo_Display(const OpenNet_VersionInfo & aIn, FILE * aOut);

}
