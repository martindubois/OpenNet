
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/VersionInfo.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Status.h>
#include <OpenNetK/Adapter_Types.h>

namespace OpenNet
{

    // Data type
    /////////////////////////////////////////////////////////////////////////

    typedef OpenNetK::VersionInfo VersionInfo;

    // Function
    /////////////////////////////////////////////////////////////////////////

    extern Status VersionInfo_Display(const VersionInfo & aIn, FILE * aOut);

}
