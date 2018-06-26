
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/VersionInfo.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================

#include <OpenNetK/Interface.h>

#include <OpenNet/Status.h>

namespace OpenNet
{

    extern Status VersionInfo_Display(const OpenNet_VersionInfo & aIn, FILE * aOut);

}
