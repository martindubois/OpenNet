
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/VersionInfo.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== OpenNet ============================================================
#include "VersionInfo.h"

namespace OpenNet
{

    // Functions
    /////////////////////////////////////////////////////////////////////////

    Status VersionInfo_Display(const VersionInfo & aIn, FILE * aOut)
    {
        if (NULL == (&aIn))
        {
            return STATUS_INVALID_REFERENCE;
        }

        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "      Version = %u.%u.%u.%u\n", aIn.mMajor, aIn.mMinor, aIn.mBuild, aIn.mCompatibility);
        fprintf(aOut, "      Comment = %s\n"         , aIn.mComment);
        fprintf(aOut, "      Type    = %s\n"         , aIn.mType   );
        fprintf(aOut, "      Compiled at %s on %s\n" , aIn.mCompiled_At, aIn.mCompiled_On);
        fprintf(aOut, "      Purchased by %s\n"      , aIn.mClient );

        return STATUS_OK;
    }

}
