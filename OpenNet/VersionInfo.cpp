
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/VersionInfo.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== includes ===========================================================
#include <OpenNet/VersionInfo.h>

namespace OpenNet
{

    // Functions
    /////////////////////////////////////////////////////////////////////////

    Status VersionInfo_Display(const OpenNet_VersionInfo & aIn, FILE * aOut)
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

        return STATUS_OK;
    }

}
