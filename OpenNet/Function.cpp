
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Function.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <memory.h>
#include <string.h>

// ===== Includes ===========================================================
#include <OpenNet/Function.h>

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Function::Function()
    {
        memset(&mFunctionName, 0, sizeof(mFunctionName));
    }

    const char * Function::GetFunctionName() const
    {
        return mFunctionName;
    }

    Status Function::SetFunctionName(const char * aFunctionName)
    {
        if (NULL == aFunctionName)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        size_t lLength = strlen(aFunctionName);

        if (0 >= lLength)
        {
            return STATUS_NAME_TOO_SHORT;
        }

        if (sizeof(mFunctionName) <= lLength)
        {
            return STATUS_NAME_TOO_LONG;
        }

        strncpy_s(mFunctionName, aFunctionName, sizeof(mFunctionName) - 1);

        return STATUS_OK;
    }

    // ===== SourceCode =====================================================

    Function::~Function()
    {
    }

    Status Function::Display(FILE * aOut) const
    {
        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "Function:\n");
        fprintf(aOut, "  Function Name = %s\n", mFunctionName);

        return SourceCode::Display(aOut);
    }

}
