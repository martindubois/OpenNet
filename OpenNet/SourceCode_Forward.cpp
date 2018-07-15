
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/SourceCode_Forward.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Includes ===========================================================
#include <OpenNet/SourceCode.h>

// ===== OpenNet ============================================================
#include "SourceCode_Forward.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

OpenNet::Status SourceCode_Forward_AddDestination(OpenNet::SourceCode * aThis, uint32_t * aDestinations, OpenNet::Adapter * aAdapter)
{
    assert(NULL != aThis        );
    assert(NULL != aDestinations);

    if (NULL == aAdapter)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    unsigned int lAdapterNo;

    OpenNet::Status lResult = aAdapter->GetAdapterNo(&lAdapterNo);
    if (OpenNet::STATUS_OK == lResult)
    {
        uint32_t lDestination = 1 << lAdapterNo;

        if (0 == ((*aDestinations) & lDestination))
        {
            (*aDestinations) |= lDestination;

            lResult = aThis->ResetCode();
            assert(OpenNet::STATUS_OK == lResult);
        }
        else
        {
            lResult = OpenNet::STATUS_DESTINATION_ALREADY_SET;
        }
    }

    return lResult;

}

OpenNet::Status SourceCode_Forward_RemoveDestination(OpenNet::SourceCode * aThis, uint32_t * aDestinations, OpenNet::Adapter * aAdapter)
{
    assert(NULL != aThis        );
    assert(NULL != aDestinations);

    if (NULL == aAdapter)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    unsigned int lAdapterNo;

    OpenNet::Status lResult = aAdapter->GetAdapterNo(&lAdapterNo);
    if (OpenNet::STATUS_OK == lResult)
    {
        uint32_t lDestination = 1 << lAdapterNo;

        if (0 == ((*aDestinations) & lDestination))
        {
            (*aDestinations) |= lDestination;

            lResult = aThis->ResetCode();
            assert(OpenNet::STATUS_OK == lResult);
        }
        else
        {
            lResult = OpenNet::STATUS_DESTINATION_ALREADY_SET;
        }
    }

    return lResult;
}

OpenNet::Status SourceCode_Forward_ResetDestinations(OpenNet::SourceCode * aThis, uint32_t * aDestinations)
{
    assert(NULL != aThis        );
    assert(NULL != aDestinations);

    if (0 == (*aDestinations))
    {
        return OpenNet::STATUS_NO_DESTINATION_SET;
    }

    (*aDestinations) = 0;

    OpenNet::Status lResult = aThis->ResetCode();
    assert(OpenNet::STATUS_OK == lResult);

    return lResult;
}

void SourceCode_Forward_GenerateCode(OpenNet::SourceCode * aThis, uint32_t aDestinations)
{
    assert(NULL != aThis);

    char lDestinations[16];

    sprintf_s(lDestinations, "0x%08x", aDestinations);

    unsigned int lRet = aThis->Edit_Replace("DESTINATIONS", lDestinations);
    assert(1 == lRet);
    (void)(lRet);
}
