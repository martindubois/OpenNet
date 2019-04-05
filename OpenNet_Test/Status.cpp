
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Test/Status.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Status.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Status_Base)
{
    KMS_TEST_ASSERT(NULL != OpenNet::Status_GetDescription(OpenNet::STATUS_QTY));
    KMS_TEST_ASSERT(NULL != OpenNet::Status_GetDescription(OpenNet::STATUS_OK ));

    KMS_TEST_ASSERT(NULL != OpenNet::Status_GetName(OpenNet::STATUS_QTY));
    KMS_TEST_ASSERT(NULL != OpenNet::Status_GetName(OpenNet::STATUS_OK ));
}
KMS_TEST_END
