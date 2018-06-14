
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Filter_Forward.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Filter_Forward.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Filter_Forward_Base)
{
    OpenNet::Filter_Forward lFF0;

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lFF0.AddDestination   (NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lFF0.RemoveDestination(NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_NO_DESTINATION_SET       , lFF0.ResetDestinations());
}
KMS_TEST_END
