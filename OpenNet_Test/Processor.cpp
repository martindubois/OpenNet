
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Processor.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Processor.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Processor_Base)
{
    OpenNet::Processor::Info   lI;
    OpenNet::Processor::Info * lIP = NULL;

    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE        , OpenNet::Processor::Display(*lIP, NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, OpenNet::Processor::Display(lI, NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , OpenNet::Processor::Display(lI, stdout));

    memset(&lI, 0, sizeof(lI));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Processor::Display(lI, stdout));
}
KMS_TEST_END