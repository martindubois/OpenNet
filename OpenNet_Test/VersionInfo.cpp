
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/VersionInfo.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/VersionInfo.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(VersionInfo_Base)
{
    OpenNet_VersionInfo   lVI;
    OpenNet_VersionInfo * lVIP = NULL;

    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE        , OpenNet::VersionInfo_Display(*lVIP, NULL ));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, OpenNet::VersionInfo_Display( lVI , NULL ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , OpenNet::VersionInfo_Display( lVI, stdout));
}
KMS_TEST_END
