
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/System.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/System.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(System_Base)
{
    OpenNet::System * lS0 = OpenNet::System::Create();

    lS0->Delete();
}
KMS_TEST_END
