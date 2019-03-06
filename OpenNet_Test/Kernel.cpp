
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Kernel.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Kernel.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Kernel_Base)
{
    OpenNet::Kernel lK0;

    KMS_TEST_COMPARE(                                        0, lK0.GetCodeLineCount());
    KMS_TEST_ASSERT (NULL                                    == lK0.GetCodeLines    ());
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lK0.Display         (NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lK0.Display         (stdout));
    KMS_TEST_COMPARE(                             0, strcmp("", lK0.GetName         ()));

    #ifdef _KMS_LINUX_
        KMS_TEST_COMPARE(OpenNet::STATUS_NOT_IMPLEMENTED      , lK0.SetCode         ("OpenNet_Test/Kernel.cpp" ));
    #endif

    #ifdef _KMS_WINDOWS_
        KMS_TEST_COMPARE(OpenNet::STATUS_OK                   , lK0.SetCode         ("OpenNet_Test/Kernel.cpp" ));
        KMS_TEST_COMPARE(                                   63, lK0.GetCodeLineCount());
        KMS_TEST_ASSERT (NULL                                != lK0.GetCodeLines    ());
        KMS_TEST_COMPARE(OpenNet::STATUS_CODE_ALREADY_SET     , lK0.SetCode         (" ", 1));
        KMS_TEST_COMPARE(OpenNet::STATUS_CODE_ALREADY_SET     , lK0.SetCode         ("OpenNet_Test/Kernel.cpp" ));
        KMS_TEST_COMPARE(OpenNet::STATUS_OK                   , lK0.ResetCode       ());
    #endif

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lK0.SetCode         ("A\nB\n\rC\rD\r\nE", 11));
    KMS_TEST_COMPARE(                11, lK0.GetCodeSize     ());
    KMS_TEST_COMPARE(                 5, lK0.GetCodeLineCount());
    KMS_TEST_ASSERT (NULL             != lK0.GetCodeLines    ());
}
KMS_TEST_END

KMS_TEST_BEGIN(Kernel_Display)
{
    OpenNet::Kernel lK0;

    lK0.Display(stdout);

    printf("QUESTION  Is the output OK? (y/n)\n");
    char lLine[1024];
    KMS_TEST_ASSERT(NULL != fgets(lLine, sizeof(lLine), stdin));
    KMS_TEST_COMPARE(0, strncmp("y", lLine, 1));
}
KMS_TEST_END
