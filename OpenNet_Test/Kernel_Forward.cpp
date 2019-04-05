
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Test/Kernel_Forward.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Kernel_Forward.h>

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Kernel_Forward_Base)
{
    OpenNet::Kernel_Forward lKF0;

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lKF0.AddDestination   (NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lKF0.RemoveDestination(NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_NO_DESTINATION_SET       , lKF0.ResetDestinations());
}
KMS_TEST_END

KMS_TEST_BEGIN(Kernel_Forward_Display)
{
    OpenNet::Kernel_Forward lKF0;

    lKF0.Display(stdout);

    printf("QUESTION  Is the output OK? (y/n)\n");
    char lLine[1024];
    KMS_TEST_ASSERT( NULL != fgets(lLine, sizeof(lLine), stdin) );
    KMS_TEST_COMPARE(0, strncmp("y", lLine, 1));
}
KMS_TEST_END
