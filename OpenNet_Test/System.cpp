
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet_Test/System.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/ThreadBase.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/System.h>

// ===== OpenNet_Test =======================================================
#include "Base.h"
#include "SetupA.h"
#include "SetupC.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static const uint8_t PACKET[ 64 ] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x0a };

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN( System_1Packet )
{
    SetupC lSetup( 2 );

    KMS_TEST_COMPARE_RETURN( 0, lSetup.Init() );
    KMS_TEST_COMPARE_RETURN( 0, lSetup.Start( OpenNet::System::START_FLAG_LOOPBACK ) );

    unsigned int i;

    for ( i = 0; i < 28; i ++ )
    {
        printf( "Sending packet %u\n", i );

        KmsLib::ThreadBase::Sleep_ms( 500 );

        lSetup.mAdapters[ 1 ]->Packet_Send( PACKET, sizeof( PACKET ) );

        KmsLib::ThreadBase::Sleep_ms( 500 );
    }

    for ( ; i < 36; i ++ )
    {
        printf( "Sending packet %u\n", i );

        KmsLib::ThreadBase::Sleep_ms( 1000 );

        lSetup.mAdapters[ 1 ]->Packet_Send( PACKET, sizeof( PACKET ) );

        KmsLib::ThreadBase::Sleep_ms( 1000 );
    }


    KmsLib::ThreadBase::Sleep_ms( 30000 );

    printf( "Stopping\n" );

    KMS_TEST_COMPARE_RETURN( 0, lSetup.Stop() );
}
KMS_TEST_END

KMS_TEST_BEGIN(System_Base)
{
    Base lSetup;

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    OpenNet::System::Config   lC0;
    OpenNet::System::Config * lCNP = NULL;

    memset(&lC0, 0, sizeof(lC0));

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mSystem->GetConfig      (NULL ));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mSystem->GetInfo        (NULL ));
    KMS_TEST_COMPARE(OpenNet::STATUS_PACKET_TOO_SMALL         , lSetup.mSystem->SetConfig      (lC0  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mSystem->Adapter_Connect(NULL ));
    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_ADAPTER          , lSetup.mSystem->Adapter_Connect(reinterpret_cast<OpenNet::Adapter *>(1)));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mSystem->Display        (NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_NO_ADAPTER_CONNECTED     , lSetup.mSystem->Start          (0));
    KMS_TEST_COMPARE(OpenNet::STATUS_SYSTEM_NOT_STARTED       , lSetup.mSystem->Stop           ());

    #ifdef _KMS_WINDOWS_
        KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE    , lSetup.mSystem->SetConfig      (*lCNP));
    #endif

    KMS_TEST_ASSERT(NULL == lSetup.mSystem->Kernel_Get(0));

    lC0.mPacketSize_byte = 0xffffffff;

    KMS_TEST_COMPARE(OpenNet::STATUS_PACKET_TOO_LARGE, lSetup.mSystem->SetConfig(lC0));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mSystem->GetConfig(&lC0));

    KMS_TEST_ASSERT(0 < lC0.mPacketSize_byte);

    OpenNet::System::Info lI0;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mSystem->GetInfo(&lI0));

    KMS_TEST_ASSERT(0 < lI0.mSystemId);

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mSystem->SetConfig    (lC0));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mSystem->Display      (stdout));

    KMS_TEST_COMPARE(0, lSetup.mSystem->Kernel_GetCount());

    KMS_TEST_ASSERT(NULL == lSetup.mSystem->Adapter_Get(0));
    KMS_TEST_COMPARE(0, lSetup.mSystem->Adapter_GetCount());

    KMS_TEST_ASSERT(NULL == lSetup.mSystem->Processor_Get(0));
    KMS_TEST_COMPARE(0, lSetup.mSystem->Processor_GetCount());

    lC0.mPacketSize_byte--;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mSystem->SetConfig(lC0));
}
KMS_TEST_END

KMS_TEST_BEGIN(System_Display)
{
    Base lSetup;

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mSystem->Display(stdout));

    printf("QUESTION  Is the output OK? (y/n)\n");
    char lLine[1024];
    KMS_TEST_ASSERT( NULL != fgets(lLine, sizeof(lLine), stdin) );
    KMS_TEST_COMPARE(0, strncmp("y", lLine, 1));
}
KMS_TEST_END

// TEST INFO  System
//            Enumerate adapter<br>
//            Enumerate processor
KMS_TEST_BEGIN(System_SetupA)
{
    SetupA lSetup(0);

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_ASSERT(0 < lSetup.mSystem->Adapter_GetCount  ());
    KMS_TEST_ASSERT(0 < lSetup.mSystem->Processor_GetCount());
}
KMS_TEST_END
