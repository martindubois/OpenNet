
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet_Test/Adapte.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/System.h>

// ====== Common ============================================================
#include "../Common/Constants.h"
#include "../Common/IoCtl.h"
#include "../Common/OpenNet/EthernetAddress.h"

// ===== OpenNet_Test =======================================================
#include "SetupA.h"

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Adapter_Base)
{
    OpenNet::Adapter::Config   lC ;
    OpenNet::Adapter::Config * lCP  = NULL;
    OpenNet::Adapter::Info     lI ;
    OpenNet::Adapter::Info   * lIP  = NULL;
    OpenNet::Adapter::State    lSe;
    OpenNet::Adapter::State  * lSeP = NULL;

    unsigned int lStatistics[1024];

    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE        , OpenNet::Adapter::Display(*lCP , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, OpenNet::Adapter::Display( lC  , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , OpenNet::Adapter::Display( lC  , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE        , OpenNet::Adapter::Display(*lIP , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, OpenNet::Adapter::Display( lI  , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , OpenNet::Adapter::Display( lI  , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE        , OpenNet::Adapter::Display(*lSeP, NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, OpenNet::Adapter::Display( lSe , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , OpenNet::Adapter::Display( lSe , stdout));

    memset(&lC , 0, sizeof(lC ));
    memset(&lI , 0, sizeof(lI ));
    memset(&lSe, 0, sizeof(lSe));

    memset(&lStatistics, 0, sizeof(lStatistics));

    lC.mPacketSize_byte = 4096;
    lI.mPacketSize_byte = 4096;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lC , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lI , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lSe, stdout));
}
KMS_TEST_END

KMS_TEST_BEGIN(Adapter_Display)
{
    OpenNet::Adapter::Config   lC ;
    OpenNet::Adapter::Info     lI ;
    OpenNet::Adapter::State    lSe;

    memset(&lC , 0, sizeof(lC));
    memset(&lI , 0, sizeof(lI));
    memset(&lSe, 0, sizeof(lSe));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lC , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lI , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lSe, stdout));

    printf("QUESTION  Is the output OK? (y/n)\n");
    char lLine[1024];
    KMS_TEST_ASSERT(NULL != fgets(lLine, sizeof(lLine), stdin));
    KMS_TEST_COMPARE(0, strncmp("y", lLine, 1));
}
KMS_TEST_END

KMS_TEST_BEGIN(Adapter_SetupA)
{
    SetupA lSetup(0);

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mAdapter->Display(stdout));

    OpenNet::Adapter::Config lC0;

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mAdapter->GetConfig(NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lSetup.mAdapter->GetConfig(&lC0));

    KMS_TEST_ASSERT(PACKET_SIZE_MAX_byte >= lC0.mPacketSize_byte);
    KMS_TEST_ASSERT(PACKET_SIZE_MIN_byte <= lC0.mPacketSize_byte);
    
    OpenNet::Adapter::Info lI0;

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mAdapter->GetInfo(NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lSetup.mAdapter->GetInfo(&lI0));

    KMS_TEST_COMPARE(OpenNetK::ADAPTER_TYPE_ETHERNET, lI0.mAdapterType);

    KMS_TEST_COMPARE(false, OpenNet::EthernetAddress_IsBroadcast(lI0.mEthernetAddress));
    KMS_TEST_COMPARE(false, OpenNet::EthernetAddress_IsMulticast(lI0.mEthernetAddress));
    KMS_TEST_COMPARE(false, OpenNet::EthernetAddress_IsZero     (lI0.mEthernetAddress));

    KMS_TEST_ASSERT(PACKET_SIZE_MAX_byte >= lI0.mPacketSize_byte);
    KMS_TEST_ASSERT(PACKET_SIZE_MIN_byte <= lI0.mPacketSize_byte);

    OpenNet::Adapter::State lSe0;

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mAdapter->GetState(NULL ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lSetup.mAdapter->GetState(&lSe0));

    KMS_TEST_COMPARE(0, lSetup.Statistics_Reset());

    unsigned int lSs0[1024];

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mAdapter->GetStatistics(NULL,            0));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mAdapter->GetStatistics(lSs0, sizeof(lSs0)));

    /*
    OpenNet::Adapter::Stats lSsE;
    OpenNet::Adapter::Stats lSsM;

    Utl_ValidateInit(&lSsE, &lSsM);

    lSsE.mDriver.mAdapter.mIoCtl              = 1;
    lSsE.mDriver.mAdapter_NoReset.mIoCtl_Last = IOCTL_STATISTICS_RESET;

    lSsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset = UTL_MASK_ABOVE;

    lSsM.mDriver.mHardware_NoReset.mStats_Reset = UTL_MASK_ABOVE;

    KMS_TEST_COMPARE(0, Utl_Validate(lSs0, lSsE, lSsM));
    */
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mAdapter->Packet_Send(NULL, 0));
    KMS_TEST_COMPARE(OpenNet::STATUS_PACKET_TOO_SMALL         , lSetup.mAdapter->Packet_Send(""  , 0));

    uint8_t lPacket[20];

    memset(&lPacket, 0xff, sizeof(lPacket));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lSetup.mAdapter->Packet_Send(lPacket, sizeof(lPacket)));

    KMS_TEST_COMPARE(OpenNet::STATUS_FILTER_NOT_SET, lSetup.mAdapter->ResetInputFilter());

    KMS_TEST_COMPARE(OpenNet::STATUS_PROCESSOR_NOT_SET, lSetup.mAdapter->ResetProcessor());

    OpenNet::Adapter::Config * lCP = NULL;

    // TODO  OpenNet.Adapter
    //       High (Issue) - This test cause a seg fault on linux when
    //       compiled with -O2<br>
    //       KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE, lSetup.mAdapter->SetConfig(*lCP));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK               , lSetup.mAdapter->SetConfig( lC0));

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mAdapter->SetInputFilter(NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lSetup.mAdapter->SetProcessor  (NULL));
}
KMS_TEST_END
