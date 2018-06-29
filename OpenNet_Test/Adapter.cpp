
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Adapte.cpp

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
#include "../Common/OpenNet/EthernetAddress.h"

// ====== OpenNet_Test ======================================================
#include "Utilities.h"

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
    OpenNet::Adapter::Stats    lSs;
    OpenNet::Adapter::Stats  * lSsP = NULL;

    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE        , OpenNet::Adapter::Display(*lCP , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, OpenNet::Adapter::Display( lC  , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , OpenNet::Adapter::Display( lC  , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE        , OpenNet::Adapter::Display(*lIP , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, OpenNet::Adapter::Display( lI  , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , OpenNet::Adapter::Display( lI  , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE        , OpenNet::Adapter::Display(*lSeP, NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, OpenNet::Adapter::Display( lSe , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , OpenNet::Adapter::Display( lSe , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE        , OpenNet::Adapter::Display(*lSsP, NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, OpenNet::Adapter::Display( lSs , NULL  ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , OpenNet::Adapter::Display( lSs , stdout));

    memset(&lC , 0, sizeof(lC ));
    memset(&lI , 0, sizeof(lI ));
    memset(&lSe, 0, sizeof(lSe));
    memset(&lSs, 0, sizeof(lSs));

    lC.mPacketSize_byte = 4096;
    lI.mPacketSize_byte = 4096;

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lC , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lI , stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lSe, stdout));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK, OpenNet::Adapter::Display(lSs, stdout));
}
KMS_TEST_END

KMS_TEST_BEGIN(Adapter_SetupA)
{
    OpenNet::System * lSystem = OpenNet::System::Create();
    KMS_TEST_ASSERT_RETURN(NULL != lSystem);

    OpenNet::Adapter * lA0 = lSystem->Adapter_Get(0);
    KMS_TEST_ASSERT(NULL != lA0);

    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_BUFFER_COUNT, lA0->Buffer_Allocate(0));
    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_BUFFER_COUNT, lA0->Buffer_Release (0));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->Display(stdout));

    OpenNet::Adapter::Config lC0;

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lA0->GetConfig(NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lA0->GetConfig(&lC0));

    KMS_TEST_ASSERT (OpenNet::EthernetAddress_IsZero(lC0.mEthernetAddress[0]));
    KMS_TEST_COMPARE(OPEN_NET_MODE_NORMAL, lC0.mMode);

    KMS_TEST_ASSERT(OPEN_NET_PACKET_SIZE_MAX_byte >= lC0.mPacketSize_byte);
    KMS_TEST_ASSERT(OPEN_NET_PACKET_SIZE_MIN_byte <= lC0.mPacketSize_byte);
    
    OpenNet::Adapter::Info lI0;

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lA0->GetInfo(NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lA0->GetInfo(&lI0));

    KMS_TEST_COMPARE(OPEN_NET_ADAPTER_TYPE_ETHERNET, lI0.mAdapterType);

    KMS_TEST_COMPARE(false, OpenNet::EthernetAddress_IsBroadcast(lI0.mEthernetAddress));
    KMS_TEST_COMPARE(false, OpenNet::EthernetAddress_IsMulticast(lI0.mEthernetAddress));
    KMS_TEST_COMPARE(false, OpenNet::EthernetAddress_IsZero     (lI0.mEthernetAddress));

    KMS_TEST_ASSERT(OPEN_NET_PACKET_SIZE_MAX_byte >= lI0.mPacketSize_byte);
    KMS_TEST_ASSERT(OPEN_NET_PACKET_SIZE_MIN_byte <= lI0.mPacketSize_byte);

    OpenNet::Adapter::State lSe0;

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lA0->GetState(NULL ));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lA0->GetState(&lSe0));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->ResetStats());

    OpenNet::Adapter::Stats lSs0;

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lA0->GetStats(NULL , false));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK                       , lA0->GetStats(&lSs0, false));

    OpenNet::Adapter::Stats lSsE;
    OpenNet::Adapter::Stats lSsM;

    Utl_ValidateInit(&lSsE, &lSsM);

    lSsE.mDriver.mAdapter.mIoCtl              = 1;
    lSsE.mDriver.mAdapter_NoReset.mIoCtl_Last = OPEN_NET_IOCTL_STATS_RESET;

    lSsM.mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset = UTL_MASK_ABOVE;
    lSsM.mDriver.mHardware_NoReset.mStats_Reset      = UTL_MASK_ABOVE;

    KMS_TEST_COMPARE(0, Utl_Validate(lSs0, lSsE, lSsM));

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lA0->Packet_Send(NULL, 0));
    KMS_TEST_COMPARE(OpenNet::STATUS_PACKET_TOO_SMALL         , lA0->Packet_Send(""  , 0));

    uint8_t lPacket[20];

    memset(&lPacket, 0xff, sizeof(lPacket));

    KMS_TEST_COMPARE(OpenNet::STATUS_OK, lA0->Packet_Send(lPacket, sizeof(lPacket)));

    KMS_TEST_COMPARE(OpenNet::STATUS_FILTER_NOT_SET, lA0->ResetInputFilter());

    KMS_TEST_COMPARE(OpenNet::STATUS_PROCESSOR_NOT_SET, lA0->ResetProcessor());

    OpenNet::Adapter::Config * lCP = NULL;

    KMS_TEST_COMPARE(OpenNet::STATUS_INVALID_REFERENCE, lA0->SetConfig(*lCP));
    KMS_TEST_COMPARE(OpenNet::STATUS_OK               , lA0->SetConfig( lC0));

    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lA0->SetInputFilter(NULL));
    KMS_TEST_COMPARE(OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT, lA0->SetProcessor  (NULL));

    lSystem->Delete();
}
KMS_TEST_END
