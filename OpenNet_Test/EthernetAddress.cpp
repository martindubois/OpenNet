
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/EthernetAddress.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Common =============================================================
#include "../Common/OpenNet/EthernetAddress.h"

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(EthernetAddress_Base)
{
    OpenNet_EthernetAddress   lEA;
    OpenNet_EthernetAddress * lEAP = NULL;

    KMS_TEST_COMPARE(false, OpenNet::EthernetAddress_IsBroadcast(*lEAP));
    KMS_TEST_COMPARE(false, OpenNet::EthernetAddress_IsMulticast(*lEAP));
    KMS_TEST_COMPARE(false, OpenNet::EthernetAddress_IsZero     (*lEAP));

    memset(&lEA.mAddress, 0xff, sizeof(lEA.mAddress));

    KMS_TEST_COMPARE(true, OpenNet::EthernetAddress_IsBroadcast(lEA));

    lEA.mAddress[0] = 0x01;
    lEA.mAddress[1] = 0x00;
    lEA.mAddress[2] = 0x5e;
    lEA.mAddress[3] = 0x00;

    KMS_TEST_COMPARE(true, OpenNet::EthernetAddress_IsMulticast(lEA));
}
KMS_TEST_END
