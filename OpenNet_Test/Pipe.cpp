
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Pipe.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsTest.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Filter_Forward.h>
#include <OpenNet/System.h>

// ===== OpenNet_Test =======================================================
#include "SetupC.h"
#include "Utilities.h"

// Configuration
/////////////////////////////////////////////////////////////////////////////

#define BUFFER_QTY       (   2)
#define PACKET_QTY       (   2)
#define PACKET_SIZE_byte (1500)

static const uint8_t PACKET[PACKET_SIZE_byte] =
{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88
};

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(Pipe_SetupC)
{
    SetupC lSetup(BUFFER_QTY);

    KMS_TEST_COMPARE_RETURN(0, lSetup.Init());

    KMS_TEST_COMPARE(0, lSetup.Stats_Reset());
    KMS_TEST_COMPARE(0, lSetup.Start      ());
    KMS_TEST_COMPARE(0, lSetup.Packet_Send(PACKET, sizeof(PACKET), PACKET_QTY));

    Sleep(2000);

    KMS_TEST_COMPARE(0, lSetup.Stats_GetAndDisplay());
    KMS_TEST_COMPARE(0, lSetup.Stop               (OpenNet::System::STOP_FLAG_LOOPBACK));
}
KMS_TEST_END
