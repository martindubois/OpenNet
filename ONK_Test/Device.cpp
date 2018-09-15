
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Test/Device.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>
#include <KmsLib/Windows/DriverHandle.h>
#include <KmsTest.h>

// ===== Includes ===========================================================
#include <OpenNetK/Adapter_Types.h>
#include <OpenNetK/Interface.h>

// ===== Common =============================================================
#include "../Common/IoCtl.h"

// Tests
/////////////////////////////////////////////////////////////////////////////

// TEST INFO  Device
//            Invalid IoCtl code<br>
//            No expected input buffer<br>
//            Invalid event handle
KMS_TEST_BEGIN(Device_SetupA)
{
    KmsLib::Windows::DriverHandle lDH0;

    lDH0.Connect(OPEN_NET_DRIVER_INTERFACE, 0, GENERIC_ALL, 0);

    try
    {
        lDH0.Control(0, NULL, 0, NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        lDH0.Control(IOCTL_CONFIG_GET, NULL, 0, NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        lDH0.Control(IOCTL_CONFIG_SET, NULL, 0, NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        OpenNetK::Adapter_Config lC;

        memset(&lC, 0, sizeof(lC));

        lDH0.Control(IOCTL_CONFIG_SET, &lC, sizeof(lC), NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        lDH0.Control(IOCTL_CONNECT, NULL, 0, NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        IoCtl_Connect_In lC;

        memset(&lC, 0, sizeof(lC));

        lDH0.Control(IOCTL_CONNECT, &lC, sizeof(lC), NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        lDH0.Control(IOCTL_INFO_GET, NULL, 0, NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        lDH0.Control(IOCTL_PACKET_SEND_EX, NULL, 0, NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        lDH0.Control(IOCTL_START, NULL, 0, NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        lDH0.Control(IOCTL_STATE_GET, NULL, 0, NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        lDH0.Control(IOCTL_STATISTICS_GET, NULL, 0, NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }

    try
    {
        lDH0.Control(IOCTL_STOP, NULL, 0, NULL, 0);
        KMS_TEST_ERROR();
    }
    catch (KmsLib::Exception * eE)
    {
        KMS_TEST_ERROR_INFO;
        eE->Write(stdout);
        KMS_TEST_COMPARE(KmsLib::Exception::CODE_IOCTL_ERROR, eE->GetCode());
    }
}
KMS_TEST_END_2
