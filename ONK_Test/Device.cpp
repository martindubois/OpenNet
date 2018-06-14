
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
#include <OpenNetK/Interface.h>

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
        lDH0.Control(OPEN_NET_IOCTL_CONNECT, NULL, 0, NULL, 0);
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
        OpenNet_Connect lC;

        lDH0.Control(OPEN_NET_IOCTL_CONNECT, &lC, sizeof(lC), NULL, 0);
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
