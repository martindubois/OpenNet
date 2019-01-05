
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Test/Device.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <stdint.h>

#ifdef _KMS_LINUX_
    #include <fcntl.h>
#endif

#ifdef _KMS_WINDOWS_
    // ===== Windows ============================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/DriverHandle.h>
#include <KmsLib/Exception.h>
#include <KmsTest.h>

// ===== Includes ===========================================================
#include <OpenNetK/Adapter_Types.h>

#ifdef _KMS_WINDOWS_
    #include <OpenNetK/Interface.h>
#endif

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
    KmsLib::DriverHandle lDH0;

    #ifdef _KMS_LINUX_
        lDH0.Connect("/dev/OpenNet0", O_RDWR);
    #endif

    #ifdef _KMS_WINDOWS_
        lDH0.Connect(OPEN_NET_DRIVER_INTERFACE, 0, GENERIC_ALL, 0);
    #endif

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
