
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Connect/Drivers.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== ONK_Connect ========================================================
#include "Adapter.h"

#include "Driver_WDM.h"

// Static variables
/////////////////////////////////////////////////////////////////////////////

static Driver_WDM sDriverContext;

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

extern "C"
{
    static MINIPORT_UNLOAD Unload;
}

// Entry point declaration
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    NDIS_STATUS DriverEntry(PVOID aDriverObject, PVOID aRegistryPath);
}

// Entry point
/////////////////////////////////////////////////////////////////////////////

#pragma NDIS_INIT_FUNCTION(DriverEntry)

NDIS_STATUS DriverEntry(PVOID aDriverObject, PVOID aRegistryPath)
{
    ASSERT(NULL != aDriverObject);
    ASSERT(NULL != aRegistryPath);

    memset(&sDriverContext, 0, sizeof(sDriverContext));

    NDIS_MINIPORT_DRIVER_CHARACTERISTICS lChar;

    Adapter_InitCharacteristics(&lChar);

    lChar.UnloadHandler = Unload;

    // NdisMRegisterMiniportDriver ==> NdisMDeregisterMiniportDriver  See Unload

    NDIS_STATUS lStatus = NdisMRegisterMiniportDriver(reinterpret_cast<PDRIVER_OBJECT>(aDriverObject), reinterpret_cast<PUNICODE_STRING>(aRegistryPath), &sDriverContext, &lChar, &sDriverContext.mDriverHandle);
    ASSERT(STATUS_SUCCESS == lStatus);
    (void)(lStatus);

    return STATUS_SUCCESS;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

#pragma NDIS_PAGEABLE_FUNCTION(Unload)

void Unload(PDRIVER_OBJECT aDriver)
{
    ASSERT(NULL != aDriver);

    ASSERT(NULL != sDriverContext.mDriverHandle);

    (void)(aDriver);

    // TODO  ONC_Connect
    //       Move to Adapter.cpp<br>
    //       NdisDeregisterDeviceEx(sDriverContext.mDevice);

    NdisMDeregisterMiniportDriver(sDriverContext.mDriverHandle);
}
