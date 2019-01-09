
// Author   KMS - Martin Dubois, ing
// Product  OpenNet
// File     ONK_NDIS/NdisDevice.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== NetAdapterCx =======================================================
#include <netadaptercx.h>

// ===== ONDK_NDIS ==========================================================
#include "NdisAdapter.h"
#include "VirtualHardware.h"

#include "NdisDevice.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    void          * mAdapter ;
    VirtualHardware mHardware;
}
NdisDeviceContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(NdisDeviceContext, GetNdisDeviceContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
extern "C"
{
    static EVT_WDF_DEVICE_ARM_WAKE_FROM_SX    ArmWakeFromSx   ;
    static EVT_WDF_DEVICE_D0_ENTRY            D0Entry         ;
    static EVT_WDF_DEVICE_D0_EXIT             D0Exit          ;
    static EVT_WDF_DEVICE_DISARM_WAKE_FROM_SX DisarmWakeFromSx;
    static EVT_WDF_DEVICE_PREPARE_HARDWARE    PrepareHardware ;
    static EVT_WDF_DEVICE_RELEASE_HARDWARE    ReleaseHardware ;
}

// Functions
/////////////////////////////////////////////////////////////////////////////

NTSTATUS NdisDevice_Create(WDFDEVICE_INIT * aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDeviceInit);

    NTSTATUS lStatus = NetAdapterDeviceInitConfig(aDeviceInit);
    ASSERT(STATUS_SUCCESS == lStatus);

    WDF_OBJECT_ATTRIBUTES lAttributes;

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttributes, NdisDeviceContext);

    WDF_PNPPOWER_EVENT_CALLBACKS lPnpPowerCallbacks;

    WDF_PNPPOWER_EVENT_CALLBACKS_INIT(&lPnpPowerCallbacks);

    lPnpPowerCallbacks.EvtDeviceD0Entry         = D0Entry        ;
    lPnpPowerCallbacks.EvtDeviceD0Exit          = D0Exit         ;
    lPnpPowerCallbacks.EvtDevicePrepareHardware = PrepareHardware;
    lPnpPowerCallbacks.EvtDeviceReleaseHardware = ReleaseHardware;

    WdfDeviceInitSetPnpPowerEventCallbacks(aDeviceInit, &lPnpPowerCallbacks);

    WDF_POWER_POLICY_EVENT_CALLBACKS lPowerCallbacks;

    WDF_POWER_POLICY_EVENT_CALLBACKS_INIT(&lPowerCallbacks);

    lPowerCallbacks.EvtDeviceArmWakeFromSx    = ArmWakeFromSx   ;
    lPowerCallbacks.EvtDeviceDisarmWakeFromSx = DisarmWakeFromSx;

    WdfDeviceInitSetPowerPolicyEventCallbacks(aDeviceInit, &lPowerCallbacks);

    // WdfDeviceInitSetPowerPolicyOwnership(aDeviceInit, TRUE);

    WDFDEVICE lDevice;

    lStatus = WdfDeviceCreate(&aDeviceInit, &lAttributes, &lDevice);
    ASSERT(STATUS_SUCCESS == lStatus);
    ASSERT(NULL           != lDevice);

    WdfDeviceSetAlignmentRequirement(lDevice, FILE_256_BYTE_ALIGNMENT);

    WDF_DEVICE_POWER_POLICY_WAKE_SETTINGS lWakeSettings;

    WDF_DEVICE_POWER_POLICY_WAKE_SETTINGS_INIT(&lWakeSettings);

    lStatus = WdfDeviceAssignSxWakeSettings(lDevice, &lWakeSettings);
    ASSERT(STATUS_SUCCESS == lStatus);

    /* WDF_DEVICE_POWER_CAPABILITIES lPowerCapabilities;

    WDF_DEVICE_POWER_CAPABILITIES_INIT(&lPowerCapabilities);

    lPowerCapabilities.DeviceD1 = WdfTrue;
    lPowerCapabilities.DeviceD2 = WdfTrue;
    lPowerCapabilities.DeviceState[PowerSystemWorking  ] = PowerDeviceD0;
    lPowerCapabilities.DeviceState[PowerSystemSleeping1] = PowerDeviceD1;
    lPowerCapabilities.DeviceState[PowerSystemSleeping2] = PowerDeviceD2;
    lPowerCapabilities.DeviceState[PowerSystemSleeping3] = PowerDeviceD2;
    lPowerCapabilities.DeviceState[PowerSystemHibernate] = PowerDeviceD3;
    lPowerCapabilities.DeviceState[PowerSystemShutdown ] = PowerDeviceD3;
    lPowerCapabilities.DeviceWake = PowerDeviceD3      ;
    lPowerCapabilities.SystemWake = PowerSystemShutdown;
    lPowerCapabilities.WakeFromD0 = WdfTrue;
    lPowerCapabilities.WakeFromD1 = WdfTrue;
    lPowerCapabilities.WakeFromD2 = WdfTrue;
    lPowerCapabilities.WakeFromD3 = WdfTrue;

    WdfDeviceSetPowerCapabilities(lDevice, &lPowerCapabilities); */

    NdisDeviceContext * lThis = GetNdisDeviceContext(lDevice);
    ASSERT(NULL != lThis);

    new (&lThis->mHardware) VirtualHardware();

    return NdisAdapter_Create(lDevice, &lThis->mAdapter, &lThis->mHardware);
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

NTSTATUS ArmWakeFromSx(WDFDEVICE aDevice)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDevice);

    UNREFERENCED_PARAMETER(aDevice);

    return STATUS_SUCCESS;
}

NTSTATUS D0Entry(WDFDEVICE aDevice, WDF_POWER_DEVICE_STATE aPreviousState)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice       );
    ASSERT(NULL != aPreviousState);

    UNREFERENCED_PARAMETER(aDevice       );
    UNREFERENCED_PARAMETER(aPreviousState);

    // TODO  ONK_NDIS
    //       Normal (Feature)

    return STATUS_SUCCESS;
}

NTSTATUS D0Exit(WDFDEVICE aDevice, WDF_POWER_DEVICE_STATE aTargetState)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice     );
    ASSERT(NULL != aTargetState);

    UNREFERENCED_PARAMETER(aDevice     );
    UNREFERENCED_PARAMETER(aTargetState);

    // TODO  ONK_NDIS
    //       Normal (Feature)

    return STATUS_SUCCESS;
}

VOID DisarmWakeFromSx(WDFDEVICE aDevice)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

    ASSERT(NULL != aDevice);

    UNREFERENCED_PARAMETER(aDevice);
}

NTSTATUS PrepareHardware(WDFDEVICE aDevice, WDFCMRESLIST aRaw, WDFCMRESLIST aTranslated)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( , ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice    );
    ASSERT(NULL != aRaw       );
    ASSERT(NULL != aTranslated);

    UNREFERENCED_PARAMETER(aDevice    );
    UNREFERENCED_PARAMETER(aRaw       );
    UNREFERENCED_PARAMETER(aTranslated);

    return STATUS_SUCCESS;
}

NTSTATUS ReleaseHardware(WDFDEVICE aDevice, WDFCMRESLIST aTranslated)
{
    DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

    ASSERT(NULL != aDevice    );
    ASSERT(NULL != aTranslated);

    UNREFERENCED_PARAMETER(aDevice    );
    UNREFERENCED_PARAMETER(aTranslated);

    return STATUS_SUCCESS;
}
