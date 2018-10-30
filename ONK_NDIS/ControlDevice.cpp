
// Author   KMS - Martin Dubois, ing
// Product  OpenNet
// File     ONK_NDIS/ControlDevice.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Adapter.h>
#include <OpenNetK/Adapter_WDF.h>

// ===== ONDK_NDIS ==========================================================
#include "ControlDevice.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Adapter mAdapter    ;
    OpenNetK::Adapter mAdapter_WDF;
}
ControlDeviceContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(ControlDeviceContext, GetControlDeviceContext)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================

// Functions
/////////////////////////////////////////////////////////////////////////////

NTSTATUS ControlDevice_Create(PWDFDEVICE_INIT aDeviceInit)
{
    DbgPrintEx(DEBUG_ID, DEBUG_FUNCTION, PREFIX __FUNCTION__ "()" DEBUG_EOL);

    ASSERT(NULL != aDeviceInit);

    // TODO  ONK_NDIS
    //       Normal (Feature)
    (void)(aDeviceInit);

    return STATUS_SUCCESS;
}