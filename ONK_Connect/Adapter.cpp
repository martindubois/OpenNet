
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Connect/Adapter.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Common =============================================================
#include "../Common/Version.h"

// ===== ONK_Connect ========================================================
#include "Driver_WDM.h"

#include "Adapter.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    NDIS_HANDLE    mDeviceHandle;
    PDEVICE_OBJECT mDeviceObject;
}
AdapterContext;

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

extern "C"
{
    static MINIPORT_CANCEL_DIRECT_OID_REQUEST CancelDirectOidRequest;
    static MINIPORT_CANCEL_OID_REQUEST        CancelOidRequest      ;
    static MINIPORT_CANCEL_SEND               CancelSend            ;
    static MINIPORT_DEVICE_PNP_EVENT_NOTIFY   DevicePnPEventNotify  ;
    static MINIPORT_DIRECT_OID_REQUEST        DirectOidRequest      ;
    static MINIPORT_HALT                      Halt                  ;
    static MINIPORT_INITIALIZE                Initialize            ;
    static MINIPORT_OID_REQUEST               OidRequest            ;
    static MINIPORT_PAUSE                     Pause                 ;
    static MINIPORT_RETURN_NET_BUFFER_LISTS   ReturnNetBufferLists  ;
    static MINIPORT_SEND_NET_BUFFER_LISTS     SendNetBufferLists    ;
    static MINIPORT_SHUTDOWN                  Shutdown              ;
};

// Functions
/////////////////////////////////////////////////////////////////////////////

void Adapter_InitCharacteristics(NDIS_MINIPORT_DRIVER_CHARACTERISTICS * aChar)
{
    ASSERT(NULL != aChar);

    memset(aChar, 0, sizeof(NDIS_MINIPORT_DRIVER_CHARACTERISTICS));

    aChar->Header.Type     = NDIS_OBJECT_TYPE_MINIPORT_DRIVER_CHARACTERISTICS;
    aChar->Header.Revision = NDIS_MINIPORT_DRIVER_CHARACTERISTICS_REVISION_3;
    aChar->Header.Size     = NDIS_SIZEOF_MINIPORT_DRIVER_CHARACTERISTICS_REVISION_3;

    aChar->MajorNdisVersion =  6;
    aChar->MinorNdisVersion = 82;

    aChar->MajorDriverVersion = VERSION_MAJOR;
    aChar->MinorDriverVersion = VERSION_MINOR;

    aChar->Flags = NDIS_WDM_DRIVER;

    aChar->InitializeHandlerEx           = Initialize            ;
    aChar->HaltHandlerEx                 = Halt                  ;
    aChar->PauseHandler                  = Pause                 ;
    aChar->OidRequestHandler             = OidRequest            ;
    aChar->SendNetBufferListsHandler     = SendNetBufferLists    ;
    aChar->ReturnNetBufferListsHandler   = ReturnNetBufferLists  ;
    aChar->CancelSendHandler             = CancelSend            ;
    aChar->DevicePnPEventNotifyHandler   = DevicePnPEventNotify  ;
    aChar->ShutdownHandlerEx             = Shutdown              ;
    aChar->CancelOidRequestHandler       = CancelOidRequest      ;
    aChar->DirectOidRequestHandler       = DirectOidRequest      ;
    aChar->CancelDirectOidRequestHandler = CancelDirectOidRequest;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

void CancelDirectOidRequest(NDIS_HANDLE aAdapterContext, PVOID aRequestId)
{
    ASSERT(NULL != aAdapterContext);
    ASSERT(NULL != aRequestId     );

    // TODO  ONK_Connect.Miniport
    (void)(aAdapterContext);
    (void)(aRequestId     );
}

void CancelOidRequest(NDIS_HANDLE aHandle, PVOID aRequest)
{
    ASSERT(NULL != aHandle );
    ASSERT(NULL != aRequest);

    // TODO  ONK_Connect.Miniport
    (void)(aHandle );
    (void)(aRequest);
}

void CancelSend(NDIS_HANDLE aAdapterContext, PVOID aCancelId)
{
    ASSERT(NULL != aAdapterContext);
    ASSERT(NULL != aCancelId      );

    // TODO  ONK_Connect.Miniport
    (void)(aAdapterContext);
    (void)(aCancelId      );
}

void DevicePnPEventNotify(NDIS_HANDLE aAdapterContext, PNET_DEVICE_PNP_EVENT aEvent)
{
    ASSERT(NULL != aAdapterContext);
    ASSERT(NULL != aEvent         );

    // TODO  ONK_Connect.Miniport
    (void)(aAdapterContext);
    (void)(aEvent         );
}

NDIS_STATUS DirectOidRequest(NDIS_HANDLE aAdapterContext, PNDIS_OID_REQUEST aRequest)
{
    ASSERT(NULL != aAdapterContext);
    ASSERT(NULL != aRequest       );

    // TODO  ONK_Connect.Miniport
    (void)(aAdapterContext);
    (void)(aRequest       );

    return STATUS_SUCCESS;
}

void Halt(NDIS_HANDLE aAdapterContext, NDIS_HALT_ACTION aAction)
{
    ASSERT(NULL != aAdapterContext);

    // TODO  ONK_Connect.Miniport
    (void)(aAdapterContext);
    (void)(aAction        );
}

NDIS_STATUS Initialize(NDIS_HANDLE aHandle, NDIS_HANDLE aDriverContext, PNDIS_MINIPORT_INIT_PARAMETERS aParams)
{
    ASSERT(NULL != aHandle       );
    ASSERT(NULL != aDriverContext);
    ASSERT(NULL != aParams       );

    // Driver_WDM * lDriverContext = reinterpret_cast<Driver_WDM *>(aDriverContext);

    // TODO  ONK_Connect.Adapter

    NDIS_DEVICE_OBJECT_ATTRIBUTES lAttr;

    memset(&lAttr, 0, sizeof(lAttr));

    lAttr.Header.Revision = NDIS_DEVICE_OBJECT_ATTRIBUTES_REVISION_1;
    lAttr.Header.Size     = NDIS_SIZEOF_DEVICE_OBJECT_ATTRIBUTES_REVISION_1;
    lAttr.Header.Type     = NDIS_OBJECT_TYPE_DEVICE_OBJECT_ATTRIBUTES;

    /* TODO  ONK_Connect.Adapter
    lAttr.DefaultSDDLString = ;
    lAttr.DeviceClassGuid   = ;
    lAttr.DeviceName        = ;
    lAttr.ExtensionSize     = sizeof(AdapterContext);
    lAttr.MajorFunctions    = ;
    lAttr.SymbolicName      = ;

    NDIS_STATUS lStatus = NdisRegisterDeviceEx(lDriverContext->mDriverHandle, &lAttr, &lDevObj, &lDevHandle);
    ASSERT(STATUS_SUCCESS == lStatus);
    */
    (void)(aHandle       );
    (void)(aDriverContext);
    (void)(aParams       );

    return STATUS_SUCCESS;
}

NDIS_STATUS OidRequest(NDIS_HANDLE aAdapterContext, PNDIS_OID_REQUEST aRequest)
{
    ASSERT(NULL != aAdapterContext);
    ASSERT(NULL != aRequest       );

    // TODO  ONK_Connect.Miniport
    (void)(aAdapterContext);
    (void)(aRequest       );

    return STATUS_SUCCESS;
}

NDIS_STATUS Pause(NDIS_HANDLE aAdapterContext, PNDIS_MINIPORT_PAUSE_PARAMETERS aParams)
{
    ASSERT(NULL != aAdapterContext);
    ASSERT(NULL != aParams        );

    // TODO  ONK_Connect.Miniport
    (void)(aAdapterContext);
    (void)(aParams        );

    return STATUS_SUCCESS;
}

void ReturnNetBufferLists(NDIS_HANDLE aAdapterContext, PNET_BUFFER_LIST aList, ULONG aFlags)
{
    ASSERT(NULL != aAdapterContext);
    ASSERT(NULL != aList          );

    // TODO  ONK_Connect.Miniport
    (void)(aAdapterContext);
    (void)(aList          );
    (void)(aFlags         );
}

void SendNetBufferLists(NDIS_HANDLE aAdapterContext, PNET_BUFFER_LIST aList, NDIS_PORT_NUMBER aPortNumber, ULONG aFlags)
{
    ASSERT(NULL != aAdapterContext);
    ASSERT(NULL != aList          );

    // TODO  ONK_Connect.Miniport
    (void)(aAdapterContext);
    (void)(aList          );
    (void)(aPortNumber    );
    (void)(aFlags         );
}

void Shutdown(NDIS_HANDLE aHandle, NDIS_SHUTDOWN_ACTION aAction)
{
    ASSERT(NULL != aHandle);

    // TODO  ONK_Connect.Miniport
    (void)(aHandle);
    (void)(aAction);
}
