
// Author / Auteur    KMS - Martin Dubois, ing.
// Product / Produit  OpenNet
// File / Fichier     ONK_NDIS/Driver.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================
#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>

// Entry point declaration / Declaration du point d'entre
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    NTSTATUS DriverEntry(PDRIVER_OBJECT aDrvObj, PUNICODE_STRING aRegPath);
}

// Entry point / Point d'entre
/////////////////////////////////////////////////////////////////////////////

#pragma alloc_text (INIT, DriverEntry)

NTSTATUS DriverEntry(PDRIVER_OBJECT aDrvObj, PUNICODE_STRING aRegPath)
{
    ASSERT(NULL != aDrvObj );
    ASSERT(NULL != aRegPath);

    // TODO  Dev
    (void)(aDrvObj );
    (void)(aRegPath);
    return STATUS_NOT_IMPLEMENTED;
}
