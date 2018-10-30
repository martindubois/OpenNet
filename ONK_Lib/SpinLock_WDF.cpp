
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Lib/SpinLock_WDF.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================

#define INITGUID

#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>

// ===== Includes ===========================================================
#include <OpenNetK/StdInt.h>

#include <OpenNetK/SpinLock_WDF.h>

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    SpinLock_WDF::SpinLock_WDF(WDFDEVICE aDevice)
    {
        ASSERT(NULL != aDevice);

        WDF_OBJECT_ATTRIBUTES lAttr;

        WDF_OBJECT_ATTRIBUTES_INIT(&lAttr);

        lAttr.ParentObject = aDevice;

        NTSTATUS lStatus = WdfSpinLockCreate(&lAttr, &mSpinLock);
        ASSERT(STATUS_SUCCESS == lStatus  );
        ASSERT(NULL           != mSpinLock);
        (void)(lStatus);
    }

    // ===== SpinLock =======================================================

    // CRITICAL PATH - Buffer
    void SpinLock_WDF::Lock()
    {
        ASSERT(NULL != mSpinLock);

        WdfSpinLockAcquire(mSpinLock);
    }

    // CRITICAL PATH - Buffer
    void SpinLock_WDF::Unlock()
    {
        ASSERT(NULL != mSpinLock);

        WdfSpinLockRelease(mSpinLock);
    }

};
