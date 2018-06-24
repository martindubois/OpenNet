
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Lib/SpinLock.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================

#define INITGUID

#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>

// ===== Includes ===========================================================
#include <OpenNetK/StdInt.h>

#include <OpenNetK/SpinLock.h>

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void * SpinLock::operator new(size_t aSize_byte, void * aAddress)
    {
        ASSERT(sizeof(SpinLock) <= aSize_byte);
        ASSERT(NULL             != aAddress  );

        (void)(aSize_byte);

        return aAddress;
    }

}
