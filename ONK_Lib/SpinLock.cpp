
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Lib/SpinLock.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/OS.h>
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
