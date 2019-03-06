
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

    void SpinLock::SetLock(void * aLock)
    {
        ASSERT(NULL != aLock);

        mLock = aLock;
    }

    void SpinLock::SetOSDep(OpenNetK_OSDep * aOSDep)
    {
        ASSERT(NULL != aOSDep);

        mOSDep = aOSDep;
    }

    uint32_t SpinLock::LockFromThread()
    {
        ASSERT( NULL != mLock                          );
        ASSERT( NULL != mOSDep                         );
        ASSERT( NULL != mOSDep->LockSpinlockFromThread );

        return mOSDep->LockSpinlockFromThread( mLock );
    }

    void SpinLock::UnlockFromThread( uint32_t aFlags )
    {
        ASSERT( NULL != mLock                            );
        ASSERT( NULL != mOSDep                           );
        ASSERT( NULL != mOSDep->UnlockSpinlockFromThread );

        mOSDep->UnlockSpinlockFromThread( mLock, aFlags );
    }

}
