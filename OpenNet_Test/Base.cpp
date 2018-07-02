
// Author   KMA - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Base.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet_Test =======================================================
#include "Base.h"

// Public
/////////////////////////////////////////////////////////////////////////////

Base::Base() : mSystem(NULL)
{
}

Base::~Base()
{
    if (NULL != mSystem)
    {
        mSystem->Delete();
    }
}

int Base::Init()
{
    assert(NULL == mSystem);

    mSystem = OpenNet::System::Create();
    if (NULL == mSystem)
    {
        return __LINE__;
    }

    return 0;
}
