
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/SetupTool.cpp

// TEST COVERAGE  2019-04-02  Martin Dubois, ing.

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

#ifdef _KMS_WINDOWS_

    // ===== Windows ========================================================

    #include <Windows.h>

    #include <SetupAPI.h>

#endif

// ===== Includes ===========================================================
#include <OpenNet/SetupTool.h>

// ===== OpenNet ============================================================

#ifdef _KMS_LINUX_
    #include "SetupTool_Linux.h"
#endif

#ifdef _KMS_WINDOWS_
    #include "SetupTool_Windows.h"
#endif

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    // NOT TESTED  OpenNet.SetupTool.Error
    //             SetupTool_... constructor throw an exception
    SetupTool * SetupTool::Create(bool aDebug)
    {
        SetupTool * lResult;

        try
        {
            #ifdef _KMS_LINUX_
                lResult = new SetupTool_Linux(aDebug);
            #endif

            #ifdef _KMS_WINDOWS_
                lResult = new SetupTool_Windows(aDebug);
            #endif
        }
        catch (...)
        {
            lResult = NULL;
        }

        return lResult;
    }

    void SetupTool::Delete()
    {
        assert(NULL != this);

        delete this;
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    SetupTool::SetupTool()
    {
    }

    SetupTool::~SetupTool()
    {
    }

}
