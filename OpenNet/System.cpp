
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/System.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== Includes ===========================================================
#include <OpenNet/System.h>

// ===== OpenNet ============================================================
#include "System_Internal.h"

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             System_Internal contructor raise an exception
    System * System::Create()
    {
        System * lResult;

        try
        {
            lResult = new System_Internal();
        }
        catch ( ... )
        {
            lResult = NULL;
        }

        return lResult;
    }

    void System::Delete()
    {
        try
        {
            delete this;
        }
        catch (...)
        {
        }
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    System::System()
    {
    }

    System::~System()
    {
    }

}
