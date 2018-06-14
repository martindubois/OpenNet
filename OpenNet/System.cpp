
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/System.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNet ===================================================
#include <OpenNet/System.h>

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet ============================================================
#include "System_Internal.h"

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    // NOT TESTED  System.ErrorHandling
    //             System_Internal contructor raise an exception
    System * System::Create()
    {
        System * lResult;

        try
        {
            lResult = new System_Internal();
        }
        catch (...)
        {
            lResult = NULL;
        }

        return lResult;
    }

    void System::Delete()
    {
        delete this;
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
