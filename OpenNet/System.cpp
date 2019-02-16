
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/System.cpp

#define __CLASS__ "System::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ============================================================
    #include <Windows.h>

    // ===== OpenCL =============================================================
    #include <CL/opencl.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>
#include <KmsLib/ValueVector.h>

// ===== Includes ===========================================================
#include <OpenNet/System.h>

// ===== Common =============================================================
#include "../Common/OpenNet/System_Statistics.h"

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"

#ifdef _KMS_LINUX_
    #include "System_CUDA.h"
#endif

#ifdef _KMS_WINDOWS_
    #include "System_OpenCL.h"
#endif

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    const unsigned int System::START_FLAG_LOOPBACK = 0x00000001;

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             System_Internal contructor raise an exception
    System * System::Create()
    {
        System * lResult;

        try
        {
            #ifdef _KMS_LINUX_
                lResult = new System_CUDA();
            #endif

            #ifdef _KMS_WINDOWS_
                lResult = new System_OpenCL();
            #endif
        }
        catch ( KmsLib::Exception * eE )
        {
            printf( __CLASS__ "Create - Exception\n" );
            eE->Write( stdout );
        }
        catch ( ... )
        {
            printf( __CLASS__ "Create - Unknown exception\n" );
            lResult = NULL;
        }

        return lResult;
    }

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             System_Internal destructor raise an exception
    void System::Delete()
    {
        try
        {
            // printf( __CLASS__ "Delete - delete 0x%lx (this)\n", reinterpret_cast< uint64_t >( this ) );

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
