
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Filter_Data.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== Includes ===========================================================
#include <OpenNet/Filter.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class Filter_Data
{

public:

    Filter_Data();

    void Release();

    void Reset();

    cl_command_queue  mCommandQueue;
    OpenNet::Filter * mFilter      ;
    cl_kernel         mKernel      ;
    cl_program        mProgram     ;

};
