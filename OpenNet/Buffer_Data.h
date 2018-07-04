
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Buffer_Data.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class Buffer_Data
{

public:

    Buffer_Data();

    uint64_t GetEventProfilingInfo(cl_profiling_info aParam);
    uint32_t GetMarkerValue       ();

    void ReleaseEvent    ();
    void ReleaseMemObject();

    void Reset           ();
    void ResetMarkerValue();

    void WaitForEvent();

    cl_event     mEvent    ;
    cl_mem       mMem      ;
    unsigned int mPacketQty;

private:

    uint32_t mMarkerValue;

};
