
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Processor_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <map>

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== Import/Includes ====================================================
#include <KmsLib/DebugLog.h>

// ===== Includes ===========================================================
#include <OpenNet/Filter.h>
#include <OpenNet/Processor.h>
#include <OpenNetK/Adapter_Types.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class Processor_Internal : public OpenNet::Processor
{

public:

    typedef struct
    {
        cl_event     mEvent      ;
        uint32_t     mMarkerValue;
        cl_mem       mMem        ;
        unsigned int mPacketQty  ;
        unsigned int mSize_byte  ;
    }
    BufferData;

    typedef struct
    {
        clEnqueueMakeBuffersResidentAMD_fn mEnqueueMakeBufferResident;
        clEnqueueWaitSignalAMD_fn          mEnqueueWaitSignal        ;
    }
    ExtensionFunctions;

    typedef struct
    {
        cl_command_queue  mCommandQueue;
        OpenNet::Filter * mFilter      ;
        cl_kernel         mKernel      ;
        cl_program        mProgram     ;
    }
    FilterData;

    Processor_Internal(cl_platform_id aPlatform, cl_device_id aDevice, ExtensionFunctions * aExtensionFunctions, KmsLib::DebugLog * aDebugLog);

    ~Processor_Internal();

    void Buffer_Allocate(unsigned int aPacketSize_byte, FilterData * aFilterData, OpenNetK::Buffer * aBuffer, BufferData * aBufferData);
    void Buffer_Release (BufferData * aBufferData);

    void Processing_Create (FilterData * aFilterData, OpenNet::Filter * aFilter);
    void Processing_Queue  (FilterData * aFilterData, BufferData * aBufferData);
    void Processing_Release(FilterData * aFilterData);
    void Processing_Wait   (FilterData * aFilterData, BufferData * aBufferData);

    // ===== OpenNet::Processor =============================================
    virtual OpenNet::Status GetInfo        (Info * aOut) const;
    virtual const char    * GetName        () const;
    virtual OpenNet::Status GetStatistics  (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
    virtual OpenNet::Status ResetStatistics();
    virtual OpenNet::Status Display        (FILE * aOut) const;

private:

    void InitInfo();

    // ===== OpenCL =========================================================

    bool GetDeviceInfo(cl_device_info aParam);
    void GetDeviceInfo(cl_device_info aParam, size_t aOutSize_byte, void * aOut);

    void GetKernelWorkGroupInfo(cl_kernel aKernel, cl_kernel_work_group_info aParam, size_t aOutSize_byte, void * aOut);

    cl_context           mContext           ;
    KmsLib::DebugLog   * mDebugLog          ;
    cl_device_id         mDevice            ;
    ExtensionFunctions * mExtensionFunctions;
    Info                 mInfo              ;

};
