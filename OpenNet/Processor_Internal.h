
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <map>

#ifdef _KMS_WINDOWS_
    // ===== OpenCL =========================================================
    #include <CL/opencl.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/DebugLog.h>

// ===== Includes ===========================================================
#include <OpenNet/Kernel.h>
#include <OpenNet/Processor.h>
#include <OpenNetK/Adapter_Types.h>

class Buffer_Data     ;
class Thread          ;
class Thread_Functions;

// Class
/////////////////////////////////////////////////////////////////////////////

class Processor_Internal : public OpenNet::Processor
{

public:

    #ifdef _KMS_WINDOWS_
        Processor_Internal(cl_platform_id aPlatform, cl_device_id aDevice, KmsLib::DebugLog * aDebugLog);
    #endif

    ~Processor_Internal();

    #ifdef _KMS_WINDOWS_
        Buffer_Data * Buffer_Allocate(unsigned int aPacketSize_byte, cl_command_queue aCommandQueue, cl_kernel aKernel, OpenNetK::Buffer * aBuffer);

        cl_command_queue CommandQueue_Create(bool aProfilingEnabled);

        cl_program Program_Create(OpenNet::Kernel * aKernel);
    #endif

    Thread_Functions * Thread_Get    ();
    Thread           * Thread_Prepare();
    void               Thread_Release();

    // ===== OpenNet::Processor =============================================
    virtual OpenNet::Status GetConfig(      Config * aOut   ) const;
    virtual void          * GetContext();
    virtual void          * GetDeviceId();
    virtual OpenNet::Status GetInfo  (      Info   * aOut   ) const;
    virtual const char    * GetName  () const;
    virtual OpenNet::Status SetConfig(const Config & aConfig);
    virtual OpenNet::Status Display  (      FILE   * aOut   ) const;

    // ===== OpenNet::StatisticsProvider ====================================
    virtual OpenNet::Status GetStatistics  (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
    virtual OpenNet::Status ResetStatistics();

private:

    void InitInfo();

    #ifdef _KMS_WINDOWS_
    
        // ===== OpenCL =========================================================

        bool GetDeviceInfo(cl_device_info aParam);
        void GetDeviceInfo(cl_device_info aParam, size_t aOutSize_byte, void * aOut);

        void GetKernelWorkGroupInfo(cl_kernel aKernel, cl_kernel_work_group_info aParam, size_t aOutSize_byte, void * aOut);
    
    #endif

    Config             mConfig  ;
    KmsLib::DebugLog * mDebugLog;
    Info               mInfo    ;
    Thread_Functions * mThread  ;
    
    #ifdef _KMS_WINDOWS_
        cl_context   mContext;
        cl_device_id mDevice ;
    #endif

};
