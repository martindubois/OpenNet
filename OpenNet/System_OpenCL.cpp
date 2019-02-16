
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/System_OpenCL.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Common =============================================================
#include "../Common/Constants.h"

// ===== OpenNet ============================================================
#include "Adapter_Windows.h"
#include "OCLW.h"
#include "Processor_OpenCL.h"

#include "System_OpenCL.h"

// Public
/////////////////////////////////////////////////////////////////////////////

System_OpenCL::System_OpenCL()
    : mPlatform(0)
{
    mInfo.mSystemId = GetCurrentProcessId();
    assert(0 != mInfo.mSystemId);

    mConnect.mSystemId = mInfo.mSystemId;

    FindAdapters  ();
    FindPlatform  ();
    FindProcessors();

    if (0 != mPlatform)
    {
        OCLW_Initialise(mPlatform);
    }

    // CreateEvent ==> CloseHandle  See the destructor
    mConnect.mEvent = reinterpret_cast<uint64_t>(CreateEvent(NULL, FALSE, TRUE, NULL));
    assert(NULL != mConnect.mEvent);

    // VirtualAlloc ==> VirtualAlloc  See the destructor
    mConnect.mSharedMemory = VirtualAlloc(NULL, SHARED_MEMORY_SIZE_byte, MEM_COMMIT, PAGE_READWRITE);
    assert(NULL != mConnect.mSharedMemory);

    memset( mConnect.mSharedMemory, 0, SHARED_MEMORY_SIZE_byte );
}

System_OpenCL::~System_OpenCL()
{
    // CreateEvent ==> CloseHandle  See the default contructor
    BOOL lRetB = CloseHandle(reinterpret_cast<HANDLE>(mConnect.mEvent));
    assert(lRetB);
    (void)(lRetB);

    // VirtualAlloc ==> VirtualAlloc  See the default constructor
    void * lRetVP = VirtualAlloc(mConnect.mSharedMemory, SHARED_MEMORY_SIZE_byte, MEM_RESET, 0);
    assert(NULL == lRetVP);
    (void)(lRetVP);
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Threads  Apps
void System_OpenCL::FindAdapters()
{
    mDebugLog.Log( "System_OpenCL::FindAdapters()" );

    for (unsigned int lIndex = 0;; lIndex++)
    {
        // new ==> delete  See Adapter_Internal::~Adapter_Internal
        KmsLib::DriverHandle * lHandle = new KmsLib::DriverHandle();
        assert(NULL != lHandle);

        try
        {
            lHandle->Connect(OPEN_NET_DRIVER_INTERFACE, lIndex, GENERIC_ALL, 0);
        }
        catch (KmsLib::Exception * eE)
        {
            (void)(eE);

            delete lHandle;
            break;
        }

        // new ==> delete  See ~System_Internal
        Adapter_Internal * lAdapter = new Adapter_Windows(lHandle, &mDebugLog);
        assert(NULL != lAdapter);

        mAdapters.push_back(lAdapter);
    }
}

// Exception  KmsLib::Exception *  See OCLW_GetPlatformIDs
// Threads  Apps
void System_OpenCL::FindPlatform()
{
    mDebugLog.Log( "System_OpenCL::FindPlatform()" );

    assert(0 == mPlatform);

    cl_platform_id lPlatforms[32];

    cl_uint lCount;

    OCLW_GetPlatformIDs(sizeof(lPlatforms) / sizeof(lPlatforms[0]), lPlatforms, &lCount);

    for (unsigned int i = 0; i < lCount; i++)
    {
        assert(0 != lPlatforms[i]);

        char lBuffer[128];

        OCLW_GetPlatformInfo(lPlatforms[i], CL_PLATFORM_VENDOR, sizeof(lBuffer), lBuffer);

        if (0 == strcmp("Advanced Micro Devices, Inc.", lBuffer))
        {
            mPlatform = lPlatforms[i];
            break;
        }
    }
}

// Exception  KmsLib::Exception *  See OCLW_GetDeviceIDs
// Threads  Apps
void System_OpenCL::FindProcessors()
{
    mDebugLog.Log( "System_OpenCL::FindProcessors()" );

    if (0 != mPlatform)
    {
        cl_device_id lDevices[255];

        cl_uint lCount;

        OCLW_GetDeviceIDs(mPlatform, CL_DEVICE_TYPE_GPU, sizeof(lDevices) / sizeof(lDevices[0]), lDevices, &lCount);
        if (0 < lCount)
        {
            for (unsigned int i = 0; i < lCount; i++)
            {
                if (IsExtensionSupported(lDevices[i]))
                {
                    // new ==> Delete  See ~System_Internal
                    mProcessors.push_back( new Processor_OpenCL(mPlatform, lDevices[i], &mDebugLog) );
                }
            }
        }
    }
}

// Threads  Apps
bool System_OpenCL::IsExtensionSupported(cl_device_id aDevice)
{
    assert(0 != aDevice);

    char   lExtNames[8192];
    size_t lSize_byte;

    cl_int lStatus = clGetDeviceInfo(aDevice, CL_DEVICE_EXTENSIONS, sizeof(lExtNames), lExtNames, &lSize_byte);
    if (CL_SUCCESS == lStatus)
    {
        if (NULL != strstr(lExtNames, "cl_amd_bus_addressable_memory"))
        {
            return true;
        }
    }

    return false;
}
