
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/System_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>
#include <string.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>
#include <KmsLib/Windows/DriverHandle.h>

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"
#include "Processor_Internal.h"

#include "System_Internal.h"

// Public
/////////////////////////////////////////////////////////////////////////////

System_Internal::System_Internal() : mEnqueueMakeBufferResident(NULL), mEnqueueWaitSignal(NULL), mPlatform(0)
{
    FindAdapters  ();
    FindPlatform  ();
    FindExtension ();
    FindProcessors();
}

System_Internal::~System_Internal()
{
    unsigned int i;
    
    for (i = 0; i < mAdapters.size(); i++)
    {
        delete mAdapters[i];
    }

    for (i = 0; i < mProcessors.size(); i++)
    {
        delete mProcessors[i];
    }
}

// ===== OpenNet::System ====================================================

OpenNet::Adapter * System_Internal::Adapter_Get(unsigned int aIndex)
{
    if (mAdapters.size() <= aIndex)
    {
        return NULL;
    }

    return mAdapters[aIndex];
}

unsigned int System_Internal::Adapter_GetCount() const
{
    return static_cast<unsigned int>(mAdapters.size());
}

OpenNet::Processor * System_Internal::Processor_Get(unsigned int aIndex)
{
    if (mProcessors.size() <= aIndex)
    {
        return NULL;
    }

    return mProcessors[aIndex];
}

unsigned int System_Internal::Processor_GetCount() const
{
    return static_cast<unsigned int>(mProcessors.size());
}

// Private
/////////////////////////////////////////////////////////////////////////////

void System_Internal::FindAdapters()
{
    for (unsigned int lIndex = 0;; lIndex++)
    {
        KmsLib::Windows::DriverHandle * lHandle = new KmsLib::Windows::DriverHandle();
        assert(NULL != lHandle);

        try
        {
            lHandle->Connect(OPEN_NET_DRIVER_INTERFACE, lIndex, GENERIC_ALL, 0);
        }
        catch (KmsLib::Exception * eE)
        {
            assert(KmsLib::Exception::CODE_NO_SUCH_DEVICE == eE->GetCode());

            (void)(eE);

            delete lHandle;
            break;
        }

        Adapter_Internal * lAdapter = new Adapter_Internal(lHandle);
        assert(NULL != lAdapter);

        mAdapters.push_back(lAdapter);
    }
}

void System_Internal::FindExtension()
{
    assert(NULL == mEnqueueMakeBufferResident);
    assert(NULL == mEnqueueWaitSignal        );

    if (0 != mPlatform)
    {
        mEnqueueMakeBufferResident = reinterpret_cast<clEnqueueMakeBuffersResidentAMD_fn>(clGetExtensionFunctionAddressForPlatform(mPlatform, "clEnqueueMakeBuffersResidentAMD"));
        mEnqueueWaitSignal         = reinterpret_cast<clEnqueueWaitSignalAMD_fn         >(clGetExtensionFunctionAddressForPlatform(mPlatform, "clEnqueueWaitSignalAMD"         ));

        assert(NULL != mEnqueueMakeBufferResident);
        assert(NULL != mEnqueueWaitSignal        );
    }
}

void System_Internal::FindPlatform()
{
    assert(0 == mPlatform);

    cl_uint lCount;

    cl_int lStatus = clGetPlatformIDs(0, NULL, &lCount);
    if ((CL_SUCCESS == lStatus) && (0 < lCount))
    {
        cl_platform_id * lPlatforms = new cl_platform_id[lCount];
        assert(NULL != lPlatforms);

        lStatus = clGetPlatformIDs(lCount, lPlatforms, NULL);
        if (CL_SUCCESS == lStatus)
        {
            for (unsigned int i = 0; i < lCount; i++)
            {
                assert(0 != lPlatforms[i]);

                char lBuffer[128];

                lStatus = clGetPlatformInfo(lPlatforms[i], CL_PLATFORM_VENDOR, sizeof(lBuffer), lBuffer, NULL);
                if (CL_SUCCESS == lStatus)
                {
                    if (0 == strcmp("Advanced Micro Devices, Inc.", lBuffer))
                    {
                        mPlatform = lPlatforms[i];
                        break;
                    }
                }
            }
        }

        delete lPlatforms;
    }
}

void System_Internal::FindProcessors()
{
    if (0 != mPlatform)
    {
        cl_uint lCount;

        cl_int lStatus = clGetDeviceIDs(mPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &lCount);
        if ((CL_SUCCESS == lStatus) && (0 < lCount))
        {
            cl_device_id * lDevices = new cl_device_id[lCount];
            assert(NULL != lDevices);

            lStatus = clGetDeviceIDs(mPlatform, CL_DEVICE_TYPE_GPU, lCount, lDevices, NULL);
            if (CL_SUCCESS == lStatus)
            {
                for (unsigned int i = 0; i < lCount; i++)
                {
                    if (IsExtensionSupported(lDevices[i]))
                    {
                        mProcessors.push_back( new Processor_Internal(mPlatform, lDevices[i]) );
                    }
                }
            }

            delete lDevices;
        }
    }
}

bool System_Internal::IsExtensionSupported(cl_device_id aDevice)
{
    assert(0 != aDevice);

    size_t lSize_byte;

    cl_int lStatus = clGetDeviceInfo(aDevice, CL_DEVICE_EXTENSIONS, 0, NULL, &lSize_byte);
    if (CL_SUCCESS == lStatus)
    {
        char * lExtNames = new char[lSize_byte];
        assert(NULL != lExtNames);

        lStatus = clGetDeviceInfo(aDevice, CL_DEVICE_EXTENSIONS, lSize_byte, lExtNames, NULL);
        if (CL_SUCCESS == lStatus)
        {
            if (NULL != strstr(lExtNames, "cl_amd_bus_addressable_memory"))
            {
                return true;
            }
        }

        delete lExtNames;
    }

    return false;
}
