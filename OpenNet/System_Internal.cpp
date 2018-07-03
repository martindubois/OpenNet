
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

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>
#include <KmsLib/Windows/DriverHandle.h>

// ===== Common =============================================================
#include "../Common/Constants.h"

// ===== OpenNet ============================================================
#include "Adapter_Internal.h"
#include "OCLW.h"
#include "Processor_Internal.h"

#include "System_Internal.h"

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static OpenNet::Status ExceptionToStatus(const KmsLib::Exception * aE);

static void SendLoopBackPackets(void * aThis, Adapter_Internal * aAdapter);

// Public
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  See FindExtension
//                                 See FindPlatform
//                                 See FindProcessors
// Threads  Apps
System_Internal::System_Internal()
    : mDebugLog( "K:\\Dossiers_Actifs\\OpenNet\\DebugLog", "OpenNet" )
    , mAdapterRunning (0)
    , mPacketSize_byte(PACKET_SIZE_MAX_byte)
    , mPlatform       (0)
{
    memset(&mConnect           , 0, sizeof(mConnect           ));
    memset(&mExtensionFunctions, 0, sizeof(mExtensionFunctions));

    FindAdapters  ();
    FindPlatform  ();
    FindExtension ();
    FindProcessors();

    // CreateEvent ==> CloseHandle  See the destructor
    mConnect.mEvent = reinterpret_cast<uint64_t>(CreateEvent(NULL, FALSE, TRUE, NULL));
    assert(NULL != mConnect.mEvent);

    // VirtualAlloc ==> VirtualAlloc  See the destructor
    mConnect.mSharedMemory = VirtualAlloc(NULL, SHARED_MEMORY_SIZE_byte, MEM_COMMIT, PAGE_READWRITE);
    assert(NULL != mConnect.mSharedMemory);

    mConnect.mSystemId = GetCurrentProcessId();
    assert(0 != mConnect.mSystemId);
}

// Threads  Apps
System_Internal::~System_Internal()
{
    assert(   0 != mConnect.mEvent       );
    assert(NULL != mConnect.mSharedMemory);

    unsigned int i;
    
    for (i = 0; i < mAdapters.size(); i++)
    {
        // new ==> delete
        delete mAdapters[i];
    }

    for (i = 0; i < mProcessors.size(); i++)
    {
        // new ==> delete
        delete mProcessors[i];
    }

    // CreateEvent ==> CloseHandle  See the default contructor
    BOOL lRetB = CloseHandle(reinterpret_cast<HANDLE>(mConnect.mEvent));
    assert(lRetB);

    // VirtualAlloc ==> VirtualAlloc  See the default constructor
    void * lRetVP = VirtualAlloc(mConnect.mSharedMemory, SHARED_MEMORY_SIZE_byte, MEM_RESET, 0);
    assert(NULL == lRetVP);
}

// ===== OpenNet::System ====================================================

unsigned int System_Internal::GetSystemId() const
{
    assert(0 != mConnect.mSystemId);

    return mConnect.mSystemId;
}

OpenNet::Status System_Internal::SetPacketSize(unsigned int aSize_byte)
{
    assert(PACKET_SIZE_MAX_byte >= mPacketSize_byte);
    assert(PACKET_SIZE_MIN_byte <= mPacketSize_byte);

    if (PACKET_SIZE_MAX_byte < aSize_byte)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    if (PACKET_SIZE_MIN_byte > aSize_byte)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    mPacketSize_byte = aSize_byte;

    OpenNet::Status lResult = OpenNet::STATUS_OK;

    for (unsigned int i = 0; i < mAdapters.size(); i++)
    {
        Adapter_Internal * lAdapter = mAdapters[i];
        assert(NULL != lAdapter);

        if (lAdapter->IsConnected(*this))
        {
            lResult = lAdapter->SetPacketSize(mPacketSize_byte);
            if (OpenNet::STATUS_OK != lResult)
            {
                mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
                break;
            }
        }
    }

    return lResult;
}

OpenNet::Status System_Internal::Adapter_Connect(OpenNet::Adapter * aAdapter)
{
    OpenNet::Status lResult = ValidateAdapter(aAdapter);
    if (OpenNet::STATUS_OK != lResult)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return lResult;
    }

    if (aAdapter->IsConnected())
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_ADAPTER_ALREADY_CONNECTED;
    }

    try
    {
        Adapter_Internal * lAdapter = dynamic_cast<Adapter_Internal *>(aAdapter);
        assert(NULL != lAdapter);

        unsigned int lPacketSize_byte = lAdapter->GetPacketSize();

        if (lPacketSize_byte < mPacketSize_byte)
        {
            lResult = SetPacketSize(lPacketSize_byte);
        }
        else if (lPacketSize_byte > mPacketSize_byte)
        {
            lResult = lAdapter->SetPacketSize(mPacketSize_byte);
        }

        if (OpenNet::STATUS_OK == lResult)
        {
            lAdapter->Connect(&mConnect);
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog.Log(eE);

        lResult = ExceptionToStatus(eE);
    }

    return lResult;
}

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

OpenNet::Status System_Internal::Display(FILE * aOut)
{
    assert(mAdapters.size() >= mAdapterRunning   );
    assert(               0 != mConnect.mSystemId);

    if (NULL == aOut)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "System :\n");
    fprintf(aOut, "  %u Adapter Running\n"    , mAdapterRunning    );
    fprintf(aOut, "  %zu Adapters\n"          , mAdapters  .size() );
    fprintf(aOut, "  %zu Processors\n"        , mProcessors.size() );
    fprintf(aOut, "  System Id   = %u\n"      , mConnect.mSystemId );
    fprintf(aOut, "  Packet Size = %u bytes\n", mPacketSize_byte   );

    return OpenNet::STATUS_OK;
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

OpenNet::Status System_Internal::Start()
{
    if (0 < mAdapterRunning)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_SYSTEM_ALREADY_STARTED;
    }

    OpenNet::Status lResult = OpenNet::STATUS_NO_ADAPTER_CONNECTED;

    unsigned int i;

    try
    {
        for (i = 0; i < mAdapters.size(); i++)
        {
            Adapter_Internal * lAdapter = mAdapters[i];
            assert(NULL != lAdapter);

            if (lAdapter->IsConnected(*this))
            {
                lAdapter->Start();
                mAdapterRunning++;
                lResult = OpenNet::STATUS_OK;
            }
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog.Log(eE);

        lResult = ExceptionToStatus(eE);

        OpenNet::Status lStatus = Stop(0);
        assert(OpenNet::STATUS_OK == lStatus);
    }

    return lResult;
}

OpenNet::Status System_Internal::Stop(unsigned int aFlags)
{
    if (0 >= mAdapterRunning)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_SYSTEM_NOT_STARTED;
    }

    OpenNet::Status lResult = OpenNet::STATUS_NO_ADAPTER_CONNECTED;

    try
    {
        for (unsigned int i = 0; (i < mAdapters.size()) && (0 < mAdapterRunning); i++)
        {
            Adapter_Internal * lAdapter = mAdapters[i];
            assert(NULL != lAdapter);

            if (lAdapter->IsConnected(*this))
            {
                lAdapter->Stop_Request();
            }
        }

        for (unsigned int i = 0; (i < mAdapters.size()) && (0 < mAdapterRunning); i++)
        {
            Adapter_Internal * lAdapter = mAdapters[i];
            assert(NULL != lAdapter);

            if (lAdapter->IsConnected(*this))
            {
                if (0 != (aFlags & STOP_FLAG_LOOPBACK))
                {
                    lAdapter->Stop_Wait(::SendLoopBackPackets, this);
                }
                else
                {
                    lAdapter->Stop_Wait(NULL, NULL);
                }

                mAdapterRunning--;
                lResult = OpenNet::STATUS_OK;
            }
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog.Log(eE);

        lResult = ExceptionToStatus(eE);
    }

    return lResult;
}

// ===== OpenNet::StatisticsProvider ========================================

OpenNet::Status System_Internal::GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset)
{
    // TODO  OpenNet.System_Internal

    return OpenNet::STATUS_OK;
}

OpenNet::Status System_Internal::ResetStatistics()
{
    // TODO  OpenNet.System_Internal

    return OpenNet::STATUS_OK;
}

// Internal
/////////////////////////////////////////////////////////////////////////////

void System_Internal::SendLoopBackPackets(Adapter_Internal * aAdapter)
{
    assert(NULL != aAdapter);

    for (unsigned int i = 0; i < mAdapters.size(); i++)
    {
        Adapter_Internal * lAdapter = mAdapters[i];
        assert(NULL != lAdapter);

        if (aAdapter != lAdapter)
        {
            lAdapter->SendLoopBackPackets();
        }
    }
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Threads  Apps
void System_Internal::FindAdapters()
{
    for (unsigned int lIndex = 0;; lIndex++)
    {
        // new ==> delete  See Adapter_Internal::~Adapter_Internal
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

        // new ==> delete  See ~System_Internal
        Adapter_Internal * lAdapter = new Adapter_Internal(lHandle, &mDebugLog);
        assert(NULL != lAdapter);

        mAdapters.push_back(lAdapter);
    }
}

// Exception  KmsLib::Exception *  See OCLW_GetExtensionFunctionAddressForPlatform
// Threads  Apps
void System_Internal::FindExtension()
{
    assert(NULL == mExtensionFunctions.mEnqueueMakeBufferResident);
    assert(NULL == mExtensionFunctions.mEnqueueWaitSignal        );

    if (0 != mPlatform)
    {
        mExtensionFunctions.mEnqueueMakeBufferResident = reinterpret_cast<clEnqueueMakeBuffersResidentAMD_fn>(OCLW_GetExtensionFunctionAddressForPlatform(mPlatform, "clEnqueueMakeBuffersResidentAMD"));
        mExtensionFunctions.mEnqueueWaitSignal         = reinterpret_cast<clEnqueueWaitSignalAMD_fn         >(OCLW_GetExtensionFunctionAddressForPlatform(mPlatform, "clEnqueueWaitSignalAMD"         ));

        assert(NULL != mExtensionFunctions.mEnqueueMakeBufferResident);
        assert(NULL != mExtensionFunctions.mEnqueueWaitSignal        );
    }
}

// Exception  KmsLib::Exception *  See OCLW_GetPlatformIDs
// Threads  Apps
void System_Internal::FindPlatform()
{
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
void System_Internal::FindProcessors()
{
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
                    mProcessors.push_back( new Processor_Internal(mPlatform, lDevices[i], &mExtensionFunctions, &mDebugLog) );
                }
            }
        }
    }
}

// Threads  Apps
bool System_Internal::IsExtensionSupported(cl_device_id aDevice)
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

// aAdapter [---;---]
//
// Threads  Apps
OpenNet::Status System_Internal::ValidateAdapter(OpenNet::Adapter * aAdapter)
{
    if (NULL == aAdapter)
    {
        mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    for (unsigned int i = 0; i < mAdapters.size(); i++)
    {
        if (mAdapters[i] == aAdapter)
        {
            return OpenNet::STATUS_OK;
        }
    }

    mDebugLog.Log(__FILE__, __FUNCTION__, __LINE__);
    return OpenNet::STATUS_INVALID_ADAPTER;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aE [---;R--]
//
// Threads  Apps
OpenNet::Status ExceptionToStatus(const KmsLib::Exception * aE)
{
    assert(NULL != aE);

    switch (aE->GetCode())
    {
    case KmsLib::Exception::CODE_IOCTL_ERROR: return OpenNet::STATUS_IOCTL_ERROR;
    }

    return OpenNet::STATUS_EXCEPTION;
}

void SendLoopBackPackets(void * aThis, Adapter_Internal * aAdapter)
{
    assert(NULL != aThis   );
    assert(NULL != aAdapter);

    System_Internal * lThis = reinterpret_cast<System_Internal *>(aThis);

    lThis->SendLoopBackPackets(aAdapter);
}