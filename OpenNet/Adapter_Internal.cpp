
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Processor_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenCL =============================================================
#include <CL/opencl.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== Includes ===========================================================
#include <OpenNet/Processor.h>
#include <OpenNet/Status.h>

// ===== OpenNet ============================================================
#include "Processor_Internal.h"

#include "Adapter_Internal.h"

// Public
/////////////////////////////////////////////////////////////////////////////

Adapter_Internal::Adapter_Internal(KmsLib::Windows::DriverHandle * aHandle) : mHandle(aHandle)
{
    assert(NULL != aHandle);

    memset(&mConfig, 0, sizeof(mConfig));
    memset(&mInfo  , 0, sizeof(mInfo  ));

    mHandle->Control(OPEN_NET_IOCTL_CONFIG_GET, NULL, 0, &mConfig, sizeof(mConfig));
    mHandle->Control(OPEN_NET_IOCTL_INFO_GET  , NULL, 0, &mInfo  , sizeof(mInfo  ));
}

// ===== OpenNet::Adapter ===================================================

OpenNet::Status Adapter_Internal::GetAdapterNo(unsigned int * aOut)
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    OpenNet::Adapter::State lState;

    OpenNet::Status lResult = GetState(&lState);
    if (OpenNet::STATUS_OK == lResult)
    {
        if (OPEN_NET_ADAPTER_NO_UNKNOWN == lState.mAdapterNo)
        {
            lResult = OpenNet::STATUS_NOT_CONNECTED;
        }
        else
        {
            if (OPEN_NET_ADAPTER_NO_QTY <= lState.mAdapterNo)
            {
                lResult = OpenNet::STATUS_CORRUPTED_DRIVER_DATA;
            }
            else
            {
                (*aOut) = lState.mAdapterNo;
            }
        }
    }

    return lResult;
}

OpenNet::Status Adapter_Internal::GetConfig(Config * aOut) const
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mConfig, sizeof(mConfig));

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::GetInfo(Info * aOut) const
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mInfo, sizeof(mInfo));

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::GetState(State * aOut)
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    assert(NULL != mHandle);

    try
    {
        mHandle->Control(OPEN_NET_IOCTL_STATE_GET, NULL, 0, aOut, sizeof(State));
    }
    catch (KmsLib::Exception * eE)
    {
        switch (eE->GetCode())
        {
        case KmsLib::Exception::CODE_IOCTL_ERROR: return OpenNet::STATUS_IOCTL_ERROR;

        default: return OpenNet::STATUS_EXCEPTION;
        }
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::GetStats(Stats * aOut)
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    assert(NULL != mHandle);

    try
    {
        mHandle->Control(OPEN_NET_IOCTL_STATS_GET, NULL, 0, aOut, sizeof(Stats));
    }
    catch (KmsLib::Exception * eE)
    {
        switch (eE->GetCode())
        {
        case KmsLib::Exception::CODE_IOCTL_ERROR: return OpenNet::STATUS_IOCTL_ERROR;

        default: return OpenNet::STATUS_EXCEPTION;
        }
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::ResetInputFilter()
{
    if (NULL == mFilter)
    {
        return OpenNet::STATUS_FILTER_NOT_SET;
    }

    mFilter = NULL;

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::ResetProcessor()
{
    if (NULL == mProcessor)
    {
        return OpenNet::STATUS_PROCESSOR_NOT_SET;
    }

    mProcessor = NULL;

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::ResetStats()
{
    assert(NULL != mHandle);

    try
    {
        mHandle->Control(OPEN_NET_IOCTL_STATS_RESET, NULL, 0, NULL, 0);
    }
    catch (KmsLib::Exception * eE)
    {
        switch (eE->GetCode())
        {
        case KmsLib::Exception::CODE_IOCTL_ERROR: return OpenNet::STATUS_IOCTL_ERROR;

        default: return OpenNet::STATUS_EXCEPTION;
        }
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::SetConfig(const Config & aConfig)
{
    if (NULL == (&aConfig))
    {
        return OpenNet::STATUS_INVALID_REFERENCE;
    }

    memcpy(&mConfig, &aConfig, sizeof(mConfig));

    try
    {
        mHandle->Control(OPEN_NET_IOCTL_CONFIG_SET, &mConfig, sizeof(mConfig), &mConfig, sizeof(mConfig));
    }
    catch (KmsLib::Exception * eE)
    {
        switch (eE->GetCode())
        {
        case KmsLib::Exception::CODE_IOCTL_ERROR: return OpenNet::STATUS_IOCTL_ERROR;

        default: return OpenNet::STATUS_EXCEPTION;
        }
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::SetInputFilter(OpenNet::Filter * aFilter)
{
    if (NULL == aFilter)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (NULL != mFilter)
    {
        return OpenNet::STATUS_FILTER_ALREADY_SET;
    }

    mFilter = aFilter;

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::SetProcessor(OpenNet::Processor * aProcessor)
{
    if (NULL == aProcessor)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (NULL != mProcessor)
    {
        return OpenNet::STATUS_PROCESSOR_ALREADY_SET;
    }

    mProcessor = dynamic_cast<Processor_Internal *>(aProcessor);

    if (NULL == mProcessor)
    {
        return OpenNet::STATUS_INVALID_PROCESSOR;
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::Buffer_Allocate(unsigned int aCount)
{
    if (0 >= aCount)
    {
        return OpenNet::STATUS_INVALID_BUFFER_COUNT;
    }

    // TODO Dev

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::Buffer_Release(unsigned int aCount)
{
    if (0 >= aCount)
    {
        return OpenNet::STATUS_INVALID_BUFFER_COUNT;
    }

    // TODO Dev

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::Display(FILE * aOut) const
{
    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "Config\n");

    OpenNet::Adapter::Display(mConfig, aOut);

    fprintf(aOut, "Info\n");

    OpenNet::Adapter::Display(mInfo, aOut);

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::Packet_Send(void * aData, unsigned int aSize_byte)
{
    assert(NULL != mHandle);

    if (NULL == aData)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 >= aSize_byte)
    {
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    if (mInfo.mPacketSize_byte < aSize_byte)
    {
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    try
    {
        mHandle->Control(OPEN_NET_IOCTL_PACKET_SEND, aData, aSize_byte, NULL, 0);
    }
    catch (KmsLib::Exception * eE)
    {
        switch (eE->GetCode())
        {
        case KmsLib::Exception::CODE_IOCTL_ERROR: return OpenNet::STATUS_IOCTL_ERROR;

        default: return OpenNet::STATUS_EXCEPTION;
        }
    }

    return OpenNet::STATUS_OK;
}
