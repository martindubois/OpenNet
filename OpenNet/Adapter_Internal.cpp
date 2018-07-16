
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Processor_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== Includes ===========================================================
#include <OpenNet/Function.h>
#include <OpenNet/Status.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/IoCtl.h"
#include "../Common/OpenNet/Adapter_Statistics.h"

// ===== OpenNet ============================================================
#include "EthernetAddress.h"
#include "OCLW.h"
#include "Thread_Functions.h"
#include "Thread_Kernel.h"

#include "Adapter_Internal.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static const uint8_t LOOP_BACK_PACKET[64] =
{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88
};

// Static functions declaration
/////////////////////////////////////////////////////////////////////////////

static OpenNet::Status ExceptionToStatus(const KmsLib::Exception * aE);

// Public
/////////////////////////////////////////////////////////////////////////////

// aHandle   [DK-;RW-]
// aDebugLog [-K-;RW-]
//
// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
Adapter_Internal::Adapter_Internal(KmsLib::Windows::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog)
    : mBufferCount(0)
    , mDebugLog   (aDebugLog)
    , mHandle     (aHandle)
    , mProcessor  (NULL)
    , mProgram    (NULL)
    , mSourceCode (NULL)
{
    assert(NULL != aHandle  );
    assert(NULL != aDebugLog);

    memset(&mConfig      , 0, sizeof(mConfig      ));
    memset(&mDriverConfig, 0, sizeof(mDriverConfig));
    memset(&mInfo        , 0, sizeof(mInfo        ));
    memset(&mName        , 0, sizeof(mName        ));
    memset(&mStatistics  , 0, sizeof(mStatistics  ));

    mConfig.mBufferQty = 4;

    mHandle->Control(IOCTL_CONFIG_GET, NULL, 0, &mDriverConfig, sizeof(mDriverConfig));
    mHandle->Control(IOCTL_INFO_GET  , NULL, 0, &mInfo        , sizeof(mInfo        ));

    Config_Update();

    strncpy_s(mName, mInfo.mVersion_Hardware.mComment, sizeof(mName) - 1);

    unsigned int lOffset_byte = static_cast<unsigned int>(strlen(mName));

    if (44 < lOffset_byte)
    {
        lOffset_byte = 44;
    }

    OpenNet::EthernetAddress_GetText(mInfo.mEthernetAddress, mName + strlen(mName), sizeof(mName) - lOffset_byte);
}

// Threads  Apps
Adapter_Internal::~Adapter_Internal()
{
    assert(NULL != mHandle);

    try
    {
        if (NULL != mProgram)
        {
            OCLW_ReleaseProgram(mProgram);
        }
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
    }

    delete mHandle;
}

unsigned int Adapter_Internal::GetBufferQty() const
{
    assert(0                   <  mConfig.mBufferQty);
    assert(OPEN_NET_BUFFER_QTY >= mConfig.mBufferQty);

    return mConfig.mBufferQty;
}

// Thread  Apps
unsigned int Adapter_Internal::GetPacketSize() const
{
    assert(PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);
    assert(PACKET_SIZE_MIN_byte <= mConfig.mPacketSize_byte);

    return mConfig.mPacketSize_byte;
}

// Exception  KmsLib::Exception *  CODE_INVALID_ARGUMENT
//                                 See KmsLib::Windows::DriverHandle::Control
// Thread     Apps
void Adapter_Internal::SetPacketSize(unsigned int aSize_byte)
{
    assert(PACKET_SIZE_MAX_byte >= aSize_byte);
    assert(PACKET_SIZE_MIN_byte <= aSize_byte);

    assert(PACKET_SIZE_MAX_byte     >= mConfig      .mPacketSize_byte);
    assert(PACKET_SIZE_MIN_byte     <= mConfig      .mPacketSize_byte);
    assert(mConfig.mPacketSize_byte == mDriverConfig.mPacketSize_byte);
    assert(NULL                     != mHandle                       );

    assert(NULL != mDebugLog);

    if (mConfig.mPacketSize_byte != aSize_byte)
    {
        mDriverConfig.mPacketSize_byte = aSize_byte;

        mHandle->Control(IOCTL_CONFIG_SET, &mDriverConfig, sizeof(mDriverConfig), &mDriverConfig, sizeof(mDriverConfig));

        assert(PACKET_SIZE_MAX_byte >= mDriverConfig.mPacketSize_byte);
        assert(PACKET_SIZE_MIN_byte <= mDriverConfig.mPacketSize_byte);

        Config_Update();

        assert(mDriverConfig.mPacketSize_byte == mConfig.mPacketSize_byte);

        if (mConfig.mPacketSize_byte != aSize_byte)
        {
            mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
            throw new KmsLib::Exception(KmsLib::Exception::CODE_INVALID_ARGUMENT,
                "Invalid packet size", NULL, __FILE__, __FUNCTION__, __LINE__, mDriverConfig.mPacketSize_byte);
        }
    }
}

// aCommandQueue [---;RW-]
// aKernel       [---;R--]
// aBuffers      [---;RW-] The caller is responsible to release the
//                         Buffer_Data instances added to this queue when
//                         they are no longer needed.
//
// Exception  KmsLib::Exception *  See Adapter_Internal::Buffer_Allocate
// Thread     Apps
void Adapter_Internal::Buffers_Allocate(cl_command_queue aCommandQueue, cl_kernel aKernel, Buffer_Data_Vector * aBuffers)
{
    assert(NULL != aCommandQueue);
    assert(NULL != aKernel      );
    assert(NULL != aBuffers     );

    assert(0 < mConfig.mBufferQty);

    for (unsigned int i = 0; i < mConfig.mBufferQty; i++)
    {
        aBuffers->push_back(Buffer_Allocate(aCommandQueue, aKernel));
    }
}

// Exception  KmsLib::Exception *  See Adapter_Internal::Buffer_Release
void Adapter_Internal::Buffers_Release()
{
    assert(0 < mBufferCount);

    mBufferCount = 0;
}

// aConnect [---;R--]
//
// Exception  KmsLib::Exception *  CODE_TIMEOUT
//                                 See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
void Adapter_Internal::Connect(IoCtl_Connect_In * aConnect)
{
    assert(NULL != aConnect);

    assert(NULL != mDebugLog);
    assert(NULL != mHandle  );

    HANDLE lEvent = reinterpret_cast<HANDLE>(aConnect->mEvent);
    assert(NULL != lEvent);

    DWORD lRet = WaitForSingleObject(lEvent, 60000);
    if (WAIT_OBJECT_0 != lRet)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_TIMEOUT,
            "WaitForSingleObject( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet);
    }

    try
    {
        mHandle->Control(IOCTL_CONNECT, aConnect, sizeof(IoCtl_Connect_In), NULL, 0);

        SetEvent(lEvent);
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        SetEvent(lEvent);
        throw;
    }
}

void Adapter_Internal::Packet_Send_Ex(const IoCtl_Packet_Send_Ex_In * aIn, unsigned int aInSize_byte)
{
    assert(NULL                            != aIn         );
    assert(sizeof(IoCtl_Packet_Send_Ex_In) <= aInSize_byte);

    unsigned int lRet = mHandle->Control(IOCTL_PACKET_SEND_EX, aIn, aInSize_byte, NULL, 0);
    assert(0 == lRet);
}

// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
void Adapter_Internal::SendLoopBackPackets()
{
    assert(NULL != mHandle);

    for (unsigned i = 0; i < 64; i++)
    {
        mHandle->Control(IOCTL_PACKET_SEND, LOOP_BACK_PACKET, sizeof(LOOP_BACK_PACKET), NULL, 0);
        mStatistics[OpenNet::ADAPTER_STATS_LOOP_BACK_PACKET] ++;
    }
}

// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
void Adapter_Internal::Start()
{
    assert(   0 <  mBufferCount);
    assert(NULL != mHandle     );

    mHandle->Control(IOCTL_START, mBuffers, sizeof(OpenNetK::Buffer) * mBufferCount, NULL, 0);
}

// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
// Threads    Worker
void Adapter_Internal::Stop()
{
    assert(NULL != mHandle);

    mHandle->Control(IOCTL_STOP, NULL, 0, NULL, 0);
}

// Thread  Apps
Thread * Adapter_Internal::Thread_Prepare()
{
    assert(NULL != mDebugLog  );
    assert(NULL != mSourceCode);
    assert(NULL != mProcessor );
    assert(NULL != mProgram   );

    OpenNet::Kernel * lKernel = dynamic_cast<OpenNet::Kernel *>(mSourceCode);
    if (NULL != lKernel)
    {
        return new Thread_Kernel(mProcessor, this, lKernel, mProgram, mDebugLog);
    }

    OpenNet::Function * lFunction = dynamic_cast<OpenNet::Function *>(mSourceCode);
    assert(NULL != lFunction);

    Thread_Functions * lThread = mProcessor->Thread_Get();
    assert(NULL != lThread);

    lThread->AddAdapter(this, *lFunction);

    return NULL;
}

// ===== OpenNet::Adapter ===================================================

OpenNet::Status Adapter_Internal::GetAdapterNo(unsigned int * aOut)
{
    assert(NULL != mDebugLog);

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    OpenNet::Adapter::State lState;

    OpenNet::Status lResult = GetState(&lState);
    if (OpenNet::STATUS_OK == lResult)
    {
        if (ADAPTER_NO_UNKNOWN == lState.mAdapterNo)
        {
            mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
            lResult = OpenNet::STATUS_ADAPTER_NOT_CONNECTED;
        }
        else
        {
            if (ADAPTER_NO_QTY <= lState.mAdapterNo)
            {
                mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
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
    assert(NULL != mDebugLog);

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mConfig, sizeof(mConfig));

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::GetInfo(Info * aOut) const
{
    assert(NULL != mDebugLog);

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mInfo, sizeof(mInfo));

    return OpenNet::STATUS_OK;
}

const char * Adapter_Internal::GetName() const
{
    return mName;
}

OpenNet::Status Adapter_Internal::GetState(State * aOut)
{
    assert(NULL != mDebugLog);
    assert(NULL != mHandle  );

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    return Control(IOCTL_STATE_GET, NULL, 0, aOut, sizeof(State));
}

OpenNet::Status Adapter_Internal::GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset)
{
    assert(NULL != mDebugLog);
    assert(NULL != mHandle  );

    if ((NULL == aOut) && (0 < aOutSize_byte))
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    unsigned int * lOut          = aOut;
    unsigned int   lOutSize_byte = aOutSize_byte;
    unsigned int   lSize_byte    = 0;

    if (sizeof(mStatistics) < lOutSize_byte)
    {
        memcpy(lOut, &mStatistics, sizeof(mStatistics));
        lSize_byte    += sizeof(mStatistics);
        lOut          += sizeof(mStatistics) / sizeof(unsigned int);
        lOutSize_byte -= sizeof(mStatistics);
    }
    else
    {
        if (0 < lOutSize_byte)
        {
            memcpy(lOut, &mStatistics, lOutSize_byte);
            lSize_byte   += lOutSize_byte;
            lOut          = NULL;
            lOutSize_byte = 0;
        }
    }

    if (aReset)
    {
        memset(&mStatistics, 0, sizeof(unsigned int) * OpenNet::ADAPTER_STATS_RESET_QTY);
    }

    IoCtl_Stats_Get_In lIn;

    memset(&lIn, 0, sizeof(lIn));

    lIn.mFlags.mReset = aReset;

    unsigned int lInfo_byte;

    OpenNet::Status lResult = Control(IOCTL_STATISTICS_GET, &lIn, sizeof(lIn), lOut, lOutSize_byte, &lInfo_byte);
    if (OpenNet::STATUS_OK == lResult)
    {
        if (NULL != aInfo_byte)
        {
            (*aInfo_byte) = lSize_byte + lInfo_byte;
        }
    }

    return lResult;
}

bool Adapter_Internal::IsConnected()
{
    State lState;

    OpenNet::Status lStatus = GetState(&lState);
    assert(OpenNet::STATUS_OK == lStatus);

    assert((ADAPTER_NO_UNKNOWN == lState.mAdapterNo) || (ADAPTER_NO_QTY > lState.mAdapterNo));

    return (ADAPTER_NO_UNKNOWN != lState.mAdapterNo);
}

bool Adapter_Internal::IsConnected(const OpenNet::System & aSystem)
{
    if (NULL == (&aSystem))
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    State lState;

    OpenNet::Status lStatus = GetState(&lState);
    assert(OpenNet::STATUS_OK == lStatus);

    OpenNet::System::Info lInfo;

    lStatus = aSystem.GetInfo(&lInfo);
    assert(OpenNet::STATUS_OK == lStatus);

    assert((ADAPTER_NO_UNKNOWN == lState.mAdapterNo) || (ADAPTER_NO_QTY > lState.mAdapterNo));

    return ((ADAPTER_NO_UNKNOWN != lState.mAdapterNo) && (lInfo.mSystemId == lState.mSystemId));
}

OpenNet::Status Adapter_Internal::ResetInputFilter()
{
    assert(NULL != mDebugLog);

    if (NULL == mSourceCode)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_FILTER_NOT_SET;
    }

    if (0 < mBufferCount)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_ADAPTER_RUNNING;
    }

    assert(NULL != mProcessor);

    mSourceCode = NULL;

    OpenNet::Status lResult = OpenNet::STATUS_OK;

    if (NULL != mProgram)
    {
        try
        {
            OCLW_ReleaseProgram(mProgram);
        }
        catch (KmsLib::Exception * eE)
        {
            mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
            mDebugLog->Log(eE);

            lResult = ExceptionToStatus(eE);
        }

        mProgram = NULL;
    }

    return lResult;
}

OpenNet::Status Adapter_Internal::ResetProcessor()
{
    assert(NULL != mDebugLog);

    if (NULL == mProcessor)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PROCESSOR_NOT_SET;
    }

    if (NULL != mSourceCode)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_FILTER_SET;
    }

    mProcessor = NULL;

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::ResetStatistics()
{
    memset(&mStatistics, 0, sizeof(mStatistics));

    return Control(IOCTL_STATISTICS_RESET, NULL, 0, NULL, 0);
}

OpenNet::Status Adapter_Internal::SetConfig(const Config & aConfig)
{
    assert(mDriverConfig.mPacketSize_byte == mConfig.mPacketSize_byte);
    assert(NULL                           != mDebugLog               );

    if (NULL == (&aConfig))
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_REFERENCE;
    }

    if (PACKET_SIZE_MAX_byte < aConfig.mPacketSize_byte)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    if (PACKET_SIZE_MIN_byte > aConfig.mPacketSize_byte)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    if (OPEN_NET_BUFFER_QTY < aConfig.mBufferQty)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_TOO_MANY_BUFFER;
    }

    if (0 >= aConfig.mBufferQty)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NO_BUFFER;
    }

    memcpy(&mConfig, &aConfig, sizeof(mConfig));

    mDriverConfig.mPacketSize_byte = mConfig.mPacketSize_byte;

    OpenNet::Status lResult = Control(IOCTL_CONFIG_SET, &mDriverConfig, sizeof(mDriverConfig), &mDriverConfig, sizeof(mDriverConfig));
    if (OpenNet::STATUS_OK == lResult)
    {
        Config_Update();
    }

    return lResult;
}

OpenNet::Status Adapter_Internal::SetInputFilter(OpenNet::SourceCode * aSourceCode)
{
    assert(NULL != mDebugLog);

    if (NULL == aSourceCode)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (NULL != mSourceCode)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_FILTER_ALREADY_SET;
    }

    assert(NULL == mProgram);

    if (NULL == mProcessor)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PROCESSOR_NOT_SET;
    }

    try
    {
        OpenNet::Kernel * lKernel = dynamic_cast<OpenNet::Kernel *>(aSourceCode);

        if (NULL != lKernel)
        {
            mProgram = mProcessor->Program_Create(lKernel);
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog->Log(eE);
        return ExceptionToStatus(eE);
    }

    mSourceCode = aSourceCode;

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::SetProcessor(OpenNet::Processor * aProcessor)
{
    assert(NULL != mDebugLog);

    if (NULL == aProcessor)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (NULL != mProcessor)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PROCESSOR_ALREADY_SET;
    }

    mProcessor = dynamic_cast<Processor_Internal *>(aProcessor);

    if (NULL == mProcessor)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_PROCESSOR;
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::Display(FILE * aOut) const
{
    assert(OPEN_NET_BUFFER_QTY >= mBufferCount);
    assert(NULL                != mDebugLog   );

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "Adapter :\n");
    fprintf(aOut, "  %u Buffers\n"      , mBufferCount);
    fprintf(aOut, "  Name        = %s\n", mName);
    fprintf(aOut, "  Processor   = %s\n", ((NULL == mProcessor ) ? "Not set" : mProcessor ->GetName()));
    fprintf(aOut, "  Source Code = %s\n", ((NULL == mSourceCode) ? "Not set" : mSourceCode->GetName()));

    OpenNet::Adapter::Display(mConfig, aOut);
    OpenNet::Adapter::Display(mInfo  , aOut);

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::Packet_Send(const void * aData, unsigned int aSize_byte)
{
    assert(NULL != mDebugLog);

    if (NULL == aData)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 >= aSize_byte)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    if (mInfo.mPacketSize_byte < aSize_byte)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    mStatistics[OpenNet::ADAPTER_STATS_PACKET_SEND] ++;

    return Control(IOCTL_PACKET_SEND, aData, aSize_byte, NULL, 0);
}

// Private
/////////////////////////////////////////////////////////////////////////////

// aCommandQueue [---;RW-]
// aKernel       [---;R--]
//
// Exception  KmsLib::Exception *  CODE_NOT_ENOUGH_MEMORY
//                                 See Process_Internal::Buffer_Allocate
// Threads    Apps
Buffer_Data * Adapter_Internal::Buffer_Allocate(cl_command_queue aCommandQueue, cl_kernel aKernel)
{
    assert(NULL != aCommandQueue);
    assert(NULL != aKernel      );

    assert(OPEN_NET_BUFFER_QTY <= mBufferCount);
    assert(NULL                != mDebugLog   );
    assert(NULL                != mProcessor  );

    if (OPEN_NET_BUFFER_QTY <= mBufferCount)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_ENOUGH_MEMORY,
            "Too many buffer", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }

    Buffer_Data * lResult = mProcessor->Buffer_Allocate(mConfig.mPacketSize_byte, aCommandQueue, aKernel, mBuffers + mBufferCount);

    mBufferCount++;

    mStatistics[OpenNet::ADAPTER_STATS_BUFFER_ALLOCATED] ++;

    return lResult;
}

void Adapter_Internal::Config_Update()
{
    assert(PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte      );
    assert(PACKET_SIZE_MIN_byte <= mConfig.mPacketSize_byte      );
    assert(PACKET_SIZE_MAX_byte >= mDriverConfig.mPacketSize_byte);
    assert(PACKET_SIZE_MIN_byte <= mDriverConfig.mPacketSize_byte);

    mConfig.mPacketSize_byte = mDriverConfig.mPacketSize_byte;
}

// aIn        [--O;R--]
// aOut       [--O;-W-]
// aInfo_byte [--O;-W-]
//
// Threads  Apps
OpenNet::Status Adapter_Internal::Control(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte)
{
    assert(0 != aCode);

    assert(NULL != mDebugLog);
    assert(NULL != mHandle  );

    try
    {
        unsigned int lInfo_byte = mHandle->Control(aCode, aIn, aInSize_byte, aOut, aOutSize_byte);

        if (NULL != aInfo_byte)
        {
            (*aInfo_byte) = lInfo_byte;
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog->Log(eE);
        return ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aE [---;R--]
//
// Threads  Apps, Worker
OpenNet::Status ExceptionToStatus(const KmsLib::Exception * aE)
{
    assert(NULL != aE);

    switch (aE->GetCode())
    {
    case KmsLib::Exception::CODE_IOCTL_ERROR      : return OpenNet::STATUS_IOCTL_ERROR    ;
    case KmsLib::Exception::CODE_NOT_ENOUGH_MEMORY: return OpenNet::STATUS_TOO_MANY_BUFFER;
    case KmsLib::Exception::CODE_OPEN_CL_ERROR    : return OpenNet::STATUS_OPEN_CL_ERROR  ;
    }

    printf("%s ==> STATUS_EXCEPTION\n", KmsLib::Exception::GetCodeName(aE->GetCode()));
    return OpenNet::STATUS_EXCEPTION;
}
