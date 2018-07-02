
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

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/IoCtl.h"
#include "../Common/OpenNet/Adapter_Statistics.h"

// ===== OpenNet ============================================================
#include "EthernetAddress.h"
#include "OCLW.h"
#include "Processor_Internal.h"

#include "Adapter_Internal.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

// --> INIT <--+<------------------------------------------------------+
//      |      |                                                       |
//      |     STOPPING <--+<------+<--------+<--------------+          |
//      |                 |       |         |               |          |
//      |                 |       |    +--> RUNNING --+     |          |
//      |                 |       |    |              |     |          |
//      +--> START_REQUESTED --> STARTING ----------->+--> STOP_REQUESTED

#define STATE_INIT            (0)
#define STATE_RUNNING         (1)
#define STATE_START_REQUESTED (2)
#define STATE_STARTING        (3)
#define STATE_STOP_REQUESTED  (4)
#define STATE_STOPPING        (5)

#define STATE_QTY (6)

static const char * STATE_NAMES[STATE_QTY] =
{
    "INIT"           ,
    "RUNNING"        ,
    "START_REQUESTED",
    "STARTING"       ,
    "STOP_REQUESTED" ,
    "STOPPING"       ,
};

static const uint8_t LOOP_BACK_PACKET[64] =
{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88
};

// Static functions declaration
/////////////////////////////////////////////////////////////////////////////

static OpenNet::Status ExceptionToStatus(const KmsLib::Exception * aE);

// ===== Entry point ========================================================
static DWORD WINAPI Run(LPVOID aParameter);

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
    , mFilter     (NULL)
    , mHandle     (aHandle)
    , mProcessor  (NULL)
    , mThread     (NULL)
    , mState      (STATE_INIT)
{
    assert(NULL != aHandle  );
    assert(NULL != aDebugLog);

    InitializeCriticalSection(&mZone0);

    memset(&mConfig, 0, sizeof(mConfig));
    memset(&mInfo  , 0, sizeof(mInfo  ));
    memset(&mName  , 0, sizeof(mName  ));

    memset(&mStatistics , 0, sizeof(mStatistics));

    mHandle->Control(IOCTL_CONFIG_GET, NULL, 0, &mConfig, sizeof(mConfig));
    mHandle->Control(IOCTL_INFO_GET  , NULL, 0, &mInfo  , sizeof(mInfo  ));

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

    EnterCriticalSection(&mZone0);
        switch (mState)
        {
        case STATE_INIT: break;

        case STATE_RUNNING :
        case STATE_STARTING:
        case STATE_STOPPING:
            Stop_Request_Zone0();
            Stop_Wait_Zone0   (NULL, NULL);
            break;

        default: assert(false);
        }
    LeaveCriticalSection(&mZone0);

    if (0 < mBufferCount)
    {
        Buffer_Release(mBufferCount);
        assert(0 == mBufferCount);
    }

    DeleteCriticalSection(&mZone0);

    delete mHandle;
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

// Thread  Apps
void Adapter_Internal::SendLoopBackPackets()
{
    assert(NULL != mHandle);

    for (unsigned i = 0; i < 64; i++)
    {
        mHandle->Control(IOCTL_PACKET_SEND, LOOP_BACK_PACKET, sizeof(LOOP_BACK_PACKET), NULL, 0);
        mStatistics[OpenNet::ADAPTER_STATS_LOOP_BACK_PACKET] ++;
    }
}

// Exception  KmsLib::Exception *  CODE_STATE_ERROR
//                                 CODE_THREAD_ERROR
// Threads  Apps
void Adapter_Internal::Start()
{
    assert(OPEN_NET_BUFFER_QTY >= mBufferCount);
    assert(NULL                != mDebugLog   );
    assert(NULL                == mThread     );
    assert(0                   == mThreadId   );

    if (0 >= mBufferCount)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_STATE_ERROR,
            "No buffers", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }

    State_Change(STATE_INIT, STATE_START_REQUESTED);

    mThread = CreateThread(NULL, 0, ::Run, this, 0, &mThreadId);
    if (NULL == mThread)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_THREAD_ERROR,
            "CreateThread( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }

    mStatistics[OpenNet::ADAPTER_STATS_START] ++;
}

// Exception  KmsLib::Exception *  CODE_THREAD_ERROR
// Threads    Apps
void Adapter_Internal::Stop_Request()
{
    EnterCriticalSection(&mZone0);
        Stop_Request_Zone0();
    LeaveCriticalSection(&mZone0);

    mStatistics[OpenNet::ADAPTER_STATS_STOP_REQUEST] ++;
}

// Exception  KmsLib::Exception *  CODE_THREAD_ERROR
// Threads  Apps
void Adapter_Internal::Stop_Wait(TryToSolveHang aTryToSolveHang, void * aContext)
{
    EnterCriticalSection(&mZone0);
        Stop_Wait_Zone0(aTryToSolveHang, aContext);
    LeaveCriticalSection(&mZone0);

    mStatistics[OpenNet::ADAPTER_STATS_STOP_WAIT] ++;
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

unsigned int Adapter_Internal::GetPacketSize() const
{
    assert(PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);
    assert(PACKET_SIZE_MIN_byte <= mConfig.mPacketSize_byte);

    return mConfig.mPacketSize_byte;
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

    assert((ADAPTER_NO_UNKNOWN == lState.mAdapterNo) || (ADAPTER_NO_QTY > lState.mAdapterNo));

    return ((ADAPTER_NO_UNKNOWN != lState.mAdapterNo) && (aSystem.GetSystemId() == lState.mSystemId));
}

OpenNet::Status Adapter_Internal::ResetInputFilter()
{
    assert(NULL != mDebugLog);

    if (NULL == mFilter)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_FILTER_NOT_SET;
    }

    assert(NULL != mProcessor);

    if (0 < mBufferCount)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_BUFFER_ALLOCATED;
    }

    mFilter = NULL;

    try
    {
        mProcessor->Processing_Release(&mFilterData);
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog->Log(eE);
        return ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::ResetProcessor()
{
    assert(NULL != mDebugLog);

    if (NULL == mProcessor)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PROCESSOR_NOT_SET;
    }

    if (0 < mBufferCount)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_BUFFER_ALLOCATED;
    }

    if (NULL != mFilter)
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
    assert(NULL != mDebugLog);

    if (NULL == (&aConfig))
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_REFERENCE;
    }

    memcpy(&mConfig, &aConfig, sizeof(mConfig));

    return Control(IOCTL_CONFIG_SET, &mConfig, sizeof(mConfig), &mConfig, sizeof(mConfig));
}

OpenNet::Status Adapter_Internal::SetInputFilter(OpenNet::Filter * aFilter)
{
    assert(NULL != mDebugLog);

    if (NULL == aFilter)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (NULL != mFilter)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_FILTER_ALREADY_SET;
    }

    if (NULL == mProcessor)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PROCESSOR_NOT_SET;
    }

    try
    {
        mProcessor->Processing_Create(&mFilterData, aFilter);
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog->Log(eE);
        return ExceptionToStatus(eE);
    }

    mFilter = aFilter;

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::SetPacketSize(unsigned int aSize_byte)
{
    assert(NULL != mDebugLog);

    if (PACKET_SIZE_MAX_byte < aSize_byte)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    if (PACKET_SIZE_MIN_byte > aSize_byte)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    Config lConfig;

    OpenNet::Status lStatus = GetConfig(&lConfig);
    if (OpenNet::STATUS_OK != lStatus)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return lStatus;
    }

    lConfig.mPacketSize_byte = aSize_byte;

    lStatus = SetConfig(lConfig);
    if (OpenNet::STATUS_OK != lStatus)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return lStatus;
    }

    lStatus = GetConfig(&lConfig);
    if (OpenNet::STATUS_OK != lStatus)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return lStatus;
    }

    if (lConfig.mPacketSize_byte != aSize_byte)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_PACKET_SIZE;
    }
    
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

OpenNet::Status Adapter_Internal::Buffer_Allocate(unsigned int aCount)
{
    assert(NULL != mDebugLog);

    if (0 >= aCount)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_BUFFER_COUNT;
    }

    if (NULL == mFilter)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_FILTER_NOT_SET;
    }

    if (NULL == mProcessor)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_PROCESSOR_NOT_SET;
    }

    if (!IsConnected())
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_ADAPTER_NOT_CONNECTED;
    }

    try
    {
        for (unsigned int i = 0; i < aCount; i++)
        {
            Buffer_Allocate();
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

OpenNet::Status Adapter_Internal::Buffer_Release(unsigned int aCount)
{
    assert(NULL != mDebugLog);

    if (0 >= aCount)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_INVALID_BUFFER_COUNT;
    }

    try
    {
        for (unsigned int i = 0; i < aCount; i++)
        {
            Buffer_Release();
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

OpenNet::Status Adapter_Internal::Display(FILE * aOut) const
{
    assert(OPEN_NET_BUFFER_QTY >= mBufferCount);
    assert(NULL                != mDebugLog   );
    assert(STATE_QTY           >  mState      );

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    fprintf(aOut, "Adapter :\n");
    fprintf(aOut, "  %u Buffers\n"    , mBufferCount);
    fprintf(aOut, "  Filter    = %s\n", ((NULL == mFilter   ) ? "Not set" : mFilter   ->GetName()));
    fprintf(aOut, "  Name      = %s\n", mName);
    fprintf(aOut, "  Processor = %s\n", ((NULL == mProcessor) ? "Not set" : mProcessor->GetName()));
    fprintf(aOut, "  State     = %s\n", STATE_NAMES[mState]);
    fprintf(aOut, "  Thread Id = %u\n", mThreadId);
    
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

// Internal
/////////////////////////////////////////////////////////////////////////////

// TODO  OpenNet.Adapter_Internal
//       Augmenter la priorite du worker thread

// Thread  Worker
void Adapter_Internal::Run()
{
    assert(NULL != mDebugLog);
    assert(NULL != mHandle  );

    mStatistics[OpenNet::ADAPTER_STATS_RUN_ENTRY] ++;

    try
    {
        State_Change(STATE_START_REQUESTED, STATE_STARTING);

        assert(                  0 <  mBufferCount);
        assert(OPEN_NET_BUFFER_QTY >= mBufferCount);
        assert(NULL                != mHandle     );
        assert(NULL                != mProcessor  );

        unsigned int i;

        for (i = 0; i < mBufferCount; i++)
        {
            mBufferData[i].mMarkerValue = 0;

            mProcessor->Processing_Queue(&mFilterData, mBufferData + i);
            mStatistics[OpenNet::ADAPTER_STATS_RUN_QUEUE] ++;
        }

        mHandle->Control(IOCTL_START, mBuffers, sizeof(OpenNetK::Buffer) * mBufferCount, NULL, 0);

        Run_Loop();

        mHandle->Control(IOCTL_STOP, NULL, 0, NULL, 0);

        Run_Wait();
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog->Log(eE);

        mStatistics[OpenNet::ADAPTER_STATS_RUN_EXCEPTION] ++;
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);

        mStatistics[OpenNet::ADAPTER_STATS_RUN_UNEXPECTED_EXCEPTION] ++;
    }

    mState = STATE_STOPPING;

    mStatistics[OpenNet::ADAPTER_STATS_RUN_EXIT] ++;
}

// Private
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  CODE_NOT_ENOUGH_MEMORY
//                                 See Process_Internal::Buffer_Allocate
// Threads    Apps
void Adapter_Internal::Buffer_Allocate()
{
    assert(OPEN_NET_BUFFER_QTY <= mBufferCount);
    assert(NULL                != mDebugLog   );
    assert(NULL                != mProcessor  );

    if (OPEN_NET_BUFFER_QTY <= mBufferCount)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_NOT_ENOUGH_MEMORY,
            "Too many buffer", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }

    mProcessor->Buffer_Allocate(mConfig.mPacketSize_byte, &mFilterData, mBuffers + mBufferCount, mBufferData + mBufferCount);

    mBufferCount++;

    mStatistics[OpenNet::ADAPTER_STATS_BUFFER_ALLOCATED] ++;
}

// Exception  KmsLib::Exception *  CODE_INVALID_DATA
//                                 See KmsLib::Windows::DriverHandle::Control
//                                 See Processor_Internal::Buffer_Release
// Threads    Apps
void Adapter_Internal::Buffer_Release()
{
    assert(                  0 <  mBufferCount);
    assert(OPEN_NET_BUFFER_QTY >= mBufferCount);
    assert(NULL                != mHandle     );
    assert(NULL                != mProcessor  );

    mBufferCount--;

    mProcessor->Buffer_Release(mBufferData + mBufferCount);

    mStatistics[OpenNet::ADAPTER_STATS_BUFFER_RELEASED] ++;
}

// aIn  [--O;R--]
// aOut [--O;-W-]
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

void Adapter_Internal::Run_Iteration(unsigned int aIndex)
{
    assert(OPEN_NET_BUFFER_QTY > aIndex);

    assert(NULL != mProcessor);

    mProcessor->Processing_Wait(&mFilterData, mBufferData + aIndex);
    mStatistics[OpenNet::ADAPTER_STATS_RUN_ITERATION_WAIT] ++;

    mProcessor->Processing_Queue(&mFilterData, mBufferData + aIndex);
    mStatistics[OpenNet::ADAPTER_STATS_RUN_ITERATION_QUEUE] ++;
}

void Adapter_Internal::Run_Loop()
{
    assert(                  0 <  mBufferCount);
    assert(OPEN_NET_BUFFER_QTY >= mBufferCount);
    assert(NULL                != mDebugLog   );
    assert(NULL                != mHandle     );

    try
    {
        unsigned lIndex = 0;

        State_Change(STATE_STARTING, STATE_RUNNING);

        while (STATE_RUNNING == mState)
        {
            Run_Iteration(lIndex);

            lIndex = (lIndex + 1) % mBufferCount;
        }

        State_Change(STATE_STOP_REQUESTED, STATE_STOPPING);

        for (unsigned int i = 0; i < mBufferCount; i++)
        {
            mProcessor->Processing_Wait(&mFilterData, mBufferData + lIndex);

            lIndex = (lIndex + 1) % mBufferCount;
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        mDebugLog->Log(eE);

        mStatistics[OpenNet::ADAPTER_STATS_RUN_LOOP_EXCEPTION] ++;
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);

        mStatistics[OpenNet::ADAPTER_STATS_RUN_LOOP_UNEXPECTED_EXCEPTION] ++;
    }
}

void Adapter_Internal::Run_Wait()
{
    assert(NULL != mDebugLog);
    assert(NULL != mHandle  );

    State lState;

    for (unsigned int i = 0; i < 600; i++)
    {
        mHandle->Control(IOCTL_STATE_GET, NULL, 0, &lState, sizeof(lState));

        if (0 >= lState.mBufferCount)
        {
            return;
        }

        Sleep(1000);
    }

    // TODO  OpenNet.Adapter_Internal.Error_Handling
    //       This is a big program because the driver still use GPU buffer
    //       and the application is maybe going to release them.

    mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
    throw new KmsLib::Exception(KmsLib::Exception::CODE_TIMEOUT,
        "The driver did not release the buffers in time", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
}

// Exception  KmsLib::Exception *  CODE_STATE_ERROR
// Threads  Apps, Worker
void Adapter_Internal::State_Change(unsigned int aFrom, unsigned int aTo)
{
    assert(STATE_QTY > aFrom);
    assert(STATE_QTY < aTo  );

    assert(NULL != mDebugLog);

    bool lOK = false;

    EnterCriticalSection(&mZone0);

        assert(STATE_QTY > mState);

        if (mState == aFrom)
        {
            lOK    = true;
            mState = aTo ;
        }

    LeaveCriticalSection(&mZone0);

    if (!lOK)
    {
        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_STATE_ERROR,
            "Invalid state transition", NULL, __FILE__, __FUNCTION__, __LINE__, aTo);
    }
}

// Threads  Apps
void Adapter_Internal::Stop_Request_Zone0()
{
    assert(NULL != mThread);

    switch (mState)
    {
    case STATE_RUNNING :
    case STATE_STARTING:
        mState = STATE_STOP_REQUESTED;
        // no break;

    case STATE_STOPPING :
        break;

    default: assert(false);
    }
}

// aTryToSolveHang [--O;--X]
// aContext        [--O;---]
//
// This method release the Zone0 when it throw an exception.
//
// Exception  KmsLib::Exception *  CODE_THREAD_ERROR
// Threads  Apps
void Adapter_Internal::Stop_Wait_Zone0(TryToSolveHang aTryToSolveHang, void * aContext)
{
    assert(NULL != mDebugLog);
    assert(NULL != mThread  );

    switch (mState)
    {
    case STATE_STOP_REQUESTED:
    case STATE_STOPPING      :
        DWORD lRet;

        if (NULL != aTryToSolveHang)
        {
            LeaveCriticalSection(&mZone0);
                lRet = WaitForSingleObject(mThread, 3000);
            EnterCriticalSection(&mZone0);

            if (WAIT_OBJECT_0 == lRet) { break; }

            for (unsigned int i = 0; i < 1104; i ++)
            {
                aTryToSolveHang(aContext, this);

                LeaveCriticalSection(&mZone0);
                    lRet = WaitForSingleObject(mThread, 500);
                EnterCriticalSection(&mZone0);

                if (WAIT_OBJECT_0 == lRet) { break; }
            }
        }
        else
        {
            LeaveCriticalSection(&mZone0);
               lRet = WaitForSingleObject(mThread, 600000);
            EnterCriticalSection(&mZone0);
        }

        if (WAIT_OBJECT_0 == lRet) { break; }

        // TODO  OpenNet.Adapter_Internal.ErrorHandling
        //       This case is a big problem. Terminating the thread
        //       interracting with the GPU may let the system in an instabla
        //       state. Worst, in this case the drive still use the GPU
        //       buffer.

        LeaveCriticalSection(&mZone0);
        if (!TerminateThread(mThread, __LINE__))
        {
            mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
            throw new KmsLib::Exception(KmsLib::Exception::CODE_THREAD_ERROR,
                "TerminateThread( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
        }
        EnterCriticalSection(&mZone0);
        break;

    default: assert(false);
    }

    mState = STATE_INIT;

    BOOL lRetB = CloseHandle(mThread);

    mThread   = NULL;
    mThreadId = 0   ;

    if (!lRetB)
    {
        LeaveCriticalSection(&mZone0);

        mDebugLog->Log(__FILE__, __FUNCTION__, __LINE__);
        throw new KmsLib::Exception(KmsLib::Exception::CODE_THREAD_ERROR,
            "CloseHandle(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, 0);
    }
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

// ===== Entry point ========================================================

// Thread  Worker
DWORD WINAPI Run(LPVOID aParameter)
{
    assert(NULL != aParameter);

    Adapter_Internal * lThis = reinterpret_cast<Adapter_Internal *>(aParameter);

    lThis->Run();

    return 0;
}
