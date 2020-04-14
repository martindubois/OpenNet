
// Author     KMS - Martin Dubois, P.Eng.
// Copyright  (C) 2018-2020 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Internal/Processor_Internal.cpp

#define __CLASS__ "Adapter_Internal::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== C ==================================================================
#include <stdint.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Common =============================================================
#include "../Common/Constants.h"

// ===== OpenNet/Internal ===================================================

#include "../EthernetAddress.h"
#include "../FolderFinder.h"
#include "../Thread_Functions.h"
#include "../Utils.h"

#include "Buffer_Internal.h"

#include "Adapter_Internal.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static const IoCtl_Packet_Send_Ex_In LOOP_BACK_PACKET_HEADER = { 0, 64, 64 };

static const uint8_t LOOP_BACK_PACKET_DATA[64] =
{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88
};

// Public
/////////////////////////////////////////////////////////////////////////////

// Threads  Apps
Adapter_Internal::~Adapter_Internal()
{
    // printf( __CLASS__ "~AdapterInternal()\n" );

    assert(NULL != mHandle);

    // new ==> delete
    delete mHandle;
}

unsigned int Adapter_Internal::GetBufferQty() const
{
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

OpenNetK::Adapter_Type Adapter_Internal::GetType() const
{
    return mInfo.mAdapterType;
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
            mDebugLog->Log(__FILE__, __CLASS__ "SetPacketSize", __LINE__);
            throw new KmsLib::Exception(KmsLib::Exception::CODE_INVALID_ARGUMENT,
                "Invalid packet size", NULL, __FILE__, __CLASS__ "SetPacketSize", __LINE__, mDriverConfig.mPacketSize_byte);
        }
    }
}

// Exception  KmsLib::Exception *  See Adapter_Internal::Buffer_Release
void Adapter_Internal::Buffers_Release()
{
    assert(0 < mBufferCount);

    mBufferCount = 0;
}

// aIn [---;R--] The IoCtl_Connect_In structure common to all connected
//               adapters in a system.
//
// Exception  KmsLib::Exception *  CODE_IOCTL_ERROR
//                                 See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
void Adapter_Internal::Connect(IoCtl_Connect_In * aIn)
{
    assert(NULL != aIn );

    assert(ADAPTER_NO_UNKNOWN == mConnect_Out.mAdapterNo);
    assert(NULL               != mHandle                );

    mHandle->Control(IOCTL_CONNECT, aIn, sizeof(IoCtl_Connect_In), &mConnect_Out, sizeof(mConnect_Out));

    if ((ADAPTER_NO_UNKNOWN == mConnect_Out.mAdapterNo) || (ADAPTER_NO_QTY <= mConnect_Out.mAdapterNo))
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_IOCTL_ERROR,
            "The connection is not valid", NULL, __FILE__, __CLASS__ "Connect", __LINE__, mConnect_Out.mAdapterNo);
    }
}

// aIn [---;R--] The header and the packet to send. The header contain the
//               packet size.
//
// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
void Adapter_Internal::Packet_Send_Ex(const IoCtl_Packet_Send_Ex_In * aIn)
{
    assert(NULL                            != aIn         );

    unsigned int lRet = mHandle->Control(IOCTL_PACKET_SEND_EX, aIn, sizeof( IoCtl_Packet_Send_Ex_In ) + aIn->mSize_byte, NULL, 0);
    assert(0 == lRet);
}

// aOut [---;-W-]
//
// Exception  KmsLib::Exception *  CODE_IOCTL_ERROR
//                                 See KmsLib::Windows::DriverHandle::Control
void Adapter_Internal::PacketGenerator_GetConfig(OpenNetK::PacketGenerator_Config * aOut)
{
    assert(NULL != aOut);

    unsigned int lRet = mHandle->Control(IOCTL_PACKET_GENERATOR_CONFIG_GET, NULL, 0, aOut, sizeof(OpenNetK::PacketGenerator_Config));
    if (sizeof(OpenNetK::PacketGenerator_Config) != lRet)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_IOCTL_ERROR, "The driver did not return enough data", NULL, __FILE__, __CLASS__ "PacketGenerator_GetConfig", __LINE__, lRet);
    }
}

// aIn [---;RW-]
//
// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
void Adapter_Internal::PacketGenerator_SetConfig(OpenNetK::PacketGenerator_Config * aInOut)
{
    assert(NULL != aInOut);

    unsigned int lRet = mHandle->Control(IOCTL_PACKET_GENERATOR_CONFIG_SET, aInOut, sizeof(OpenNetK::PacketGenerator_Config), aInOut, sizeof(OpenNetK::PacketGenerator_Config));
    assert(sizeof(OpenNetK::PacketGenerator_Config) == lRet);
}

// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
void Adapter_Internal::PacketGenerator_Start()
{
    unsigned int lRet = mHandle->Control(IOCTL_PACKET_GENERATOR_START, NULL, 0, NULL, 0);
    assert(0 == lRet);
}

// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
void Adapter_Internal::PacketGenerator_Stop()
{
    unsigned int lRet = mHandle->Control(IOCTL_PACKET_GENERATOR_STOP, NULL, 0, NULL, 0);
    assert(0 == lRet);
}

// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
void Adapter_Internal::SendLoopBackPackets()
{
    assert(NULL != mHandle);

    try
    {
        mHandle->Control(IOCTL_PACKET_SEND_EX, &LOOP_BACK_PACKET_HEADER, sizeof(LOOP_BACK_PACKET_HEADER) + sizeof(LOOP_BACK_PACKET_DATA), NULL, 0);
        mStatistics[OpenNet::ADAPTER_STATS_LOOP_BACK_PACKET] += LOOP_BACK_PACKET_HEADER.mRepeatCount;
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->LogTime();
        mDebugLog->Log(__FILE__, __CLASS__ "SendLoopBackPakets", __LINE__);
        mDebugLog->Log(eE);
    }
}

// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
void Adapter_Internal::Start()
{
    // printf( __CLASS__ "Start()\n" );

    assert(   0 <  mBufferCount);
    assert(NULL != mHandle     );

    // We must call IOCTL_START before starting the event processing thread
    // because IOCTL_START flush events from previous acquisition.
    mHandle->Control(IOCTL_START, mBuffers, sizeof(OpenNetK::Buffer) * mBufferCount, NULL, 0);

    if (NULL != mEvent_Callback)
    {
        ThreadBase::Start();
    }

    mRunning = true;
}

// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
// Threads    Worker
void Adapter_Internal::Stop()
{
    assert(NULL != mHandle);

    mRunning = false;

    if ((NULL != mEvent_Callback) && IsRunning())
    {
        ThreadBase::Stop();

        Stop_Internal();
    }

    mHandle->Control(IOCTL_STOP, NULL, 0, NULL, 0);
}

// Return  This method return the address of the create Thread instance.
//
// Thread  Apps
Thread * Adapter_Internal::Thread_Prepare()
{
    // printf( __CLASS__ "Thread_Pepare()\n" );

    assert(NULL == mThread);

    if (NULL != mSourceCode)
    {
        assert(NULL != mProcessor);

        OpenNet::Kernel * lKernel = dynamic_cast<OpenNet::Kernel *>(mSourceCode);
        if (NULL != lKernel)
        {
            mThread = Thread_Prepare_Internal(lKernel);
            assert(NULL != mThread);

            return mThread;
        }

        OpenNet::Function * lFunction = dynamic_cast<OpenNet::Function *>(mSourceCode);
        assert(NULL != lFunction);

        Thread_Functions * lThread = mProcessor->Thread_Get();
        assert(NULL != lThread);

        lThread->AddAdapter(this, *lFunction);

        mThread = lThread;
    }

    return NULL;
}

// ===== OpenNet::Adapter ===================================================

OpenNet::Status Adapter_Internal::GetAdapterNo(unsigned int * aOut)
{
    assert(NULL != mDebugLog);

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "GetAdapterNo", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    OpenNet::Status lResult;

    if (ADAPTER_NO_UNKNOWN == mConnect_Out.mAdapterNo)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "GetAdapterNo", __LINE__);
        lResult = OpenNet::STATUS_ADAPTER_NOT_CONNECTED;
    }
    else
    {
        if (ADAPTER_NO_QTY <= mConnect_Out.mAdapterNo)
        {
            mDebugLog->Log(__FILE__, __CLASS__ "GetAdapterNo", __LINE__);
            lResult = OpenNet::STATUS_CORRUPTED_DRIVER_DATA;
        }
        else
        {
            (*aOut) = mConnect_Out.mAdapterNo;
            lResult = OpenNet::STATUS_OK;
        }
    }

    return lResult;
}

OpenNet::Status Adapter_Internal::GetConfig(Config * aOut) const
{
    assert(NULL != mDebugLog);

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "GetConfig", __LINE__);
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
        mDebugLog->Log(__FILE__, __CLASS__ "GetInfo", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    memcpy(aOut, &mInfo, sizeof(mInfo));

    return OpenNet::STATUS_OK;
}

const char * Adapter_Internal::GetName() const
{
    return mName;
}

OpenNet::Status Adapter_Internal::GetState(Adapter::State * aOut)
{
    assert(NULL != mDebugLog);
    assert(NULL != mHandle  );

    if (NULL == aOut)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "GetState", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    return Control(IOCTL_STATE_GET, NULL, 0, aOut, sizeof(Adapter::State));
}

OpenNet::Status Adapter_Internal::GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset)
{
    assert(NULL != mDebugLog);
    assert(NULL != mHandle  );

    if ((NULL == aOut) && (0 < aOutSize_byte))
    {
        mDebugLog->Log(__FILE__, __CLASS__ "GetStatistics", __LINE__);
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

    IoCtl_Statistics_Get_In lIn;

    memset(&lIn, 0, sizeof(lIn));

    lIn.mFlags.mReset = aReset;
    lIn.mOutputSize_byte = aOutSize_byte;

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
    // printf( __CLASS__ "IsConnected()\n" );

    Adapter::State lState;

    OpenNet::Status lStatus = GetState(&lState);
    assert(OpenNet::STATUS_OK == lStatus);

    assert((ADAPTER_NO_UNKNOWN == lState.mAdapterNo) || (ADAPTER_NO_QTY > lState.mAdapterNo));

    return (ADAPTER_NO_UNKNOWN != lState.mAdapterNo);
}

bool Adapter_Internal::IsConnected(const OpenNet::System & aSystem)
{
    if (NULL == (&aSystem))
    {
        mDebugLog->Log(__FILE__, __CLASS__ "IsConnected", __LINE__);
        return false;
    }

    Adapter::State lState;

    OpenNet::Status lStatus = GetState(&lState);
    assert(OpenNet::STATUS_OK == lStatus);

    OpenNet::System::Info lInfo;

    lStatus = aSystem.GetInfo(&lInfo);
    assert(OpenNet::STATUS_OK == lStatus);

    assert((ADAPTER_NO_UNKNOWN == lState.mAdapterNo) || (ADAPTER_NO_QTY > lState.mAdapterNo));

    return ((ADAPTER_NO_UNKNOWN != lState.mAdapterNo) && (lInfo.mSystemId == lState.mSystemId));
}

OpenNet::Status Adapter_Internal::Packet_Send(const void * aData, unsigned int aSize_byte)
{
    assert(NULL != mDebugLog);

    if (NULL == aData)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Packet_Send", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 >= aSize_byte)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Packet_Send", __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    if (mInfo.mPacketSize_byte < aSize_byte)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Packet_Send", __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    mStatistics[OpenNet::ADAPTER_STATS_PACKET_SEND] ++;

    unsigned char * lBuffer = new unsigned char[sizeof(IoCtl_Packet_Send_Ex_In) + aSize_byte];
    assert(NULL != lBuffer);

    IoCtl_Packet_Send_Ex_In * lIn = reinterpret_cast<IoCtl_Packet_Send_Ex_In *>(lBuffer);

    memset(lIn, 0, sizeof(IoCtl_Packet_Send_Ex_In));
    memcpy(lIn + 1, aData, aSize_byte);

    lIn->mRepeatCount =          1;
    lIn->mSize_byte   = aSize_byte;

    OpenNet::Status lResult;

    try
    {
        Packet_Send_Ex(lIn);
        lResult = OpenNet::STATUS_OK;
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(eE);
        lResult = Utl_ExceptionToStatus(eE);
    }

    delete[] lBuffer;

    return lResult;
}

OpenNet::Status Adapter_Internal::ResetConfig()
{
    OpenNet::Status lResult = Control(IOCTL_CONFIG_RESET, NULL, 0, NULL, 0);
    if (OpenNet::STATUS_OK == lResult)
    {
        Config_Reset();

        lResult = Control(IOCTL_CONFIG_GET, NULL, 0, &mDriverConfig, sizeof(mDriverConfig));
        if (OpenNet::STATUS_OK == lResult)
        {
            Config_Update();
        }
    }

    return lResult;
}

OpenNet::Status Adapter_Internal::ResetInputFilter()
{
    // printf( __CLASS__ "ResetInputFilter()\n" );

    assert(OPEN_NET_BUFFER_QTY >= mBufferCount);
    assert(NULL                != mDebugLog   );

    if (NULL == mSourceCode)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "ResetInputFilter", __LINE__);
        return OpenNet::STATUS_FILTER_NOT_SET;
    }

    if (0 < mBufferCount)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "ResetInputFilter", __LINE__);
        return OpenNet::STATUS_ADAPTER_RUNNING;
    }

    mSourceCode = NULL;

    try
    {
        ResetInputFilter_Internal();
    }
    catch ( KmsLib::Exception * eE )
    {
        mDebugLog->Log( __FILE__, __CLASS__ "ResetInputFilter", __LINE__ );
        mDebugLog->Log( eE );

        return Utl_ExceptionToStatus( eE );
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::ResetProcessor()
{
    assert(NULL != mDebugLog);

    if (NULL == mProcessor)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "ResetProcessor", __LINE__);
        return OpenNet::STATUS_PROCESSOR_NOT_SET;
    }

    if (NULL != mSourceCode)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "ResetProcessor", __LINE__);
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
    assert(mDriverConfig.mFlags.mMulticastPromiscuousDisable == mConfig.mFlags.mMulticastPromiscuousDisable);
    assert(mDriverConfig.mFlags.mUnicastPromiscuousDisable   == mConfig.mFlags.mUnicastPromiscuousDisable  );
    assert(mDriverConfig.mPacketSize_byte == mConfig.mPacketSize_byte);
    assert(0 == memcmp(&mDriverConfig.mEthernetAddress, mConfig.mEthernetAddress, sizeof(mDriverConfig.mEthernetAddress)));
    assert(NULL                           != mDebugLog               );

    if (NULL == (&aConfig))
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetConfig", __LINE__);
        return OpenNet::STATUS_INVALID_REFERENCE;
    }

    if (PACKET_SIZE_MAX_byte < aConfig.mPacketSize_byte)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetConfig", __LINE__);
        return OpenNet::STATUS_PACKET_TOO_LARGE;
    }

    if (PACKET_SIZE_MIN_byte > aConfig.mPacketSize_byte)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetConfig", __LINE__);
        return OpenNet::STATUS_PACKET_TOO_SMALL;
    }

    if (OPEN_NET_BUFFER_QTY < aConfig.mBufferQty)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetConfig", __LINE__);
        return OpenNet::STATUS_TOO_MANY_BUFFER;
    }

    memcpy(&mConfig, &aConfig, sizeof(mConfig));

    mDriverConfig.mFlags.mMulticastPromiscuousDisable = mConfig.mFlags.mMulticastPromiscuousDisable;
    mDriverConfig.mFlags.mUnicastPromiscuousDisable   = mConfig.mFlags.mUnicastPromiscuousDisable  ;
    mDriverConfig.mPacketSize_byte = mConfig.mPacketSize_byte;

    memcpy(&mDriverConfig.mEthernetAddress, &aConfig.mEthernetAddress, sizeof(mDriverConfig.mEthernetAddress));

    OpenNet::Status lResult = Control(IOCTL_CONFIG_SET, &mDriverConfig, sizeof(mDriverConfig), &mDriverConfig, sizeof(mDriverConfig));
    if (OpenNet::STATUS_OK == lResult)
    {
        Config_Update();
    }

    return lResult;
}

// TODO  OpenNet.Adapter
//       Normal (Feature) - Permettre de changer le filtre pendant
//       l'execution pour le Kernel en premier et pour les Fonction aussi.

OpenNet::Status Adapter_Internal::SetInputFilter(OpenNet::SourceCode * aSourceCode)
{
    assert(NULL != mDebugLog);

    if (NULL == aSourceCode)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetInputFilter", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (NULL != mSourceCode)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetInputFilter", __LINE__);
        return OpenNet::STATUS_FILTER_ALREADY_SET;
    }

    if (NULL == mProcessor)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetInputFilter", __LINE__);
        return OpenNet::STATUS_PROCESSOR_NOT_SET;
    }

    if (ADAPTER_NO_UNKNOWN == mConnect_Out.mAdapterNo)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetInputFilter", __LINE__);
        return OpenNet::STATUS_ADAPTER_NOT_CONNECTED;
    }

    try
    {
        OpenNet::Kernel * lKernel = dynamic_cast<OpenNet::Kernel *>(aSourceCode);
        if (NULL != lKernel)
        {
            SetInputFilter_Internal(lKernel);
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetInputFilter", __LINE__);
        mDebugLog->Log(eE);
        return Utl_ExceptionToStatus(eE);
    }

    mSourceCode = aSourceCode;

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::SetProcessor(OpenNet::Processor * aProcessor)
{
    assert(NULL != mDebugLog);

    if (NULL == aProcessor)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetProcessor", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (NULL != mProcessor)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetProcessor", __LINE__);
        return OpenNet::STATUS_PROCESSOR_ALREADY_SET;
    }

    mProcessor = dynamic_cast<Processor_Internal *>(aProcessor);

    if (NULL == mProcessor)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "SetProcessor", __LINE__);
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
        mDebugLog->Log(__FILE__, __CLASS__ "Display", __LINE__);
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

OpenNet::Status Adapter_Internal::Event_RegisterCallback(Event_Callback aCallback, void * aContext)
{
    // printf(__CLASS__ "Event_RegisterCallback( ,  )\n");

    assert(NULL != mHandle);

    try
    {
        if ((NULL != mEvent_Callback) && IsRunning())
        {
            ThreadBase::Stop();

            mHandle->Control(IOCTL_EVENT_WAIT_CANCEL, NULL, 0, NULL, 0);

            Wait(true, 1000);
        }

        mEvent_Callback = aCallback;
        mEvent_Context  = aContext ;

        if ((NULL != mEvent_Callback) && mRunning)
        {
            ThreadBase::Start();
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Event_RegisterCallback", __LINE__);
        mDebugLog->Log(eE);
        return Utl_ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::Read(void * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte)
{
    if ((NULL == aOut) || (NULL == aInfo_byte))
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Read", __LINE__);
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 >= aOutSize_byte)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Read", __LINE__);
        return OpenNet::STATUS_INVALID_SIZE;
    }

    try
    {
        (*aInfo_byte) = mHandle->Read(aOut, aOutSize_byte);
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Read", __LINE__);
        mDebugLog->Log(eE);
        return Utl_ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status Adapter_Internal::Tx_Disable()
{
    return Control(IOCTL_TX_DISABLE, NULL, 0, NULL, 0);
}

OpenNet::Status Adapter_Internal::Tx_Enable()
{
    return Control(IOCTL_TX_ENABLE, NULL, 0, NULL, 0);
}

// ===== KmsLib::ThreadBase =================================================

// CRITICAL PATH  BufferEvent
unsigned int Adapter_Internal::Run()
{
    assert(NULL != mEvent_Callback);

    try
    {
        IoCtl_Event_Wait_In lIn;
        OpenNetK::Event     lOut[32];

        memset(&lIn, 0, sizeof(lIn));

        lIn.mOutputSize_byte = sizeof(lOut);

        while (IsRunning())
        {
            unsigned int lInfo_byte = mHandle->Control(IOCTL_EVENT_WAIT, &lIn, sizeof(lIn), lOut, sizeof(lOut));

            unsigned int lCount = lInfo_byte / sizeof(OpenNetK::Event);
            if (0 == lCount)
            {
                mDebugLog->Log(__FILE__, __CLASS__ "Run", __LINE__);
                break;
            }

            for (unsigned int i = 0; i < lCount; i++)
            {
                Event_Process(lOut[i]);
            }
        }
    }
    catch (KmsLib::Exception * eE)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Run", __LINE__);
        mDebugLog->Log(eE);
    }
    catch (...)
    {
        mDebugLog->Log(__FILE__, __CLASS__ "Run", __LINE__);
    }

    return 0;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// aHandle    [DK-;RW-]
// aDebugLog  [-K-;RW-]
//
// Exception  KmsLib::Exception *  See KmsLib::Windows::DriverHandle::Control
// Threads    Apps
Adapter_Internal::Adapter_Internal(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog)
    : mBufferCount(        0)
    , mDebugLog   (aDebugLog)
    , mHandle     (aHandle  )
    , mProcessor  (NULL     )
    , mRunning    (false    )
    , mSourceCode (NULL     )
    , mThread     (NULL     )
{
    assert(NULL != aHandle  );
    assert(NULL != aDebugLog);

    mDebugLog->Log( "Adapter_Internal::Adapter_Internal( , ,  )" );

    memset(&mConnect_Out , 0, sizeof(mConnect_Out ));
    memset(&mDriverConfig, 0, sizeof(mDriverConfig));
    memset(&mInfo        , 0, sizeof(mInfo        ));
    memset(&mName        , 0, sizeof(mName        ));
    memset(&mStatistics  , 0, sizeof(mStatistics  ));

    mConnect_Out.mAdapterNo = ADAPTER_NO_UNKNOWN;

    mDriverConfig.mFlags.mMulticastPromiscuousDisable = false;
    mDriverConfig.mFlags.mUnicastPromiscuousDisable   = false;
	mDriverConfig.mPacketSize_byte = PACKET_SIZE_MAX_byte;

    mHandle->Control(IOCTL_CONFIG_GET, NULL, 0, &mDriverConfig, sizeof(mDriverConfig));
    mHandle->Control(IOCTL_INFO_GET  , NULL, 0, &mInfo        , sizeof(mInfo        ));

    Config_Reset ();
    Config_Update();

    strncpy_s(mName, mInfo.mVersion_Hardware.mComment, sizeof(mName) - 1);
    strcat_s (mName, " ");

    unsigned int lOffset_byte = static_cast<unsigned int>(strlen(mName));

    if (44 < lOffset_byte)
    {
        lOffset_byte = 44;
    }

    OpenNet::EthernetAddress_GetText(mInfo.mEthernetAddress, mName + strlen(mName), sizeof(mName) - lOffset_byte);

    License_Verify();
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
    assert(NULL != mHandle);

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
        mDebugLog->Log(__FILE__, __CLASS__ "Control", __LINE__);
        mDebugLog->Log(eE);
        return Utl_ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}

// Private
/////////////////////////////////////////////////////////////////////////////

// aIndex  The index of the buffer to retrieve
//
// Return  The Buffer_Internal instance or NULL if the index is not valid.
//
// CRITICAL PATH  BufferEvent  1 / Buffer event
Buffer_Internal * Adapter_Internal::GetBuffer(unsigned int aIndex)
{
    assert(NULL != mThread);

    return mThread->GetBuffer(this, aIndex);
}

void Adapter_Internal::Config_Reset()
{
    assert(PACKET_SIZE_MAX_byte >= mInfo.mPacketSize_byte);
    assert(PACKET_SIZE_MIN_byte <= mInfo.mPacketSize_byte);

    memset(&mConfig, 0, sizeof(mConfig));

    mConfig.mBufferQty       = 4;
    mConfig.mPacketSize_byte = mInfo.mPacketSize_byte;
}

void Adapter_Internal::Config_Update()
{
    assert(PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte      );
    assert(PACKET_SIZE_MIN_byte <= mConfig.mPacketSize_byte      );
    assert(PACKET_SIZE_MAX_byte >= mDriverConfig.mPacketSize_byte);
    assert(PACKET_SIZE_MIN_byte <= mDriverConfig.mPacketSize_byte);

    mConfig.mFlags.mMulticastPromiscuousDisable = mDriverConfig.mFlags.mMulticastPromiscuousDisable;
    mConfig.mFlags.mUnicastPromiscuousDisable   = mDriverConfig.mFlags.mUnicastPromiscuousDisable  ;
    mConfig.mPacketSize_byte = mDriverConfig.mPacketSize_byte;

    memcpy(&mConfig.mEthernetAddress, mDriverConfig.mEthernetAddress, sizeof(mConfig.mEthernetAddress));
}

// aEvent [---;R--] The event to proces
//
// Thread  Event
//
// CRITICAL PATH  BufferEvent  1 / Buffer event
void Adapter_Internal::Event_Process(const OpenNetK::Event & aEvent)
{
    assert(NULL != (&aEvent));

    assert(NULL != mEvent_Callback);

    switch (aEvent.mType)
    {
    case OpenNetK::EVENT_TYPE_BUFFER :
        Buffer_Internal * lBuffer;

        lBuffer = GetBuffer(aEvent.mData);
        if (NULL != lBuffer)
        {
            lBuffer->FetchBufferInfo();

            mEvent_Callback(mEvent_Context, aEvent.mType, aEvent.mTimestamp_us, aEvent.mData, lBuffer);
        }
        break;

    case OpenNetK::EVENT_TYPE_WAIT_CANCEL :
        break;

    default: assert(false);
    }
}

void Adapter_Internal::License_Verify()
{
    assert(NULL != mDebugLog);

    assert(NULL != gFolderFinder);

    char lFileName[1024];

    sprintf_s(lFileName, "%s" SLASH "License.txt", gFolderFinder->GetBinaryFolder());

    FILE * lFile;

    if (0 == fopen_s(&lFile, lFileName, "r"))
    {
        char lLine[1024];

        while (NULL != fgets(lLine, sizeof(lLine), lFile))
        {
            unsigned int lAddr[6];
            unsigned int lKey;

            if (7 == sscanf_s(lLine, "%x %x %x %x %x %x %x", lAddr + 0, lAddr + 1, lAddr + 2, lAddr + 3, lAddr + 4, lAddr + 5, &lKey))
            {
                bool lFound = true;

                for (unsigned int i = 0; i < 6; i++)
                {
                    if (mInfo.mEthernetAddress.mAddress[i] != lAddr[i])
                    {
                        lFound = false;
                        break;
                    }
                }

                if (lFound)
                {
                    IoCtl_License_Set_In  lIn ;
                    IoCtl_License_Set_Out lOut;

                    memset(&lIn , 0, sizeof(lIn ));
                    memset(&lOut, 0, sizeof(lOut));

                    lIn.mKey = lKey;

                    mHandle->Control(IOCTL_LICENSE_SET, &lIn, sizeof(lIn), &lOut, sizeof(lOut));

                    if (!lOut.mFlags.mLicenseOk)
                    {
                        mDebugLog->Log(__FILE__, __CLASS__ "License_Verify", __LINE__);
                    }
                }
            }
        }
    }
    else
    {
        mDebugLog->Log(__FILE__, __CLASS__ "License_Verify", __LINE__);
    }
}
