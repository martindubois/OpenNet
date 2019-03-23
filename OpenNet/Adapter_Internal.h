
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Adapter_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <vector>

// ===== Import/Includes ====================================================
#include <KmsLib/DebugLog.h>
#include <KmsLib/DriverHandle.h>
#include <KmsLib/ThreadBase.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/System.h>
#include <OpenNetK/Constants.h>
#include <OpenNetK/PacketGenerator_Types.h>

// ===== Common =============================================================
#include "../Common/IoCtl.h"
#include "../Common/OpenNet/Adapter_Statistics.h"

// ===== OpenNet ============================================================
#include "Buffer_Internal.h"
#include "Processor_Internal.h"

class Buffer_Internal;
class Thread         ;

// Class
/////////////////////////////////////////////////////////////////////////////

class Adapter_Internal : public OpenNet::Adapter, private KmsLib::ThreadBase
{

public:

    typedef void(*TryToSolveHang)(void *, Adapter_Internal *);

    virtual ~Adapter_Internal();

    unsigned int GetBufferQty () const;
    unsigned int GetPacketSize() const;

    OpenNetK::Adapter_Type GetType() const;

    void SetPacketSize(unsigned int aSize_byte);

    void Buffers_Release ();

    void Connect(IoCtl_Connect_In * aConnect);

    void Packet_Send_Ex(const IoCtl_Packet_Send_Ex_In * aIn);

    void PacketGenerator_GetConfig(OpenNetK::PacketGenerator_Config * aOut  );
    void PacketGenerator_SetConfig(OpenNetK::PacketGenerator_Config * aInOut);
    void PacketGenerator_Start    ();
    void PacketGenerator_Stop     ();

    void SendLoopBackPackets();

    void Start();
    void Stop ();

    virtual Thread * Thread_Prepare();

    // ===== OpenNet::Adapter ===============================================

    virtual OpenNet::Status GetAdapterNo    (unsigned int * aOut);
    virtual OpenNet::Status GetConfig       (Config       * aOut) const;
    virtual OpenNet::Status GetInfo         (Info         * aOut) const;
    virtual const char    * GetName         () const;
    virtual OpenNet::Status GetState        (Adapter::State * aOut);
    virtual OpenNet::Status GetStatistics   (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
    virtual bool            IsConnected     ();
    virtual bool            IsConnected     (const OpenNet::System & aSystem);
    virtual OpenNet::Status Packet_Send     (const void * aData, unsigned int aSize_byte);
    virtual OpenNet::Status ResetInputFilter();
    virtual OpenNet::Status ResetProcessor  ();
    virtual OpenNet::Status ResetStatistics ();
    virtual OpenNet::Status SetConfig       (const Config        & aConfig    );
    virtual OpenNet::Status SetInputFilter  (OpenNet::SourceCode * aSourceCode);
    virtual OpenNet::Status SetProcessor    (OpenNet::Processor  * aProcessor );
    virtual OpenNet::Status Display         (FILE * aOut) const;
    virtual OpenNet::Status Event_RegisterCallback(Event_Callback aCallback, void * aContext);
    virtual OpenNet::Status Read            (void  * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte);
    virtual OpenNet::Status Tx_Disable      ();
    virtual OpenNet::Status Tx_Enable       ();

    // ===== KmsLib::ThreadBase =============================================
    virtual unsigned int Run();

protected:

    Adapter_Internal(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog);

    OpenNet::Status Control(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte = NULL);

    // Threads  Apps
    //
    // SetInputFilter_Internal ==> ResetInputFilter_Internal
    virtual void ResetInputFilter_Internal() = 0;

    // Threads  Apps
    //
    // SetInputFilter_Internal ==> ResetInputFilter_Internal
    virtual void SetInputFilter_Internal  (OpenNet::Kernel * aKernel) = 0;

    // Thread  Apps
    virtual void Stop_Internal() = 0;

    // Threads  Apps
    //
    // Thread_Prepare_Internal == delete
    virtual Thread * Thread_Prepare_Internal(OpenNet::Kernel * aKernel) = 0;


    unsigned int                    mBufferCount ;
    OpenNetK::Buffer                mBuffers[OPEN_NET_BUFFER_QTY];
    Config                          mConfig      ;
    IoCtl_Connect_Out               mConnect_Out ;
    KmsLib::DebugLog              * mDebugLog    ;
    KmsLib::DriverHandle          * mHandle      ;
    Info                            mInfo        ;
    Processor_Internal            * mProcessor   ;
    OpenNet::SourceCode           * mSourceCode  ;
    unsigned int                    mStatistics[OpenNet::ADAPTER_STATS_QTY];

private:

    Buffer_Internal * GetBuffer(unsigned int aIndex);

    void Config_Update();

    void Event_Process(const OpenNetK::Event & aEvent);

    OpenNetK::Adapter_Config        mDriverConfig;
    Event_Callback                  mEvent_Callback;
    void                          * mEvent_Context ;
    char                            mName   [ 64];
    bool                            mRunning       ;
    Thread                        * mThread        ;
    
};

typedef std::vector<Adapter_Internal *> Adapter_Vector;
