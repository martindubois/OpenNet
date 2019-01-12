
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

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/System.h>
#include <OpenNetK/Constants.h>

// ===== Common =============================================================
#include "../Common/IoCtl.h"
#include "../Common/OpenNet/Adapter_Statistics.h"

// ===== OpenNet ============================================================
#include "Buffer_Data.h"
#include "Processor_Internal.h"

class Thread;

// Class
/////////////////////////////////////////////////////////////////////////////

class Adapter_Internal : public OpenNet::Adapter
{

public:

    typedef void(*TryToSolveHang)(void *, Adapter_Internal *);

    Adapter_Internal(KmsLib::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog);

    virtual ~Adapter_Internal();

    unsigned int GetBufferQty () const;
    unsigned int GetPacketSize() const;

    void SetPacketSize(unsigned int aSize_byte);

    void Buffers_Release ();

    void Packet_Send_Ex(const IoCtl_Packet_Send_Ex_In * aIn);

    void SendLoopBackPackets();

    void Start();
    void Stop ();

    virtual void Connect(IoCtl_Connect_In * aConnect) = 0;

    virtual Thread * Thread_Prepare() = 0;

    // ===== OpenNet::Adapter ===============================================

    virtual OpenNet::Status GetAdapterNo    (unsigned int * aOut);
    virtual OpenNet::Status GetConfig       (Config       * aOut) const;
    virtual OpenNet::Status GetInfo         (Info         * aOut) const;
    virtual const char    * GetName         () const;
    virtual OpenNet::Status GetState        (State        * aOut);
    virtual OpenNet::Status GetStatistics   (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
    virtual bool            IsConnected     ();
    virtual bool            IsConnected     (const OpenNet::System & aSystem);
    virtual OpenNet::Status ResetInputFilter();
    virtual OpenNet::Status ResetProcessor  ();
    virtual OpenNet::Status ResetStatistics ();
    virtual OpenNet::Status SetConfig       (const Config        & aConfig    );
    virtual OpenNet::Status SetInputFilter  (OpenNet::SourceCode * aSourceCode);
    virtual OpenNet::Status SetProcessor    (OpenNet::Processor  * aProcessor );

    virtual OpenNet::Status Display(FILE * aOut) const;

protected:

    KmsLib::DebugLog              * mDebugLog    ;
    KmsLib::DriverHandle          * mHandle      ;
    Info                            mInfo        ;
    Processor_Internal            * mProcessor   ;
    OpenNet::SourceCode           * mSourceCode  ;
    unsigned int                    mStatistics[OpenNet::ADAPTER_STATS_QTY];

private:

    void Config_Update();

    OpenNet::Status Control(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte = NULL);

    unsigned int                    mBufferCount ;
    OpenNetK::Buffer                mBuffers[OPEN_NET_BUFFER_QTY];
    Config                          mConfig      ;
    OpenNetK::Adapter_Config        mDriverConfig;
    char                            mName   [ 64];
    
};

typedef std::vector<Adapter_Internal *> Adapter_Vector;
