
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Adapter_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <vector>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Import/Includes ====================================================
#include <KmsLib/DebugLog.h>
#include <KmsLib/Windows/DriverHandle.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/System.h>
#include <OpenNetK/Constants.h>

// ===== Common =============================================================
#include "../Common/IoCtl.h"
#include "../Common/OpenNet/Adapter_Statistics.h"

// ===== OpenNet ============================================================
#include "Buffer_Data.h"
#include "Filter_Data.h"
#include "Processor_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Adapter_Internal : public OpenNet::Adapter
{

public:

    typedef void(*TryToSolveHang)(void *, Adapter_Internal *);

    Adapter_Internal(KmsLib::Windows::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog);

    virtual ~Adapter_Internal();

    void Connect(IoCtl_Connect_In * aConnect);

    void SendLoopBackPackets();

    void Start       ();
    void Stop_Request();
    void Stop_Wait   (TryToSolveHang aTryToSolveHang, void * aContext);

    // ===== OpenNet::Adapter ===============================================

    virtual OpenNet::Status GetAdapterNo    (unsigned int * aOut);
    virtual OpenNet::Status GetConfig       (Config       * aOut) const;
    virtual OpenNet::Status GetInfo         (Info         * aOut) const;
    virtual const char    * GetName         () const;
    virtual unsigned int    GetPacketSize   () const;
    virtual OpenNet::Status GetState        (State        * aOut);
    virtual OpenNet::Status GetStatistics   (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
    virtual bool            IsConnected     ();
    virtual bool            IsConnected     (const OpenNet::System & aSystem);
    virtual OpenNet::Status ResetInputFilter();
    virtual OpenNet::Status ResetProcessor  ();
    virtual OpenNet::Status ResetStatistics ();
    virtual OpenNet::Status SetConfig       (const Config       & aConfig   );
    virtual OpenNet::Status SetInputFilter  (OpenNet::Filter    * aFilter   );
    virtual OpenNet::Status SetPacketSize   (unsigned int         aSize_byte);
    virtual OpenNet::Status SetProcessor    (OpenNet::Processor * aProcessor);

    virtual OpenNet::Status Display(FILE * aOut) const;

    virtual OpenNet::Status Packet_Send(const void * aData, unsigned int aSize_byte);

// internal:

    void Run();

private:

    void Buffers_Allocate();
    void Buffers_Release ();

    void Buffer_Allocate();
    void Buffer_Release ();

    OpenNet::Status Control(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte = NULL);

    void Run_Iteration(unsigned int aIndex);
    void Run_Loop     ();
    void Run_Wait     ();

    void State_Change(unsigned int aFrom, unsigned int aTo);

    void Stop_Request_Zone0();
    void Stop_Wait_Zone0   (TryToSolveHang aTryToSolveHang, void * aContext);

    unsigned int                    mBufferCount ;
    Buffer_Data                     mBufferData[OPEN_NET_BUFFER_QTY];
    OpenNetK::Buffer                mBuffers   [OPEN_NET_BUFFER_QTY];
    Config                          mConfig      ;
    KmsLib::DebugLog              * mDebugLog    ;
    OpenNetK::Adapter_Config        mDriverConfig;
    OpenNet::Filter               * mFilter      ;
    Filter_Data                     mFilterData  ;
    KmsLib::Windows::DriverHandle * mHandle      ;
    Info                            mInfo        ;
    char                            mName    [64];
    Processor_Internal            * mProcessor   ;
    unsigned int                    mStatistics[OpenNet::ADAPTER_STATS_QTY];
    HANDLE                          mThread      ;
    DWORD                           mThreadId    ;

    // ===== Zone 0 =========================================================
    // Threads  Apps
    //          Worker
    CRITICAL_SECTION mZone0;

    unsigned int mState;

};

typedef std::vector<Adapter_Internal *> Adapter_Vector;
