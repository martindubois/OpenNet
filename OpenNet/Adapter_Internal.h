
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Adapter_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Windows ============================================================
#include <Windows.h>

// ===== Import/Includes ====================================================
#include <KmsLib/DebugLog.h>
#include <KmsLib/Windows/DriverHandle.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/System.h>
#include <OpenNetK/Constants.h>

// ===== OpenNet ============================================================
#include "Processor_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Adapter_Internal : public OpenNet::Adapter
{

public:

    typedef void(*TryToSolveHang)(void *, Adapter_Internal *);

    Adapter_Internal(KmsLib::Windows::DriverHandle * aHandle, KmsLib::DebugLog * aDebugLog);

    virtual ~Adapter_Internal();

    void Connect(OpenNet_Connect * aConnect);

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
    virtual OpenNet::Status GetStats        (Stats        * aOut, bool aReset);
    virtual bool            IsConnected     ();
    virtual bool            IsConnected     (const OpenNet::System & aSystem);
    virtual OpenNet::Status ResetInputFilter();
    virtual OpenNet::Status ResetProcessor  ();
    virtual OpenNet::Status ResetStats      ();
    virtual OpenNet::Status SetConfig       (const Config       & aConfig   );
    virtual OpenNet::Status SetInputFilter  (OpenNet::Filter    * aFilter   );
    virtual OpenNet::Status SetPacketSize   (unsigned int         aSize_byte);
    virtual OpenNet::Status SetProcessor    (OpenNet::Processor * aProcessor);

    virtual OpenNet::Status Buffer_Allocate(unsigned int aCount);
    virtual OpenNet::Status Buffer_Release (unsigned int aCount);

    virtual OpenNet::Status Display(FILE * aOut) const;

    virtual OpenNet::Status Packet_Send(const void * aData, unsigned int aSize_byte);

// internal:

    void Run();

private:

    void Buffer_Allocate();
    void Buffer_Release ();

    OpenNet::Status Control(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte);

    void Run_Iteration(unsigned int aIndex);
    void Run_Loop     ();
    void Run_Wait     ();

    void State_Change(unsigned int aFrom, unsigned int aTo);

    void Stop_Request_Zone0();
    void Stop_Wait_Zone0   (TryToSolveHang aTryToSolveHang, void * aContext);

    unsigned int                    mBufferCount;
    Processor_Internal::BufferData  mBufferData[OPEN_NET_BUFFER_QTY];
    OpenNet_BufferInfo              mBuffers   [OPEN_NET_BUFFER_QTY];
    Config                          mConfig     ;
    KmsLib::DebugLog              * mDebugLog   ;
    OpenNet::Filter               * mFilter     ;
    Processor_Internal::FilterData  mFilterData ;
    KmsLib::Windows::DriverHandle * mHandle     ;
    Info                            mInfo       ;
    char                            mName   [64];
    Processor_Internal            * mProcessor  ;
    Stats_Dll                       mStats      ;
    HANDLE                          mThread     ;
    DWORD                           mThreadId   ;

    // ===== Zone 0 =========================================================
    // Threads  Apps
    //          Worker
    CRITICAL_SECTION mZone0;

    unsigned int mState;

};
