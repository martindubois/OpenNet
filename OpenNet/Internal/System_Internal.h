
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/System_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <vector>

// ===== Import/Includes ====================================================
#include <KmsLib/DebugLog.h>

// ===== Includes ===========================================================
#include <OpenNet/System.h>

#ifdef _KMS_WINDOWS_
    #include <OpenNetK/Interface.h>
#endif

// ===== OpenNet ============================================================
#include "Processor_Internal.h"

class Adapter_Internal  ;

// Class
/////////////////////////////////////////////////////////////////////////////

class System_Internal : public OpenNet::System
{

public:

    System_Internal();

    virtual ~System_Internal();

    // ===== OpenNet::System ================================================

    virtual OpenNet::Status   GetConfig     (OpenNet::System::Config * aOut) const;
    virtual OpenNet::Status   GetInfo       (OpenNet::System::Info   * aOut) const;

    virtual OpenNet::Status SetConfig(const OpenNet::System::Config & aConfig);

    virtual OpenNet::Status    Adapter_Connect (OpenNet::Adapter * aAdapter);
    virtual OpenNet::Adapter * Adapter_Get     (unsigned int aIndex);
    virtual OpenNet::Adapter * Adapter_Get     (OpenNetK::Adapter_Type aType, unsigned int aIndex);
    virtual OpenNet::Adapter * Adapter_Get     (const unsigned char * aAddress, const unsigned char * aMask, const unsigned char * aMaskDiff);
    virtual unsigned int       Adapter_GetCount() const;

    virtual OpenNet::Status Display(FILE * aOut);

    virtual OpenNet::Kernel * Kernel_Get     (unsigned int aIndex);
    virtual unsigned int      Kernel_GetCount() const;

    virtual OpenNet::Processor * Processor_Get     (unsigned int aIndex);
    virtual unsigned int         Processor_GetCount() const;

    virtual OpenNet::Status Start(unsigned int aFlags);
    virtual OpenNet::Status Stop ();

    // ===== OpenNet::StatisticsProvider ====================================
    virtual OpenNet::Status GetStatistics  (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
    virtual OpenNet::Status ResetStatistics();

// internal

    typedef enum
    {
        STATE_IDLE   ,
        STATE_RUNNING,

        STATE_QTY
    }
    State;

    void SendLoopBackPackets(Adapter_Internal * aAdapter);

protected:

    void Cleanup();

    typedef std::vector<Processor_Internal *> ProcessorVector;

    Adapter_Vector    mAdapters   ;
    IoCtl_Connect_In  mConnect_In ;
    IoCtl_Connect_Out mConnect_Out;
    KmsLib::DebugLog  mDebugLog   ;
    Info              mInfo       ;
    ProcessorVector   mProcessors ;

private:

    typedef std::vector<Thread             *> ThreadVector   ;

    void SetPacketSize(unsigned int aSize_byte);

    OpenNet::Status Adapter_Validate(OpenNet::Adapter * aAdapter);

    OpenNet::Status Config_Apply   (const Config & aConfig);
    OpenNet::Status Config_Validate(const Config & aConfig);

    void Threads_Release();
    
    Config           mConfig    ;
    unsigned int     mStartFlags;
    State            mState     ;
    ThreadVector     mThreads   ;
    
};
