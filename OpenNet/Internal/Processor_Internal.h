
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <map>

// ===== Import/Includes ====================================================
#include <KmsLib/DebugLog.h>

// ===== Includes ===========================================================
#include <OpenNet/Kernel.h>
#include <OpenNet/Processor.h>
#include <OpenNet/UserBuffer.h>
#include <OpenNetK/Adapter_Types.h>

class Buffer_Internal ;
class Thread          ;
class Thread_Functions;

// Class
/////////////////////////////////////////////////////////////////////////////

class Processor_Internal : public OpenNet::Processor
{

public:

    Processor_Internal( KmsLib::DebugLog * aDebugLog );

    Thread           * Thread_Prepare();
    void               Thread_Release();

    // Return  The adresse of the newly created instance
    //
    // Threads  Apps
    //
    // Thread_Get ==> delete
    virtual Thread_Functions * Thread_Get() = 0;

    // ===== OpenNet::Processor =============================================

    virtual ~Processor_Internal();

    virtual OpenNet::Status GetConfig(      Config * aOut   ) const;
    virtual OpenNet::Status GetInfo  (      Info   * aOut   ) const;
    virtual const char    * GetName  () const;
    virtual OpenNet::Status SetConfig(const Config & aConfig);

    virtual OpenNet::UserBuffer * AllocateUserBuffer(unsigned int aSize_byte);

    virtual OpenNet::Status Display  (      FILE   * aOut   ) const;

    // ===== OpenNet::StatisticsProvider ====================================
    virtual OpenNet::Status GetStatistics  (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
    virtual OpenNet::Status ResetStatistics();

protected:

    virtual OpenNet::UserBuffer * AllocateUserBuffer_Internal(unsigned int aSize_byte) = 0;

    Config             mConfig  ;
    KmsLib::DebugLog * mDebugLog;
    Info               mInfo    ;
    Thread_Functions * mThread  ;

};
