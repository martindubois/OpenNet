
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
#include <OpenNetK/Adapter_Types.h>

class Buffer_Data     ;
class Thread          ;
class Thread_Functions;

// Class
/////////////////////////////////////////////////////////////////////////////

class Processor_Internal : public OpenNet::Processor
{

public:

    Processor_Internal( KmsLib::DebugLog * aDebugLog );

    virtual ~Processor_Internal();

    Thread           * Thread_Prepare();
    void               Thread_Release();

    // Return  The adresse of the newly created instance
    //
    // Threads  Apps
    //
    // Thread_Get ==> delete
    virtual Thread_Functions * Thread_Get() = 0;

    // ===== OpenNet::Processor =============================================
    virtual OpenNet::Status GetConfig(      Config * aOut   ) const;
    virtual OpenNet::Status GetInfo  (      Info   * aOut   ) const;
    virtual const char    * GetName  () const;
    virtual OpenNet::Status SetConfig(const Config & aConfig);
    virtual OpenNet::Status Display  (      FILE   * aOut   ) const;

    // ===== OpenNet::StatisticsProvider ====================================
    virtual OpenNet::Status GetStatistics  (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
    virtual OpenNet::Status ResetStatistics();

protected:

    Config             mConfig  ;
    KmsLib::DebugLog * mDebugLog;
    Info               mInfo    ;
    Thread_Functions * mThread  ;

};
