
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/System_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <vector>

// ===== Import/Includes ====================================================
#include <KmsLib/DebugLog.h>

// ===== Includes ===========================================================
#include <OpenNet/System.h>
#include <OpenNetK/Interface.h>

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

    virtual unsigned int GetSystemId() const;

    virtual OpenNet::Status SetPacketSize(unsigned int aSize_byte);

    virtual OpenNet::Status    Adapter_Connect (OpenNet::Adapter * aAdapter);
    virtual OpenNet::Adapter * Adapter_Get     (unsigned int aIndex);
    virtual unsigned int       Adapter_GetCount() const;

    virtual OpenNet::Status Display(FILE * aOut);

    virtual OpenNet::Processor * Processor_Get     (unsigned int aIndex);
    virtual unsigned int         Processor_GetCount() const;

    virtual OpenNet::Status Start();
    virtual OpenNet::Status Stop (unsigned int aFlags);

// internal

    void SendLoopBackPackets(Adapter_Internal * aAdapter);

private:

    typedef std::vector<Adapter_Internal   *> AdapterVector  ;
    typedef std::vector<Processor_Internal *> ProcessorVector;

    void FindAdapters        ();
    void FindExtension       ();
    void FindPlatform        ();
    void FindProcessors      ();
    bool IsExtensionSupported(cl_device_id aDevice);

    OpenNet::Status ValidateAdapter(OpenNet::Adapter * aAdapter);

    unsigned int                           mAdapterRunning    ;
    AdapterVector                          mAdapters          ;
    IoCtl_Connect_In                       mConnect           ;
    KmsLib::DebugLog                       mDebugLog          ;
    Processor_Internal::ExtensionFunctions mExtensionFunctions;
    clEnqueueWaitSignalAMD_fn              mEnqueueWaitSignal ;
    clEnqueueWriteSignalAMD_fn             mEnqueueWriteSignal;
    unsigned int                           mPacketSize_byte   ;
    cl_platform_id                         mPlatform          ;
    ProcessorVector                        mProcessors        ;

};
