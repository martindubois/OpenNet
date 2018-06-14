
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/System_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <vector>

// ===== Includes ===========================================================
#include <OpenNet/System.h>

// ===== OpenNet ============================================================
class Adapter_Internal  ;
class Processor_Internal;

// Class
/////////////////////////////////////////////////////////////////////////////

class System_Internal : public OpenNet::System
{

public:

    System_Internal();

    virtual ~System_Internal();

    // ===== OpenNet::System ================================================

    virtual OpenNet::Adapter * Adapter_Get     (unsigned int aIndex);
    virtual unsigned int       Adapter_GetCount() const;

    virtual OpenNet::Processor * Processor_Get     (unsigned int aIndex);
    virtual unsigned int         Processor_GetCount() const;

private:

    void FindAdapters        ();
    void FindExtension       ();
    void FindPlatform        ();
    void FindProcessors      ();
    bool IsExtensionSupported(cl_device_id aDevice);

    std::vector<Adapter_Internal   *>  mAdapters                 ;
    clEnqueueMakeBuffersResidentAMD_fn mEnqueueMakeBufferResident;
    clEnqueueWaitSignalAMD_fn          mEnqueueWaitSignal        ;
    cl_platform_id                     mPlatform                 ;
    std::vector<Processor_Internal *>  mProcessors               ;

};
