
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Adapter_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Windows ============================================================
#include <Windows.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Windows/DriverHandle.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>

// ===== OpenNet ============================================================
#include "Processor_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Adapter_Internal : public OpenNet::Adapter
{

public:

    Adapter_Internal(KmsLib::Windows::DriverHandle * aHandle);

    // ===== OpenNet::Adapter ===============================================

    virtual OpenNet::Status GetAdapterNo    (unsigned int * aOut);
    virtual OpenNet::Status GetConfig       (Config       * aOut) const;
    virtual OpenNet::Status GetInfo         (Info         * aOut) const;
    virtual OpenNet::Status GetState        (State        * aOut);
    virtual OpenNet::Status GetStats        (Stats        * aOut);
    virtual OpenNet::Status ResetInputFilter();
    virtual OpenNet::Status ResetProcessor  ();
    virtual OpenNet::Status ResetStats      ();
    virtual OpenNet::Status SetConfig       (const Config       & aConfig   );
    virtual OpenNet::Status SetInputFilter  (OpenNet::Filter    * aFilter   );
    virtual OpenNet::Status SetProcessor    (OpenNet::Processor * aProcessor);

    virtual OpenNet::Status Buffer_Allocate(unsigned int aCount);
    virtual OpenNet::Status Buffer_Release (unsigned int aCount);

    virtual OpenNet::Status Display(FILE * aOut) const;

    virtual OpenNet::Status Packet_Send(void * aData, unsigned int aSize_byte);

private:

    Config                          mConfig   ;
    OpenNet::Filter               * mFilter   ;
    KmsLib::Windows::DriverHandle * mHandle   ;
    Info                            mInfo     ;
    Processor_Internal            * mProcessor;

};
