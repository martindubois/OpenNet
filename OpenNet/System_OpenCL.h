
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/System_OpenCL.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "System_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class System_OpenCL : public System_Internal
{

public:

    System_OpenCL();

    // ====== System_Internal ===============================================

    virtual ~System_OpenCL();

private:

    void FindAdapters  ();
    void FindPlatform  ();
    void FindProcessors();

    bool IsExtensionSupported(cl_device_id aDevice);

    cl_platform_id   mPlatform  ;

};
